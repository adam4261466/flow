"""Bulk-upload finished videos to YouTube Studio with scheduled publish dates.

Usage from the GUI: press "Upload all finished videos".  The uploader:

1. Scans ``outbox/story_*`` (from the batch factory) and
   ``output/story_*/video/final_video.mp4`` (from the single-story pipeline).
2. Reads ``title.txt`` and ``description.txt`` from each story folder.
3. Uploads only videos that are not already scheduled/uploaded.
4. Schedules the remaining videos public, ``--per-day`` videos per day
   (default: 3), evenly spaced from mid-morning until ``--time``.
5. Records the result in ``output/youtube_uploads.json`` so a second click
   never creates duplicates.

Standalone CLI::

    python youtube_uploader.py --start-date 2026-11-06 --time 18:00
    python youtube_uploader.py --per-day 3 --ramp-days 2
    python youtube_uploader.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any

from utils.console import setup_console

# Studio scrapes text containing U+202F; without this a print() would abort a
# real upload with UnicodeEncodeError.
setup_console()

from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
)

from ai.chrome_cdp import CDP_URL, ChromeSession


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
OUTBOX_ROOT = PROJECT_ROOT / "outbox"
MANIFEST_PATH = OUTPUT_ROOT / "youtube_uploads.json"
DEBUG_DIR = OUTPUT_ROOT / "_youtube_debug"
LOCK_PATH = OUTPUT_ROOT / "youtube_upload.lock"

STORY_PATTERN = re.compile(r"^story_(\d+)$", re.IGNORECASE)

DEFAULT_PUBLISH_TIME = "18:00"
DEFAULT_INTERVAL_DAYS = 1
DEFAULT_VIDEOS_PER_DAY = 3
DEFAULT_RAMP_DAYS = 0

DONE_STATUSES = {"scheduled", "uploaded", "published"}

# YouTube Studio renders the real Create button as
# <button aria-label="Create"> inside <ytcp-button id="create-icon">.
# The generic "aria-label*=Create" form also matches "Create post", which
# opens an entirely different dialog, so keep this list exact.
CREATE_BUTTON_SELECTORS = (
    'button[aria-label="Create"]',
    'ytcp-button#create-icon button',
    'ytcp-button#create-icon',
    '#create-icon',
)

UPLOAD_MENU_ITEM_SELECTORS = (
    'tp-yt-paper-item#text-item-0',
    'tp-yt-paper-item[role="menuitem"]:has-text("Upload videos")',
    'ytcp-ve:has-text("Upload videos")',
    'text=Upload videos',
)

# New Studio UI (2025/2026): the Next control is a native <button> nested
# inside <ytcp-button id="next-button">; clicking the wrapper itself is
# unreliable while the dialog is saving.
NEXT_BUTTON_SELECTORS = (
    'ytcp-button#next-button button',
    '#next-button button',
    "ytcp-uploads-dialog button[aria-label='Next']",
    '#next-button',
    "ytcp-button:has-text('Next')",
)

# The final action button is #done-button; its label changes from
# "Publish" to "Schedule" once a publish date is chosen.
DONE_BUTTON_SELECTORS = (
    '#done-button button',
    'ytcp-button#done-button button',
    '#done-button',
    'ytcp-button#done-button',
)

VISIBILITY_CONTAINER_SELECTORS = (
    '#visibility-container',
    'ytcp-video-visibility-select',
)

PUBLIC_RADIO_SELECTORS = (
    "tp-yt-paper-radio-button[name='PUBLIC']",
    "ytcp-radio-button[name='PUBLIC']",
    "tp-yt-paper-radio-button:has-text('Public')",
)

SCHEDULE_EXPAND_SELECTORS = (
    '#second-container-expand-button',
    "#visibility-container ytcp-icon-button#second-container-expand-button",
)

DATETIME_PICKER_SELECTORS = (
    '#publish-from-private-non-sponsor-selector ytcp-datetime-picker',
    '#visibility-container ytcp-datetime-picker',
    '#second-container ytcp-datetime-picker',
    'ytcp-datetime-picker',
)

UPLOAD_LIMIT_MARKERS = (
    "daily upload limit",
    "upload limit reached",
    "reached your daily upload limit",
    "upload more videos daily",
)

YOUTUBE_TITLE_LIMIT = 100
YOUTUBE_DESCRIPTION_LIMIT = 5000


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------

class YouTubeUploadError(RuntimeError):
    """Base error for YouTube browser automation."""


class YouTubeLoginRequired(YouTubeUploadError):
    """The automation Chrome is not signed in to YouTube Studio."""


# ----------------------------------------------------------------------
# Single-instance lock
# ----------------------------------------------------------------------

class UploadLock:
    """Stop two uploader processes from driving the same Studio tab.

    The batch factory uploads after every finished story while the Windows
    scheduled tasks also run ``run_upload.bat`` at fixed times.  Both attach to
    the same Chrome tab, so a second process used to navigate (refresh) the
    page while the first upload was still in flight, which made YouTube show an
    upload error.  Only the lock holder is allowed to touch Studio.
    """

    def __init__(self, path: str | Path = LOCK_PATH, stale_seconds: float = 6 * 3600):
        self.path = Path(path)
        self.stale_seconds = max(60.0, float(stale_seconds))
        self.acquired = False

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def peek(self) -> dict[str, Any]:
        return self._read()

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                handle = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                info = self._read()
                pid = int(info.get("pid") or 0)
                started = float(info.get("started") or 0.0)
                stale = (
                    not self._pid_alive(pid)
                    or (time.time() - started) > self.stale_seconds
                )
                if not stale:
                    return False
                # Remove a lock left behind by a crashed/killed uploader.
                try:
                    self.path.unlink()
                except OSError:
                    pass
                continue
            else:
                with os.fdopen(handle, "w", encoding="utf-8") as writer:
                    json.dump(
                        {
                            "pid": os.getpid(),
                            "started": time.time(),
                            "started_at": datetime.now().isoformat(
                                sep=" ", timespec="seconds"
                            ),
                        },
                        writer,
                    )
                self.acquired = True
                return True
        return False

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            if int(self._read().get("pid") or 0) == os.getpid():
                self.path.unlink()
        except OSError:
            pass
        self.acquired = False

    def __enter__(self) -> "UploadLock":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


# ----------------------------------------------------------------------
# Planning helpers (pure, unit-testable)
# ----------------------------------------------------------------------

@dataclass
class UploadItem:
    story: str
    video: Path
    title: str
    description: str
    thumbnail: Path | None = None
    scheduled_at: datetime | None = None
    source: str = "output"


def _metadata_for_story(
    story: str,
    folder: Path,
    output_root: Path,
) -> tuple[str, str]:
    """Read title/description, falling back to the pipeline's output folder."""
    title = _read_text(folder / "title.txt").strip()
    description = _read_text(folder / "description.txt").strip()

    if not title or not description:
        source_folder = output_root / story
        if source_folder != folder and source_folder.is_dir():
            title = title or _read_text(source_folder / "title.txt").strip()
            description = description or _read_text(
                source_folder / "description.txt"
            ).strip()

    return title or story, description


def discover_outbox_videos(
    outbox_root: str | Path = OUTBOX_ROOT,
    output_root: str | Path = OUTPUT_ROOT,
) -> list[UploadItem]:
    """Return finished batch-factory videos copied into ``outbox/``.

    A batch story folder looks like::

        outbox/story_1712345678/video.mp4
        outbox/story_1712345678/thumbnail.png
        outbox/story_1712345678/title.txt
        outbox/story_1712345678/description.txt

    Older outbox folders may not have metadata; in that case the matching
    ``output/<story>`` folder is used as a fallback.
    """
    outbox_root = Path(outbox_root)
    output_root = Path(output_root)
    if not outbox_root.is_dir():
        return []

    items: list[UploadItem] = []
    for folder in outbox_root.iterdir():
        if not folder.is_dir() or not STORY_PATTERN.match(folder.name):
            continue

        video = folder / "video.mp4"
        try:
            if not video.is_file() or video.stat().st_size <= 0:
                continue
        except OSError:
            continue

        title, description = _metadata_for_story(
            folder.name,
            folder,
            output_root,
        )
        thumbnail = folder / "thumbnail.png"
        if not thumbnail.is_file():
            thumbnail = output_root / folder.name / "thumbnail" / "thumbnail.png"
        if not thumbnail.is_file():
            thumbnail = None

        items.append(
            UploadItem(
                story=folder.name,
                video=video,
                title=title,
                description=description,
                thumbnail=thumbnail,
                source="outbox",
            )
        )

    items.sort(key=lambda item: _story_number(item.story))
    return items


def discover_all_videos(
    output_root: str | Path = OUTPUT_ROOT,
    outbox_root: str | Path = OUTBOX_ROOT,
    *,
    include_outbox: bool = True,
) -> list[UploadItem]:
    """Combine outbox and pipeline videos, preferring the outbox copy."""
    items = (
        discover_outbox_videos(outbox_root, output_root)
        if include_outbox
        else []
    )
    seen = {item.story for item in items}

    for item in discover_finished_videos(output_root):
        if item.story not in seen:
            items.append(item)
            seen.add(item.story)

    items.sort(key=lambda item: _story_number(item.story))
    return items


def discover_finished_videos(output_root: str | Path = OUTPUT_ROOT) -> list[UploadItem]:
    """Return every story folder that has a non-empty final video."""
    output_root = Path(output_root)
    if not output_root.is_dir():
        return []

    items: list[UploadItem] = []
    for folder in output_root.iterdir():
        if not folder.is_dir():
            continue
        if not STORY_PATTERN.match(folder.name):
            continue

        video = folder / "video" / "final_video.mp4"
        try:
            if not video.is_file() or video.stat().st_size <= 0:
                continue
        except OSError:
            continue

        title, description = _metadata_for_story(
            folder.name,
            folder,
            output_root,
        )
        thumbnail = folder / "thumbnail" / "thumbnail.png"
        if not thumbnail.is_file():
            thumbnail = None

        items.append(
            UploadItem(
                story=folder.name,
                video=video,
                title=title,
                description=description,
                thumbnail=thumbnail,
            )
        )

    items.sort(key=lambda item: _story_number(item.story))
    return items


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return ""


def _story_number(story: str) -> int:
    match = STORY_PATTERN.match(story)
    return int(match.group(1)) if match else 0


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_manifest(
    manifest: dict[str, dict[str, Any]],
    path: str | Path = MANIFEST_PATH,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_start_date(value: str | date | datetime | None) -> date:
    if value is None or value == "":
        return date.today() + timedelta(days=1)
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Invalid start date {value!r}. Use YYYY-MM-DD (for example 2026-11-06)."
    )


def parse_publish_time(value: str | dt_time | None) -> dt_time:
    if value is None or value == "":
        value = DEFAULT_PUBLISH_TIME
    if isinstance(value, dt_time):
        return value.replace(second=0, microsecond=0)
    text = str(value).strip()
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
        try:
            return datetime.strptime(text, fmt).time().replace(second=0, microsecond=0)
        except ValueError:
            continue
    raise ValueError(
        f"Invalid publish time {value!r}. Use 24-hour HH:MM (for example 18:00)."
    )


def _manifest_scheduled_dates(manifest: dict[str, dict[str, Any]]) -> set[date]:
    used: set[date] = set()
    for record in manifest.values():
        if not isinstance(record, dict):
            continue
        if record.get("status") not in DONE_STATUSES:
            continue
        value = record.get("scheduled_at") or record.get("scheduled_date")
        if not value:
            continue
        try:
            used.add(datetime.fromisoformat(str(value)).date())
            continue
        except ValueError:
            pass
        try:
            used.add(parse_start_date(str(value).split(" ")[0]))
        except ValueError:
            continue
    return used


def _manifest_scheduled_datetimes(
    manifest: dict[str, dict[str, Any]],
    publish_time: dt_time,
) -> tuple[set[datetime], set[date]]:
    """Return used datetimes plus dates recorded by older, date-only manifests."""
    used: set[datetime] = set()
    legacy_dates: set[date] = set()

    for record in manifest.values():
        if not isinstance(record, dict):
            continue
        if record.get("status") not in DONE_STATUSES:
            continue

        value = record.get("scheduled_at")
        if value:
            try:
                parsed = datetime.fromisoformat(str(value))
            except ValueError:
                parsed = None
            if parsed is not None:
                # A midnight value written by an older date-only manifest
                # still means "one video on this date".
                if parsed.time() == dt_time(0, 0):
                    legacy_dates.add(parsed.date())
                    used.add(
                        datetime.combine(parsed.date(), publish_time)
                    )
                else:
                    used.add(
                        parsed.replace(second=0, microsecond=0)
                    )
                continue

        value = record.get("scheduled_date")
        if value:
            try:
                day = parse_start_date(str(value))
            except ValueError:
                continue
            legacy_dates.add(day)
            used.add(datetime.combine(day, publish_time))

    return used, legacy_dates


def daily_publish_times(
    videos_per_day: int = 1,
    publish_time: str | dt_time = DEFAULT_PUBLISH_TIME,
) -> list[dt_time]:
    """Evenly space ``n`` daily publish times from mid-morning until ``publish_time``.

    ``publish_time`` is the *last* slot of the day:
    ``1 -> [18:00]``, ``2 -> [12:00, 18:00]``, ``3 -> [10:00, 14:00, 18:00]``.
    """
    count = max(1, int(videos_per_day))
    end = datetime.combine(date(2000, 1, 1), parse_publish_time(publish_time))
    gap = timedelta(minutes=(12 * 60) // count)
    return [
        (end - gap * (count - 1 - index)).time()
        for index in range(count)
    ]


def _videos_for_day(
    day_offset: int,
    videos_per_day: int,
    start_videos_per_day: int | None,
    ramp_days: int,
) -> int:
    target = max(1, int(videos_per_day))
    if ramp_days <= 0:
        return target
    start = max(1, int(start_videos_per_day or 1))
    if day_offset >= ramp_days:
        return target
    return min(target, start + day_offset)


def iter_publish_slots(
    used: set[datetime],
    start_date: date,
    publish_time: str | dt_time = DEFAULT_PUBLISH_TIME,
    *,
    videos_per_day: int = 1,
    interval_days: int = DEFAULT_INTERVAL_DAYS,
    now: datetime | None = None,
    ramp_days: int = 0,
    start_videos_per_day: int | None = None,
    blocked_dates: set[date] | None = None,
):
    """Yield free publish datetimes, ``videos_per_day`` at a time."""
    interval_days = max(1, int(interval_days))
    now = now or datetime.now()
    blocked_dates = blocked_dates or set()

    candidate_date = start_date
    day_offset = 0
    while True:
        count = _videos_for_day(
            day_offset,
            videos_per_day,
            start_videos_per_day,
            ramp_days,
        )
        if candidate_date not in blocked_dates:
            for slot_time in daily_publish_times(count, publish_time):
                candidate = datetime.combine(candidate_date, slot_time)
                if candidate > now and candidate not in used:
                    yield candidate
        candidate_date += timedelta(days=interval_days)
        day_offset += interval_days


def allocate_publish_slot(
    used_dates: set[date],
    start_date: date,
    publish_time: dt_time,
    *,
    interval_days: int = DEFAULT_INTERVAL_DAYS,
    now: datetime | None = None,
) -> datetime:
    """Return the first free date at ``publish_time``, one video per day."""
    interval_days = max(1, int(interval_days))
    now = now or datetime.now()

    candidate_date = start_date
    while True:
        candidate = datetime.combine(candidate_date, publish_time)
        if candidate > now and candidate_date not in used_dates:
            return candidate
        candidate_date += timedelta(days=interval_days)


def plan_uploads(
    items: list[UploadItem],
    manifest: dict[str, dict[str, Any]],
    start_date: str | date | datetime | None = None,
    publish_time: str | dt_time | None = None,
    *,
    interval_days: int = DEFAULT_INTERVAL_DAYS,
    now: datetime | None = None,
    videos_per_day: int = 1,
    ramp_days: int = 0,
    start_videos_per_day: int | None = None,
) -> list[UploadItem]:
    """Assign scheduled datetimes to every video that still needs uploading."""
    start = parse_start_date(start_date)
    publish_at = parse_publish_time(publish_time)
    used, legacy_dates = _manifest_scheduled_datetimes(manifest, publish_at)
    now = now or datetime.now()

    # Older manifests recorded only a date, meaning "one video that day".
    # Keep that promise when running in one-per-day mode.
    blocked_dates = legacy_dates if int(videos_per_day) <= 1 else set()

    slots = iter_publish_slots(
        used,
        start,
        publish_at,
        videos_per_day=videos_per_day,
        interval_days=interval_days,
        now=now,
        ramp_days=ramp_days,
        start_videos_per_day=start_videos_per_day,
        blocked_dates=blocked_dates,
    )

    plan: list[UploadItem] = []
    for item in items:
        record = manifest.get(item.story)
        if isinstance(record, dict) and record.get("status") in DONE_STATUSES:
            continue

        slot = next(slots)
        item.scheduled_at = slot
        used.add(slot)
        plan.append(item)

    return plan


def _youtube_date_text(value: datetime) -> str:
    # YouTube Studio displays dates like "Nov 6, 2026".
    return f"{value.strftime('%b')} {value.day}, {value.year}"


def _youtube_time_text(value: datetime) -> str:
    # ... and times like "6:00 PM".
    return value.strftime("%I:%M %p").lstrip("0")


# ----------------------------------------------------------------------
# Browser automation
# ----------------------------------------------------------------------

class YouTubeStudioUploader:
    """Upload and schedule videos through the real YouTube Studio UI."""

    def __init__(self, cdp_url: str = CDP_URL) -> None:
        self.session = ChromeSession(cdp_url)
        self.page = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self.page is not None and not self.page.is_closed():
            return
        self.session.connect()
        self.page = self.session.page_for(
            "https://studio.youtube.com/",
            match_substring="studio.youtube.com",
            navigate=False,
        )

    def open_studio(self, *, force: bool = False) -> None:
        if self.page is None:
            self.connect()
        assert self.page is not None

        # Never reload Studio while a file is still uploading: that cancels the
        # transfer and YouTube reports an error. If another process (or an
        # interrupted run) left an upload in flight, wait for it to finish.
        if self._upload_in_progress():
            self._wait_for_upload_transfer()

        if not force and self._studio_ready() and self._dialog() is None:
            print(f"YouTube Studio already open: {self.page.url}")
            return

        print("Opening YouTube Studio...")
        self.page.goto(
            "https://studio.youtube.com/",
            wait_until="domcontentloaded",
            timeout=60000,
        )
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=20000)
        except PlaywrightTimeoutError:
            pass
        self.page.wait_for_timeout(2500)
        self._raise_if_login_required()

        try:
            self.page.locator(
                'button[aria-label="Create"], '
                "ytcp-button#create-icon button, "
                "ytcp-button#create-icon, "
                "#create-icon"
            ).first.wait_for(state="visible", timeout=45000)
        except PlaywrightTimeoutError:
            self._debug_screenshot("studio_not_ready")
            raise YouTubeLoginRequired(
                "YouTube Studio did not finish loading. Use the 'Open Claude + "
                "YouTube login' button, sign in, and make sure the correct "
                "channel is open."
            )

        print(f"YouTube Studio ready: {self.page.url}")

    def close(self) -> None:
        self.session.close()
        self.page = None

    # ------------------------------------------------------------------
    # Small Playwright helpers
    # ------------------------------------------------------------------

    def _first_visible(self, selectors, require_enabled: bool = False, root=None):
        assert self.page is not None
        root = root or self.page
        for selector in selectors:
            try:
                matches = root.locator(selector)
                count = matches.count()
            except PlaywrightError:
                continue
            for index in range(count - 1, -1, -1):
                item = matches.nth(index)
                try:
                    if not item.is_visible():
                        continue
                    if require_enabled and not self._is_enabled(item):
                        continue
                    return item
                except PlaywrightError:
                    continue
        return None

    @staticmethod
    def _is_enabled(locator) -> bool:
        try:
            if (locator.get_attribute("aria-disabled") or "").lower() == "true":
                return False
            if locator.get_attribute("disabled") is not None:
                return False
            return locator.is_enabled()
        except PlaywrightError:
            return False

    @staticmethod
    def _normalize(text: str | None) -> str:
        return " ".join((text or "").split()).strip()

    def _body_text(self) -> str:
        assert self.page is not None
        try:
            return self._normalize(self.page.locator("body").inner_text(timeout=3000))
        except PlaywrightError:
            return ""

    def _dialog(self):
        assert self.page is not None
        return self._first_visible(
            (
                "ytcp-uploads-dialog",
                "ytcp-upload-video-dialog",
                "tp-yt-paper-dialog",
                "div[role='dialog']",
            )
        )

    def _dialog_text(self) -> str:
        dialog = self._dialog()
        if dialog is None:
            return self._body_text()
        try:
            return self._normalize(dialog.inner_text(timeout=3000))
        except PlaywrightError:
            return self._body_text()

    def _click(self, selectors, description: str, root=None, timeout: int = 20) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            target = self._first_visible(selectors, require_enabled=False, root=root)
            if target is not None:
                try:
                    target.click(timeout=6000)
                    return
                except PlaywrightError:
                    pass
            assert self.page is not None
            self.page.wait_for_timeout(500)
        raise YouTubeUploadError(f"Could not click {description}.")

    def _debug_screenshot(self, name: str) -> Path | None:
        if self.page is None:
            return None
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        path = DEBUG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.png"
        try:
            self.page.screenshot(path=str(path), full_page=False)
            print(f"Debug screenshot saved: {path}")
            return path
        except PlaywrightError:
            return None

    def _raise_if_login_required(self) -> None:
        assert self.page is not None
        url = (self.page.url or "").lower()
        if "accounts.google.com" in url or "servicelogin" in url:
            raise YouTubeLoginRequired(
                "YouTube Studio requires sign-in. Use the 'Open Claude + "
                "YouTube login' button, sign in, then upload again."
            )

        create = self._first_visible(CREATE_BUTTON_SELECTORS)
        if create is not None:
            return

        body = self._body_text().lower()
        if "sign in" in body and "studio" in body:
            raise YouTubeLoginRequired(
                "YouTube Studio is showing a sign-in screen. Use the login "
                "button, sign in, then upload again."
            )

    def _raise_if_upload_limit(self) -> None:
        """Turn YouTube's daily upload limit into a clear error."""
        text = f"{self._body_text()} {self._dialog_text()}".lower()
        if any(marker in text for marker in UPLOAD_LIMIT_MARKERS):
            raise YouTubeUploadError(
                "YouTube's daily upload limit has been reached for this "
                "channel. Verify the account at https://www.youtube.com/verify "
                "(one-time verification) or wait 24 hours, then run the bulk "
                "upload again."
            )

    def _studio_ready(self) -> bool:
        """True when the current tab already shows a usable Studio home."""
        assert self.page is not None
        if "studio.youtube.com" not in (self.page.url or ""):
            return False
        return self._first_visible(CREATE_BUTTON_SELECTORS) is not None

    def _upload_progress(self) -> tuple[bool, int | None]:
        """Return (file_transfer_still_running, percent) for the upload dialog."""
        if self.page is None:
            return False, None
        dialog = self._dialog()
        if dialog is None:
            return False, None

        percent: int | None = None
        try:
            bars = dialog.locator(
                "[role='progressbar'], ytcp-linear-progress-bar, .progress-bar"
            )
            for index in range(bars.count() - 1, -1, -1):
                bar = bars.nth(index)
                if not bar.is_visible():
                    continue
                raw = bar.get_attribute("aria-valuenow")
                if raw is not None:
                    percent = int(float(raw))
                break
        except (PlaywrightError, ValueError, TypeError):
            pass

        text = self._dialog_text().lower()
        uploading = (
            "uploading" in text
            or ("%" in text and ("left" in text or "remaining" in text))
            or (percent is not None and percent < 100)
        )
        return uploading, percent

    def _upload_in_progress(self) -> bool:
        return self._upload_progress()[0]

    def _wait_for_upload_transfer(self, timeout: int = 3600) -> None:
        """Wait for an in-flight file transfer to finish before navigating."""
        if self.page is None:
            return
        deadline = time.monotonic() + max(30, timeout)
        announced = False
        while time.monotonic() < deadline:
            uploading, percent = self._upload_progress()
            if not uploading:
                return
            if not announced:
                suffix = f" ({percent}%)" if percent is not None else ""
                print(
                    "An upload is already in progress"
                    f"{suffix}; waiting for the file transfer to finish before "
                    "touching the page."
                )
                announced = True
            self.page.wait_for_timeout(2000)
        raise YouTubeUploadError(
            "A YouTube upload has been running for too long; not refreshing "
            "the page because that would cancel it."
        )

    # ------------------------------------------------------------------
    # Upload dialog
    # ------------------------------------------------------------------

    def _open_upload_dialog(self) -> None:
        assert self.page is not None

        print("Opening the YouTube upload dialog...")
        for attempt in range(2):
            try:
                self._click(
                    CREATE_BUTTON_SELECTORS,
                    "Create",
                    timeout=45,
                )
                break
            except YouTubeUploadError:
                # If Studio is showing the daily-limit banner there is no
                # Create button to click; report the real cause, not a
                # misleading click timeout.
                self._raise_if_upload_limit()
                if attempt:
                    raise
                # A previous failed run can leave an upload dialog or overlay
                # covering Studio, which swallows the Create click. No file is
                # transferring here (upload_video waits for that first), so it
                # is safe to reopen Studio home and try once more.
                print(
                    "Create button was not clickable; reopening YouTube "
                    "Studio and retrying."
                )
                self._wait_for_upload_transfer()
                self.open_studio(force=True)
                self.page.wait_for_timeout(1500)

        self.page.wait_for_timeout(900)

        self._click(
            UPLOAD_MENU_ITEM_SELECTORS,
            "Upload videos",
        )

        try:
            self.page.locator("input[type='file']").first.wait_for(
                state="attached",
                timeout=30000,
            )
        except PlaywrightTimeoutError as exc:
            self._debug_screenshot("upload_dialog")
            raise YouTubeUploadError(
                "YouTube's upload dialog did not open. Check the Studio tab."
            ) from exc

        print("Upload dialog opened.")

    def _find_file_input(self):
        assert self.page is not None
        try:
            inputs = self.page.locator("input[type='file']")
            count = inputs.count()
        except PlaywrightError:
            return None

        fallback = None
        for index in range(count - 1, -1, -1):
            item = inputs.nth(index)
            try:
                accept = (item.get_attribute("accept") or "").lower()
                multiple = (item.get_attribute("multiple") or "").lower()
            except PlaywrightError:
                continue
            if fallback is None:
                fallback = item
            if "video" in accept or "video" in multiple or not accept:
                return item
        return fallback

    def _attach_file_via_cdp(self, selector: str, path: Path) -> bool:
        """Attach a local file to an <input type=file> through raw CDP.

        Playwright refuses to send files larger than 50 MB when the browser was
        attached with ``connect_over_cdp`` (\"Cannot transfer files larger than
        50Mb to a browser not co-located with the server\").  The videos here are
        ~77 MB, so we set the files directly with ``DOM.setFileInputFiles``.
        """
        assert self.page is not None
        try:
            cdp = self.page.context.new_cdp_session(self.page)
        except PlaywrightError as exc:
            print(f"Note: raw CDP session unavailable ({exc}).")
            return False

        try:
            cdp.send("DOM.enable")
            # DOM.requestNode() returns nodeId 0 for YouTube's inputs, so
            # resolve the node through the document tree instead.
            document = cdp.send("DOM.getDocument", {"depth": 1, "pierce": True})
            root_id = (document.get("root") or {}).get("nodeId")
            if not root_id:
                print("Note: CDP could not read the document root.")
                return False

            found = cdp.send(
                "DOM.querySelectorAll",
                {"nodeId": root_id, "selector": selector},
            )
            node_ids = found.get("nodeIds") or []
            if not node_ids:
                print(f"Note: CDP could not find {selector!r}.")
                return False

            cdp.send(
                "DOM.setFileInputFiles",
                {
                    "files": [str(path.resolve())],
                    "nodeId": node_ids[0],
                },
            )
            print(
                f"File attached through CDP: {path.name} "
                f"({path.stat().st_size} bytes)"
            )
            return True
        except Exception as exc:  # noqa: BLE001 - fall back to Playwright
            print(f"Note: direct CDP file attach failed ({exc}).")
            return False
        finally:
            try:
                cdp.detach()
            except Exception:  # noqa: BLE001
                pass

    def _set_video_file(self, video: Path) -> None:
        file_input = self._find_file_input()
        if file_input is None:
            self._debug_screenshot("no_file_input")
            raise YouTubeUploadError("Could not find YouTube's video file input.")

        print(f"Uploading file: {video}")
        attached = self._attach_file_via_cdp(
            "ytcp-uploads-dialog input[type='file']",
            video,
        )
        if not attached:
            try:
                file_input.set_input_files(str(video))
            except PlaywrightError as exc:
                raise YouTubeUploadError(
                    f"Could not attach the video file: {exc}"
                ) from exc

        # Wait for the details form (title) so upload has definitely started.
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self._dialog_text():
                if self._first_visible(("#title-textarea", "#description-textarea")) is not None:
                    print("Upload started; details form is ready.")
                    return
            assert self.page is not None
            self.page.wait_for_timeout(1000)

        self._debug_screenshot("details_timeout")
        raise YouTubeUploadError(
            "The upload did not reach the details form within 3 minutes."
        )

    def _set_editable(self, locator, text: str, description: str) -> None:
        assert self.page is not None
        try:
            locator.click(timeout=6000)
        except PlaywrightError:
            try:
                locator.click(force=True, timeout=6000)
            except PlaywrightError:
                pass

        try:
            locator.press("Control+A", timeout=5000)
            locator.press("Delete", timeout=5000)
        except PlaywrightError:
            pass

        try:
            locator.fill(text, timeout=15000)
            return
        except PlaywrightError:
            pass

        try:
            locator.evaluate(
                """(el, value) => {
                    el.innerText = value;
                    el.dispatchEvent(new InputEvent('input', {
                        bubbles: true,
                        cancelable: true,
                        inputType: 'insertText',
                        data: value,
                    }));
                    el.dispatchEvent(new Event('change', {bubbles: true}));
                }""",
                text,
            )
            return
        except PlaywrightError:
            pass

        try:
            self.page.keyboard.insert_text(text)
        except PlaywrightError as exc:
            raise YouTubeUploadError(f"Could not set {description}: {exc}") from exc

    def _set_title_and_description(self, title: str, description: str) -> None:
        assert self.page is not None
        dialog = self._dialog()

        # YouTube titles are single-line.
        title = " ".join(title.split())[:YOUTUBE_TITLE_LIMIT]
        description = description.strip()[:YOUTUBE_DESCRIPTION_LIMIT]

        title_box = self._first_visible(
            (
                "#title-textarea #textbox",
                "ytcp-social-suggestions-textbox#title-textarea #textbox",
                "#title-textarea",
                "div#textbox[contenteditable='true']",
            ),
            root=dialog,
        )
        if title_box is None:
            self._debug_screenshot("title_box")
            raise YouTubeUploadError("Could not find the YouTube title field.")
        self._set_editable(title_box, title, "title")
        print(f"Title set: {title}")

        description_box = self._first_visible(
            (
                "#description-textarea #textbox",
                "ytcp-social-suggestions-textbox#description-textarea #textbox",
                "#description-textarea",
            ),
            root=dialog,
        )
        if description_box is not None and description:
            self._set_editable(description_box, description, "description")
            print("Description set.")
        elif description_box is None:
            print("Warning: description field was not found; uploading without it.")

    def _set_thumbnail(self, thumbnail: str | Path | None) -> None:
        """Best-effort custom thumbnail upload.

        Thumbnail upload can require a verified channel, so a failure here is
        never allowed to abort the video upload.
        """
        if not thumbnail:
            return
        path = Path(thumbnail)
        if not path.is_file():
            return
        assert self.page is not None

        try:
            inputs = self.page.locator("input[type='file']")
            count = inputs.count()
        except PlaywrightError:
            return

        for index in range(count - 1, -1, -1):
            item = inputs.nth(index)
            try:
                accept = (item.get_attribute("accept") or "").lower()
            except PlaywrightError:
                continue
            if "image" not in accept:
                continue
            try:
                item.set_input_files(str(path))
                print(f"Thumbnail attached: {path.name}")
                self.page.wait_for_timeout(2500)
            except PlaywrightError as exc:
                print(
                    "Note: could not attach the thumbnail "
                    f"({exc}); continuing without it."
                )
            return

        print("Note: thumbnail input not found; uploading without a custom thumbnail.")

    def _select_not_made_for_kids(self) -> None:
        """Select "No, it's not made for kids" (required by YouTube)."""
        dialog = self._dialog()
        target = self._first_visible(
            (
                "tp-yt-paper-radio-button[name='VIDEO_MADE_FOR_KIDS_NOT_MFK']",
                "ytcp-radio-button[name='VIDEO_MADE_FOR_KIDS_NOT_MFK']",
                "tp-yt-paper-radio-button[name='NOT_MADE_FOR_KIDS']",
                "ytcp-radio-button[name='NOT_MADE_FOR_KIDS']",
            ),
            root=dialog,
        )
        if target is None:
            return
        try:
            selected = (target.get_attribute("aria-checked") or "").lower() == "true"
        except PlaywrightError:
            selected = False
        if selected:
            return
        try:
            target.click(timeout=6000)
            print("Audience set: not made for kids.")
        except PlaywrightError:
            pass

    # ------------------------------------------------------------------
    # Step navigation
    # ------------------------------------------------------------------

    @staticmethod
    def _button_label(locator) -> str:
        parts: list[str] = []
        for attribute in ("aria-label", "title"):
            try:
                value = locator.get_attribute(attribute)
            except PlaywrightError:
                value = None
            if value:
                parts.append(value)
        try:
            parts.append(locator.inner_text(timeout=1500))
        except PlaywrightError:
            pass
        return " ".join(" ".join(parts).split()).strip().lower()

    def _next_button(self):
        dialog = self._dialog()
        return self._first_visible(NEXT_BUTTON_SELECTORS, root=dialog)

    def _wait_for_next_enabled(self, timeout: int = 900) -> bool:
        assert self.page is not None
        deadline = time.monotonic() + timeout
        last_notice = 0.0
        while time.monotonic() < deadline:
            button = self._next_button()
            if button is not None and self._is_enabled(button):
                return True
            now = time.monotonic()
            if now - last_notice >= 20:
                self._raise_if_upload_limit()
                print("Waiting for YouTube to finish processing/checks...")
                last_notice = now
            self.page.wait_for_timeout(1000)
        return False

    def _click_next(self) -> None:
        if not self._wait_for_next_enabled():
            self._debug_screenshot("next_disabled")
            raise YouTubeUploadError(
                "YouTube's Next button stayed disabled. The upload may still be "
                "processing, or a required option is unanswered."
            )
        button = self._next_button()
        if button is None:
            raise YouTubeUploadError("Could not find YouTube's Next button.")
        try:
            button.click(timeout=10000)
        except PlaywrightError:
            try:
                button.click(force=True, timeout=10000)
            except PlaywrightError as exc:
                raise YouTubeUploadError("Could not click Next.") from exc

    def _visibility_visible(self) -> bool:
        assert self.page is not None
        dialog = self._dialog()
        root = dialog or self.page

        if self._first_visible(VISIBILITY_CONTAINER_SELECTORS, root=root) is not None:
            return True

        for selector in PUBLIC_RADIO_SELECTORS:
            try:
                matches = root.locator(selector)
                for index in range(matches.count()):
                    if matches.nth(index).is_visible():
                        return True
            except PlaywrightError:
                continue

        text = self._dialog_text().lower()
        return "choose when to publish" in text

    def _wait_for_step_change(self, before: str, timeout: int = 45) -> bool:
        """Wait until the dialog content changes after clicking Next."""
        assert self.page is not None
        deadline = time.monotonic() + timeout
        marker = before[:200]
        while time.monotonic() < deadline:
            if self._visibility_visible():
                return True
            if self._dialog_text()[:200] != marker:
                return True
            self.page.wait_for_timeout(500)
        return False

    def _advance_to_visibility(self) -> None:
        assert self.page is not None

        for _ in range(10):
            if self._visibility_visible():
                return

            self._raise_if_upload_limit()
            if self._dialog() is None:
                self._raise_if_upload_limit()
                raise YouTubeUploadError(
                    "The YouTube upload dialog closed unexpectedly before the "
                    "Visibility step. Check the Studio tab."
                )

            before = self._dialog_text()
            text = before.lower()
            if "checking" in text or "checks in progress" in text:
                print("YouTube is checking the video...")

            if not self._wait_for_next_enabled(timeout=300):
                self._raise_if_upload_limit()
                self._debug_screenshot("next_disabled")
                raise YouTubeUploadError(
                    "YouTube's Next button stayed disabled. The upload may still "
                    "be processing, or a required option is unanswered."
                )

            self._click_next()
            # A click while YouTube shows "Saving..." is ignored. If the step
            # did not change, loop and click again once it is enabled.
            self._wait_for_step_change(before, timeout=45)

            if self._visibility_visible():
                return

        self._debug_screenshot("visibility_step")
        raise YouTubeUploadError(
            "Could not reach YouTube's Visibility/Schedule step."
        )

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _select_schedule(self) -> None:
        assert self.page is not None
        dialog = self._dialog()
        root = dialog or self.page

        # The 2025/2026 Studio UI no longer has a "Schedule" radio button.
        # Visibility is Private / Unlisted / Public; choosing Public reveals a
        # collapsible "Schedule" box underneath the radio group.
        self._click(PUBLIC_RADIO_SELECTORS, "Public visibility", root=root)
        self.page.wait_for_timeout(800)

        if not self._schedule_visible():
            self._click(
                SCHEDULE_EXPAND_SELECTORS,
                "the Schedule box",
                root=root,
                timeout=10,
            )
            self.page.wait_for_timeout(1200)

        if not self._wait_for_schedule_visible(timeout=20):
            self._debug_screenshot("schedule_panel")
            raise YouTubeUploadError(
                "YouTube's Schedule section did not open after choosing Public."
            )

        print("Visibility set: Public, scheduled.")

    def _find_datetime_picker(self):
        assert self.page is not None
        dialog = self._dialog()
        root = dialog or self.page
        return self._first_visible(DATETIME_PICKER_SELECTORS, root=root)

    def _schedule_visible(self) -> bool:
        picker = self._find_datetime_picker()
        if picker is None:
            return False
        try:
            return picker.is_visible()
        except PlaywrightError:
            return False

    def _wait_for_schedule_visible(self, timeout: int = 20) -> bool:
        assert self.page is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._schedule_visible():
                return True
            self.page.wait_for_timeout(500)
        return False

    def _date_trigger(self):
        picker = self._find_datetime_picker()
        if picker is None:
            return None
        return self._first_visible(("#datepicker-trigger",), root=picker)

    def _date_trigger_text(self) -> str:
        trigger = self._date_trigger()
        if trigger is None:
            return ""
        try:
            return self._normalize(trigger.inner_text(timeout=2000))
        except PlaywrightError:
            return ""

    @staticmethod
    def _date_matches(text: str, value: datetime) -> bool:
        normalized = " ".join(
            text.replace("\u202f", " ").replace("\xa0", " ").split()
        ).lower()
        if not normalized:
            return False
        month = value.strftime("%b").lower()
        full_month = value.strftime("%B").lower()
        if str(value.year) not in normalized:
            return False
        if month not in normalized and full_month not in normalized:
            return False
        return re.search(rf"\b{value.day}\b", normalized) is not None

    @staticmethod
    def _time_matches(text: str, value: datetime) -> bool:
        normalized = (
            text.replace("\u202f", " ").replace("\xa0", " ").strip().upper()
        )
        for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M"):
            try:
                parsed = datetime.strptime(normalized, fmt)
            except ValueError:
                continue
            return parsed.hour == value.hour and parsed.minute == value.minute
        return False

    @staticmethod
    def _is_visible(locator) -> bool:
        try:
            return locator.is_visible()
        except PlaywrightError:
            return False

    def _date_picker_input(self):
        assert self.page is not None
        for selector in (
            "ytcp-date-picker input",
            "ytcp-datepicker input",
            "[class*='date-picker'] input",
        ):
            try:
                matches = self.page.locator(selector)
                for index in range(matches.count() - 1, -1, -1):
                    item = matches.nth(index)
                    if item.is_visible():
                        return item
            except PlaywrightError:
                continue
        return None

    def _wait_for_date_input(self, timeout: int = 8):
        assert self.page is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            item = self._date_picker_input()
            if item is not None:
                return item
            self.page.wait_for_timeout(400)
        return None

    def _find_time_input(self):
        picker = self._find_datetime_picker()
        if picker is None:
            return None
        try:
            inputs = picker.locator("input")
            for index in range(inputs.count() - 1, -1, -1):
                item = inputs.nth(index)
                if item.is_visible():
                    return item
        except PlaywrightError:
            return None
        return None

    def _click_calendar_day(self, scheduled_at: datetime) -> bool:
        """Fallback for the date field: navigate the calendar and click a day."""
        assert self.page is not None
        picker = self._first_visible(
            (
                "ytcp-date-picker",
                "ytcp-datepicker",
                "tp-yt-iron-dropdown ytcp-date-picker",
            )
        )
        if picker is None:
            return False

        target_month = scheduled_at.strftime("%B %Y").lower()
        for _ in range(24):
            try:
                text = self._normalize(picker.inner_text(timeout=2000)).lower()
            except PlaywrightError:
                text = ""
            if target_month in text:
                break
            moved = False
            for selector in (
                "ytcp-icon-button[aria-label*='next month' i]",
                "ytcp-icon-button[aria-label*='next' i]",
                "button[aria-label*='next month' i]",
                "#next-month",
            ):
                try:
                    matches = picker.locator(selector)
                    for index in range(matches.count() - 1, -1, -1):
                        item = matches.nth(index)
                        if item.is_visible():
                            item.click(timeout=3000)
                            moved = True
                            break
                except PlaywrightError:
                    continue
                if moved:
                    break
            if not moved:
                break
            self.page.wait_for_timeout(500)

        labels = (
            f"{scheduled_at.strftime('%B')} {scheduled_at.day}, {scheduled_at.year}",
            f"{scheduled_at.strftime('%b')} {scheduled_at.day}, {scheduled_at.year}",
            f"{scheduled_at.strftime('%B')} {scheduled_at.day} {scheduled_at.year}",
        )
        for label in labels:
            try:
                matches = self.page.locator(f'[aria-label="{label}"]')
                for index in range(matches.count() - 1, -1, -1):
                    item = matches.nth(index)
                    if item.is_visible():
                        item.click(timeout=5000)
                        self.page.wait_for_timeout(700)
                        return True
            except PlaywrightError:
                continue
        return False

    def _set_schedule_date(self, scheduled_at: datetime) -> bool:
        trigger = self._date_trigger()
        if trigger is None:
            return False

        print(f"Schedule date: {_youtube_date_text(scheduled_at)}")
        try:
            trigger.click(timeout=8000)
        except PlaywrightError:
            return False

        date_input = self._wait_for_date_input(timeout=8)
        formats = (
            _youtube_date_text(scheduled_at),
            scheduled_at.strftime("%m/%d/%Y"),
            scheduled_at.strftime("%Y-%m-%d"),
        )
        if date_input is not None:
            for text in formats:
                if not self._is_visible(date_input):
                    # Some formats close the dropdown. Re-open it and try the
                    # next format rather than silently giving up.
                    try:
                        trigger.click(timeout=5000)
                    except PlaywrightError:
                        pass
                    date_input = self._wait_for_date_input(timeout=5)
                    if date_input is None:
                        break
                self._fill_datetime_input(date_input, text)
                if self._date_matches(self._date_trigger_text(), scheduled_at):
                    return True

        if self._click_calendar_day(scheduled_at):
            if self._date_matches(self._date_trigger_text(), scheduled_at):
                return True

        return self._date_matches(self._date_trigger_text(), scheduled_at)

    def _fill_datetime_input(self, locator, text: str) -> None:
        assert self.page is not None
        try:
            locator.click(timeout=6000)
            locator.press("Control+A", timeout=4000)
            locator.press("Delete", timeout=4000)
        except PlaywrightError:
            pass

        try:
            locator.fill(text, timeout=8000)
        except PlaywrightError:
            try:
                locator.type(text, delay=40, timeout=15000)
            except PlaywrightError:
                try:
                    self.page.keyboard.type(text, delay=40)
                except PlaywrightError:
                    pass

        try:
            locator.press("Enter", timeout=3000)
        except PlaywrightError:
            pass
        self.page.wait_for_timeout(500)

    def _set_schedule_datetime(self, scheduled_at: datetime) -> None:
        if not self._set_schedule_date(scheduled_at):
            self._debug_screenshot("datetime_inputs")
            raise YouTubeUploadError(
                "Could not set YouTube's schedule date. Check the Studio tab."
            )

        # Commit the date and close the calendar before touching the time
        # field, otherwise the dropdown overlay swallows the click.
        try:
            self.page.keyboard.press("Tab")
        except PlaywrightError:
            pass
        self.page.wait_for_timeout(600)

        time_input = self._find_time_input()
        if time_input is None:
            self._debug_screenshot("time_input")
            raise YouTubeUploadError(
                "Could not find YouTube's schedule time field."
            )

        time_text = _youtube_time_text(scheduled_at)
        print(f"Schedule time: {time_text}")
        candidates = (
            time_text,
            scheduled_at.strftime("%I:%M%p").lstrip("0"),
            scheduled_at.strftime("%H:%M"),
        )
        for text in candidates:
            self._fill_datetime_input(time_input, text)
            try:
                value = time_input.input_value(timeout=1000)
            except PlaywrightError:
                value = ""
            if self._time_matches(value or "", scheduled_at):
                print(f"Schedule time set to: {value}")
                break
        else:
            try:
                print(f"Warning: time field kept {time_input.input_value()!r}.")
            except PlaywrightError:
                pass

        # Move focus out of the picker so the dropdown closes. Do NOT press
        # Escape: in Studio that can close the whole upload dialog.
        try:
            self.page.keyboard.press("Tab")
        except PlaywrightError:
            pass
        self.page.wait_for_timeout(700)

        current_date = self._date_trigger_text()
        if current_date:
            print(f"Schedule date set to: {current_date}")

    def _click_schedule_button(self) -> None:
        assert self.page is not None
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            button = self._first_visible(
                DONE_BUTTON_SELECTORS,
                require_enabled=True,
                root=self._dialog(),
            )
            if button is not None:
                label = self._button_label(button)
                if "schedule" in label:
                    try:
                        button.click(timeout=8000)
                        print("Clicked Schedule.")
                        return
                    except PlaywrightError:
                        pass
                elif "publish" in label:
                    # Never accept "Publish" here: that would make the video
                    # public immediately instead of scheduling it.
                    print(
                        "Waiting for YouTube to switch the action button "
                        "from Publish to Schedule..."
                    )
            self.page.wait_for_timeout(500)

        self._debug_screenshot("schedule_button")
        raise YouTubeUploadError(
            "Could not confirm the schedule. Check that the date is valid and "
            "in the future."
        )

    def _wait_for_scheduled(self) -> str | None:
        assert self.page is not None
        deadline = time.monotonic() + 90
        last_url = self.page.url or ""

        while time.monotonic() < deadline:
            self._raise_if_upload_limit()
            body = self._body_text().lower()
            dialog = self._dialog()
            dialog_visible = False
            if dialog is not None:
                try:
                    dialog_visible = dialog.is_visible()
                except PlaywrightError:
                    dialog_visible = False

            if "video scheduled" in body or "scheduled for" in body:
                print("YouTube confirmed: video scheduled.")
                return self._extract_video_id()

            # Studio sometimes confirms with snackbar wording such as
            # "Your video will be set to Public on Nov 7, 2026 6:00 PM after
            # the standard definition (SD) finishes processing" instead of
            # the literal words "video scheduled".
            schedule_confirmation_markers = (
                "will be set to public on",
                "will be set to private on",
                "will be set to unlisted on",
                "after the standard definition",
            )
            if any(
                marker in body
                for marker in schedule_confirmation_markers
            ):
                print("YouTube confirmed: video scheduled.")
                return self._extract_video_id()

            error_markers = (
                "something went wrong",
                "upload failed",
                "couldn't upload",
                "an error occurred",
                "invalid date",
                "date is in the past",
            )
            if any(marker in body for marker in error_markers):
                self._debug_screenshot("schedule_error")
                raise YouTubeUploadError(
                    "YouTube reported an error while scheduling the video. "
                    "Check the Studio tab."
                )

            if not dialog_visible:
                # The dialog normally closes after a successful schedule.  Wait
                # briefly for the snackbar so an error is not misread as success.
                self.page.wait_for_timeout(2500)
                body = self._body_text().lower()
                if "scheduled" in body or "video scheduled" in body:
                    print("YouTube confirmed: video scheduled.")
                    return self._extract_video_id()
                if any(marker in body for marker in error_markers):
                    self._debug_screenshot("schedule_error")
                    raise YouTubeUploadError(
                        "YouTube reported an error while scheduling the video."
                    )
                print("Upload dialog closed; assuming the schedule was accepted.")
                return self._extract_video_id()

            current_url = self.page.url or ""
            if current_url != last_url and "/video/" in current_url:
                print(f"YouTube navigated to: {current_url}")
                return self._extract_video_id()

            self.page.wait_for_timeout(1000)

        self._debug_screenshot("schedule_confirmation")
        raise YouTubeUploadError(
            "YouTube did not confirm the schedule. The date may be invalid or "
            "the upload encountered an error."
        )

    def _extract_video_id(self) -> str | None:
        assert self.page is not None
        url = self.page.url or ""
        match = re.search(r"/video/([^/?#]+)", url)
        if match:
            return match.group(1)
        return None

    # ------------------------------------------------------------------
    # Public operation
    # ------------------------------------------------------------------

    def upload_video(
        self,
        video: str | Path,
        title: str,
        description: str,
        scheduled_at: datetime,
        *,
        thumbnail: str | Path | None = None,
    ) -> dict[str, Any]:
        video = Path(video)
        if not video.is_file() or video.stat().st_size <= 0:
            raise YouTubeUploadError(f"Video file is missing or empty: {video}")

        if scheduled_at <= datetime.now():
            raise YouTubeUploadError(
                f"Schedule datetime must be in the future: {scheduled_at}."
            )

        self.open_studio()
        self._wait_for_upload_transfer()
        self._open_upload_dialog()
        self._set_video_file(video)
        self._raise_if_upload_limit()
        self._set_title_and_description(title, description)
        self._set_thumbnail(thumbnail)
        self._select_not_made_for_kids()
        # Do not advance while the file is still transferring: YouTube can show
        # an upload error if the dialog changes underneath an active transfer.
        self._wait_for_upload_transfer()
        self._advance_to_visibility()
        self._select_schedule()
        self._set_schedule_datetime(scheduled_at)
        self._click_schedule_button()
        video_id = self._wait_for_scheduled()

        return {
            "video_id": video_id,
            "video_url": (
                f"https://youtube.com/watch?v={video_id}"
                if video_id
                else None
            ),
        }

    def _wait_for_dialog_with(self, marker: str, timeout: int = 20):
        """Return the visible <tp-yt-paper-dialog> containing ``marker``."""
        assert self.page is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            dialogs = self.page.locator("tp-yt-paper-dialog")
            for index in range(dialogs.count() - 1, -1, -1):
                dialog = dialogs.nth(index)
                try:
                    if not dialog.is_visible():
                        continue
                    text = self._normalize(dialog.inner_text(timeout=2000)).lower()
                except PlaywrightError:
                    continue
                if marker in text:
                    return dialog
            self.page.wait_for_timeout(500)
        return None

    def _first_visible_text(self, root, text: str):
        try:
            matches = root.get_by_text(text, exact=True)
            count = matches.count()
        except PlaywrightError:
            return None
        for index in range(count - 1, -1, -1):
            item = matches.nth(index)
            if self._is_visible(item):
                return item
        return None

    def schedule_existing_video(
        self,
        video_id: str,
        scheduled_at: datetime,
        *,
        expected_title: str | None = None,
    ) -> dict[str, Any]:
        """Schedule a video that already exists, without uploading it again.

        This is used for a video left behind as a draft by an interrupted
        upload. It opens the video's details page, sets Public + Schedule,
        verifies the date/time, then saves.
        """
        assert self.page is not None
        if scheduled_at <= datetime.now():
            raise YouTubeUploadError(
                f"Schedule datetime must be in the future: {scheduled_at}."
            )

        print(f"Opening existing video {video_id} ...")
        self.page.goto(
            f"https://studio.youtube.com/video/{video_id}/edit",
            wait_until="domcontentloaded",
            timeout=60000,
        )
        self.page.wait_for_timeout(5000)

        body = self._body_text().lower()
        if "sign in" in body and "studio" in body:
            raise YouTubeLoginRequired(
                "YouTube Studio is showing a sign-in screen. Sign in, then try again."
            )
        if expected_title and expected_title.lower() not in body:
            raise YouTubeUploadError(
                f"Video {video_id} does not show the expected title "
                f"{expected_title!r}. Refusing to change it."
            )

        trigger = self._first_visible(
            (
                "ytcp-video-metadata-visibility #select-button",
                "ytcp-icon-button#select-button",
                "[aria-label='Edit video visibility status']",
            ),
            require_enabled=True,
        )
        if trigger is None:
            self._debug_screenshot("draft_visibility_button")
            raise YouTubeUploadError(
                "Could not find the Visibility editor on the video page."
            )
        try:
            trigger.click(timeout=8000)
        except PlaywrightError as exc:
            raise YouTubeUploadError(
                "Could not open the Visibility editor."
            ) from exc

        dialog = self._wait_for_dialog_with("save or publish", timeout=20)
        if dialog is None:
            self._debug_screenshot("draft_visibility_dialog")
            raise YouTubeUploadError(
                "YouTube's visibility dialog did not open."
            )

        # Choose Public first: in the current Studio UI this reveals the
        # "Schedule" option. We still do not save until the date/time is set.
        public = self._first_visible(PUBLIC_RADIO_SELECTORS, root=dialog)
        if public is not None:
            try:
                public.click(timeout=6000)
                self.page.wait_for_timeout(800)
            except PlaywrightError:
                pass

        if not self._schedule_visible():
            option = self._first_visible_text(dialog, "Schedule")
            if option is not None:
                try:
                    option.click(timeout=6000)
                    self.page.wait_for_timeout(1200)
                except PlaywrightError:
                    pass

        if not self._wait_for_schedule_visible(timeout=20):
            self._debug_screenshot("draft_schedule_missing")
            raise YouTubeUploadError(
                "YouTube's schedule date/time fields did not appear."
            )

        self._set_schedule_datetime(scheduled_at)

        if not self._date_matches(self._date_trigger_text(), scheduled_at):
            raise YouTubeUploadError(
                "YouTube did not accept the requested schedule date."
            )
        time_input = self._find_time_input()
        if time_input is not None:
            try:
                time_value = time_input.input_value(timeout=1000)
            except PlaywrightError:
                time_value = ""
            if time_value and not self._time_matches(time_value, scheduled_at):
                raise YouTubeUploadError(
                    f"YouTube did not accept the schedule time ({time_value!r})."
                )

        done = self._first_visible(
            (
                "#done-button button",
                "ytcp-button#done-button button",
                "#done-button",
                "ytcp-button#done-button",
            ),
            require_enabled=True,
            root=dialog,
        )
        if done is None:
            done = self._first_visible_text(dialog, "Done")
        if done is None:
            self._debug_screenshot("draft_done_button")
            raise YouTubeUploadError("Could not find the visibility Done button.")
        try:
            done.click(timeout=8000)
        except PlaywrightError as exc:
            raise YouTubeUploadError("Could not confirm visibility.") from exc

        # The page-level visibility label must say Scheduled before we save.
        staged_text = ""
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            label = self._first_visible(
                (
                    "ytcp-video-metadata-visibility #visibility-text",
                    "ytcp-video-metadata-visibility #content",
                )
            )
            if label is not None:
                try:
                    staged_text = self._normalize(label.inner_text(timeout=1500))
                except PlaywrightError:
                    staged_text = ""
            if "schedul" in staged_text.lower():
                break
            self.page.wait_for_timeout(500)

        if "schedul" not in staged_text.lower():
            self._debug_screenshot("draft_not_scheduled")
            raise YouTubeUploadError(
                "YouTube did not stage the video as Scheduled (current: "
                f"{staged_text!r}). Refusing to save, to avoid publishing it "
                "immediately."
            )

        save = self._first_visible(
            (
                "button[aria-label='Save']",
                "ytcp-button#save-button button",
                "#save-button button",
            ),
            require_enabled=True,
        )
        if save is None:
            self._debug_screenshot("draft_save_button")
            raise YouTubeUploadError("Could not find the Save button.")
        try:
            save.click(timeout=8000)
        except PlaywrightError as exc:
            raise YouTubeUploadError("Could not save the schedule.") from exc

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            text = self._body_text().lower()
            if "video scheduled" in text or "changes saved" in text:
                break
            save = self._first_visible(
                (
                    "button[aria-label='Save']",
                    "ytcp-button#save-button button",
                )
            )
            if save is not None and not self._is_enabled(save):
                break
            self.page.wait_for_timeout(1000)

        print(
            f"Existing video {video_id} scheduled for "
            f"{scheduled_at.strftime('%Y-%m-%d %H:%M')}."
        )
        return {
            "video_id": video_id,
            "video_url": f"https://youtube.com/watch?v={video_id}",
            "scheduled_at": scheduled_at.isoformat(sep=" "),
            "status": "scheduled",
        }


# ----------------------------------------------------------------------
# Bulk operation
# ----------------------------------------------------------------------

def upload_all_finished_videos(
    *,
    output_root: str | Path = OUTPUT_ROOT,
    outbox_root: str | Path = OUTBOX_ROOT,
    include_outbox: bool = True,
    manifest_path: str | Path = MANIFEST_PATH,
    start_date: str | date | datetime | None = None,
    publish_time: str | dt_time | None = DEFAULT_PUBLISH_TIME,
    interval_days: int = DEFAULT_INTERVAL_DAYS,
    videos_per_day: int = DEFAULT_VIDEOS_PER_DAY,
    ramp_days: int = DEFAULT_RAMP_DAYS,
    start_videos_per_day: int | None = None,
    dry_run: bool = False,
    cdp_url: str = CDP_URL,
) -> dict[str, Any]:
    """Upload every finished, not-yet-scheduled video and return a summary."""
    output_root = Path(output_root)
    outbox_root = Path(outbox_root)
    manifest_path = Path(manifest_path)

    items = discover_all_videos(
        output_root,
        outbox_root,
        include_outbox=include_outbox,
    )
    manifest = load_manifest(manifest_path)
    plan = plan_uploads(
        items,
        manifest,
        start_date=start_date,
        publish_time=publish_time,
        interval_days=interval_days,
        videos_per_day=videos_per_day,
        ramp_days=ramp_days,
        start_videos_per_day=start_videos_per_day,
    )

    skipped = [
        item.story
        for item in items
        if item.story not in {planned.story for planned in plan}
    ]

    summary: dict[str, Any] = {
        "found": len(items),
        "videos_per_day": max(1, int(videos_per_day)),
        "ramp_days": max(0, int(ramp_days)),
        "planned": [
            {
                "story": item.story,
                "title": item.title,
                "scheduled_at": item.scheduled_at.isoformat(sep=" ")
                if item.scheduled_at
                else None,
                "video": str(item.video),
                "source": item.source,
            }
            for item in plan
        ],
        "uploaded": [],
        "failed": [],
        "skipped": skipped,
        "login_required": False,
        "upload_limit": False,
    }

    print()
    print("=" * 40)
    print("YOUTUBE BULK UPLOAD")
    print("=" * 40)
    daily_slots = daily_publish_times(videos_per_day, publish_time)
    print(
        "Cadence: "
        f"{max(1, int(videos_per_day))} video(s) per day at "
        + ", ".join(slot.strftime("%H:%M") for slot in daily_slots)
    )
    if ramp_days:
        print(
            f"Ramp: {max(1, int(start_videos_per_day or 1))}/day for "
            f"{ramp_days} day(s), then {max(1, int(videos_per_day))}/day."
        )
    print(f"Finished videos found: {len(items)}")
    print(f"Already scheduled/uploaded: {len(skipped)}")
    print(f"To upload now: {len(plan)}")

    if not plan:
        print("Nothing to upload.")
        return summary

    for item in plan:
        print(
            f"  - {item.story}: {item.title}  ->  "
            f"{item.scheduled_at.strftime('%Y-%m-%d %H:%M') if item.scheduled_at else '?'}"
        )

    if dry_run:
        print("Dry run: no browser actions performed.")
        return summary

    # The factory uploads after each story while the Windows scheduled tasks
    # also run run_upload.bat. Only one process may drive the shared Studio tab.
    lock = UploadLock()
    if not lock.acquire():
        holder = lock.peek()
        pid = holder.get("pid")
        print(
            "Another YouTube upload is already running"
            + (f" (pid {pid})" if pid else "")
            + "; skipping this pass. The next upload run will pick these up."
        )
        summary["skipped_locked"] = True
        return summary

    # Another uploader may have finished between planning and taking the lock.
    # Re-read the manifest and drop anything it already scheduled so we never
    # create a duplicate upload.
    manifest = load_manifest(manifest_path)
    plan = [
        item
        for item in plan
        if not (
            isinstance(manifest.get(item.story), dict)
            and manifest[item.story].get("status") in DONE_STATUSES
        )
    ]
    summary["planned"] = [
        {
            "story": item.story,
            "title": item.title,
            "scheduled_at": item.scheduled_at.isoformat(sep=" ")
            if item.scheduled_at
            else None,
            "video": str(item.video),
            "source": item.source,
        }
        for item in plan
    ]
    if not plan:
        print("Nothing left to upload; another pass already scheduled them.")
        lock.release()
        return summary

    uploader = YouTubeStudioUploader(cdp_url=cdp_url)
    try:
        uploader.connect()
        uploader.open_studio()

        for item in plan:
            assert item.scheduled_at is not None
            print()
            print("-" * 40)
            print(f"Uploading {item.story}: {item.title}")
            print(f"Scheduled for: {item.scheduled_at.strftime('%Y-%m-%d %H:%M')}")
            print("-" * 40)

            try:
                result = uploader.upload_video(
                    video=item.video,
                    title=item.title,
                    description=item.description,
                    scheduled_at=item.scheduled_at,
                    thumbnail=item.thumbnail,
                )
            except YouTubeLoginRequired as exc:
                print(f"LOGIN REQUIRED: {exc}")
                summary["login_required"] = True
                summary["failed"].append(
                    {"story": item.story, "error": str(exc)}
                )
                break
            except Exception as exc:  # noqa: BLE001 - keep uploading the rest
                print(f"Upload failed for {item.story}: {exc}")
                summary["failed"].append(
                    {"story": item.story, "error": str(exc)}
                )
                manifest[item.story] = {
                    "story": item.story,
                    "title": item.title,
                    "video": str(item.video),
                    "source": item.source,
                    "scheduled_at": item.scheduled_at.isoformat(sep=" "),
                    "status": "failed",
                    "error": str(exc),
                    "attempted_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
                }
                save_manifest(manifest, manifest_path)
                if "upload limit" in str(exc).lower():
                    summary["upload_limit"] = True
                    print(
                        "Stopping: YouTube's daily upload limit was reached. "
                        "Verify the channel or wait 24 hours."
                    )
                    break
                continue

            manifest[item.story] = {
                "story": item.story,
                "title": item.title,
                "video": str(item.video),
                "source": item.source,
                "scheduled_at": item.scheduled_at.isoformat(sep=" "),
                "scheduled_date": item.scheduled_at.date().isoformat(),
                "status": "scheduled",
                "uploaded_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
                "video_id": result.get("video_id"),
                "video_url": result.get("video_url"),
            }
            save_manifest(manifest, manifest_path)
            summary["uploaded"].append(
                {
                    "story": item.story,
                    "title": item.title,
                    "scheduled_at": item.scheduled_at.isoformat(sep=" "),
                    "video_id": result.get("video_id"),
                }
            )
            print(f"Recorded in manifest: {manifest_path}")

    finally:
        try:
            uploader.close()
        finally:
            lock.release()

    print()
    print("=" * 40)
    print("YOUTUBE UPLOAD SUMMARY")
    print("=" * 40)
    print(f"Uploaded/scheduled: {len(summary['uploaded'])}")
    print(f"Failed: {len(summary['failed'])}")
    print(f"Skipped (already done): {len(summary['skipped'])}")
    if summary["login_required"]:
        print("Sign in to YouTube Studio and run the upload again.")
    if summary["upload_limit"]:
        print(
            "YouTube's daily upload limit was reached. Verify the channel at "
            "https://www.youtube.com/verify or wait 24 hours, then run the "
            "upload again."
        )

    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk upload finished story videos to YouTube Studio.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="First publish date (YYYY-MM-DD). Default: tomorrow.",
    )
    parser.add_argument(
        "--time",
        default=DEFAULT_PUBLISH_TIME,
        help="Publish time in 24-hour format (default: 18:00).",
    )
    parser.add_argument(
        "--interval-days",
        type=int,
        default=DEFAULT_INTERVAL_DAYS,
        help="Days between each upload batch (default: 1).",
    )
    parser.add_argument(
        "--per-day",
        type=int,
        default=DEFAULT_VIDEOS_PER_DAY,
        help="Videos to schedule per day (default: 3).",
    )
    parser.add_argument(
        "--start-per-day",
        type=int,
        default=None,
        help=(
            "Videos per day on the first day of a ramp (default: same as "
            "--per-day). Use with --ramp-days."
        ),
    )
    parser.add_argument(
        "--ramp-days",
        type=int,
        default=DEFAULT_RAMP_DAYS,
        help=(
            "Ramp from --start-per-day up to --per-day over this many days "
            "(default: 0, no ramp)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the upload schedule without opening YouTube.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Pipeline story output folder (default: output/).",
    )
    parser.add_argument(
        "--outbox-root",
        default=str(OUTBOX_ROOT),
        help="Batch factory outbox folder (default: outbox/).",
    )
    parser.add_argument(
        "--no-outbox",
        action="store_true",
        help="Ignore the outbox and only scan output/.",
    )
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH),
        help="Upload manifest path.",
    )
    args = parser.parse_args()

    upload_all_finished_videos(
        output_root=args.output_root,
        outbox_root=args.outbox_root,
        include_outbox=not args.no_outbox,
        manifest_path=args.manifest,
        start_date=args.start_date,
        publish_time=args.time,
        interval_days=args.interval_days,
        videos_per_day=args.per_day,
        ramp_days=args.ramp_days,
        start_videos_per_day=args.start_per_day,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _cli()
