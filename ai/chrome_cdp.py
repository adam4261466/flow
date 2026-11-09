"""Shared Chrome/CDP plumbing for all browser automation in this project.

Every browser-driven feature (Gemini images, Claude stories, YouTube Studio)
talks to one real Chrome instance:

    http://127.0.0.1:9222

Chrome is started with the project ``chrome-profile`` so Google/Claude logins
survive between runs.  Keeping the launch and connection logic here means the
new modules can reuse the exact same session as the existing Gemini flow.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
)


CDP_URL = "http://127.0.0.1:9222"

GEMINI_URL = "https://gemini.google.com/app"
CLAUDE_URL = "https://claude.ai/new"
YOUTUBE_STUDIO_URL = "https://studio.youtube.com/"

# Pages the user should sign in to before automation runs.
LOGIN_URLS = (
    CLAUDE_URL,
    YOUTUBE_STUDIO_URL,
)


# ----------------------------------------------------------------------
# Chrome profile discovery / launch
# ----------------------------------------------------------------------

def candidate_chrome_profiles() -> list[Path]:
    """Return profile paths in priority order.

    The first entry is the profile used by the older project and the one that
    actually holds the existing Gemini login.  Project-local paths come last.
    """
    home = Path.home()
    candidates = [
        home / "Desktop" / "flow" / "chrome-profile",
        home / "OneDrive" / "Desktop" / "projects" / "flow" / "chrome-profile",
        Path.cwd() / "chrome-profile",
        Path(__file__).resolve().parent.parent / "chrome-profile",
    ]

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()).lower()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def chrome_profile() -> Path:
    """Return the profile Chrome should use (existing one when possible)."""
    candidates = candidate_chrome_profiles()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


CHROME_PROFILE = chrome_profile()


def cdp_is_ready(cdp_url: str = CDP_URL) -> bool:
    """True when Chrome is already exposing its debugging endpoint."""
    try:
        with urlopen(f"{cdp_url.rstrip('/')}/json/version", timeout=2) as response:
            return response.status == 200
    except (OSError, URLError, TimeoutError, ValueError):
        return False


def find_chrome() -> str | None:
    """Locate chrome.exe through PATH or the usual Windows install folders."""
    which = shutil.which("chrome") or shutil.which("chrome.exe")
    if which:
        return which

    local = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    program_files = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    program_files_x86 = Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))

    candidates = [
        local / "Google" / "Chrome" / "Application" / "chrome.exe",
        program_files / "Google" / "Chrome" / "Application" / "chrome.exe",
        program_files_x86 / "Google" / "Chrome" / "Application" / "chrome.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def launch_chrome(
    initial_urls: list[str] | tuple[str, ...] | None = None,
    *,
    cdp_url: str = CDP_URL,
    profile: Path | None = None,
    wait_seconds: int = 45,
) -> subprocess.Popen:
    """Launch Chrome with remote debugging enabled and wait for CDP."""
    chrome = find_chrome()
    if not chrome:
        raise FileNotFoundError(
            "Google Chrome was not found. Install Chrome or add chrome.exe to PATH."
        )

    profile = Path(profile) if profile else chrome_profile()
    profile.mkdir(parents=True, exist_ok=True)

    port = _port_from_cdp_url(cdp_url)
    command = [
        chrome,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile.resolve()}",
        "--no-first-run",
        "--no-default-browser-check",
        "--start-maximized",
    ]
    command.extend(initial_urls or ["about:blank"])

    print("Starting Chrome with remote debugging...")
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if cdp_is_ready(cdp_url):
            print("Chrome CDP is ready.")
            return process
        if process.poll() is not None:
            raise RuntimeError(
                "Chrome exited before remote debugging became ready. "
                f"Close any Chrome window using this profile ({profile}) or "
                f"check whether another process is using port {port}."
            )
        time.sleep(0.5)

    raise TimeoutError(
        f"Chrome did not expose the debugging endpoint on port {port} "
        f"within {wait_seconds} seconds."
    )


def ensure_chrome(
    initial_urls: list[str] | tuple[str, ...] | None = None,
    *,
    cdp_url: str = CDP_URL,
    profile: Path | None = None,
) -> bool:
    """Make sure Chrome/CDP is available.

    Returns True when Chrome had to be launched, False when it was already up.
    """
    if cdp_is_ready(cdp_url):
        print("Chrome CDP is already available.")
        return False
    launch_chrome(initial_urls, cdp_url=cdp_url, profile=profile)
    return True


def _port_from_cdp_url(cdp_url: str) -> int:
    try:
        return int(cdp_url.rstrip("/").rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        return 9222


# ----------------------------------------------------------------------
# Playwright connection / tab helpers
# ----------------------------------------------------------------------

def connect_over_cdp(
    cdp_url: str = CDP_URL,
) -> tuple[Playwright, Browser, BrowserContext]:
    """Connect Playwright to the running Chrome instance."""
    if not cdp_is_ready(cdp_url):
        launch_chrome(cdp_url=cdp_url)

    print("Connecting to Chrome through CDP...")
    playwright = sync_playwright().start()
    try:
        browser = playwright.chromium.connect_over_cdp(cdp_url, timeout=30000)
        contexts = browser.contexts
        if contexts:
            context = contexts[0]
        else:
            context = browser.new_context(accept_downloads=True)
        print("Chrome connection established.")
        return playwright, browser, context
    except Exception:
        try:
            playwright.stop()
        except Exception:
            pass
        raise


def find_page(context: BrowserContext, url_substring: str) -> Page | None:
    """Return an existing tab whose URL contains ``url_substring``."""
    for candidate in context.pages:
        try:
            if url_substring in (candidate.url or ""):
                return candidate
        except Exception:
            continue
    return None


def get_or_create_page(
    context: BrowserContext,
    url: str,
    *,
    match_substring: str | None = None,
) -> Page:
    """Reuse a matching tab or create a new one and navigate to ``url``."""
    page = find_page(context, match_substring or url)
    if page is None:
        page = context.new_page()
    try:
        page.bring_to_front()
    except Exception:
        pass
    page.set_default_timeout(15000)
    page.set_default_navigation_timeout(60000)
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_load_state("domcontentloaded", timeout=15000)
    except Exception:
        pass
    return page


class ChromeSession:
    """Small convenience wrapper around a CDP connection.

    Usage::

        session = ChromeSession()
        session.connect()
        page = session.page_for(CLAUDE_URL, match_substring="claude.ai")
        ...
        session.close()
    """

    def __init__(self, cdp_url: str = CDP_URL) -> None:
        self.cdp_url = cdp_url.rstrip("/")
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None

    def connect(self) -> None:
        if self.context is not None:
            return
        self.playwright, self.browser, self.context = connect_over_cdp(self.cdp_url)

    def page_for(
        self,
        url: str,
        *,
        match_substring: str | None = None,
        navigate: bool = True,
    ) -> Page:
        if self.context is None:
            self.connect()
        assert self.context is not None

        page = find_page(self.context, match_substring or url)
        if page is None:
            page = self.context.new_page()

        try:
            page.bring_to_front()
        except Exception:
            pass
        page.set_default_timeout(15000)
        page.set_default_navigation_timeout(60000)

        if navigate:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            try:
                page.wait_for_load_state("domcontentloaded", timeout=15000)
            except Exception:
                pass
        return page

    def close(self) -> None:
        """Disconnect Playwright without killing the user's Chrome."""
        try:
            if self.playwright is not None:
                self.playwright.stop()
        except Exception as exc:
            print(f"Note while disconnecting from Chrome: {exc}")
        finally:
            self.playwright = None
            self.browser = None
            self.context = None


def open_login_tabs(urls: list[str] | tuple[str, ...] = LOGIN_URLS) -> list[str]:
    """Open Claude and YouTube Studio in the automation Chrome for sign-in.

    The tabs are left open on purpose.  The user can sign in once, and every
    later browser run reuses the same profile/tabs.
    """
    session = ChromeSession()
    opened: list[str] = []
    try:
        session.connect()
        assert session.context is not None
        for url in urls:
            page = get_or_create_page(
                session.context,
                url,
                match_substring=_match_for(url),
            )
            opened.append(page.url)
            print(f"Opened login tab: {page.url}")
    finally:
        # Only disconnect Playwright.  Chrome and its tabs stay open.
        session.close()
    return opened


def _match_for(url: str) -> str:
    if "claude.ai" in url:
        return "claude.ai"
    if "studio.youtube.com" in url:
        return "studio.youtube.com"
    if "youtube.com" in url:
        return "youtube.com"
    if "gemini.google.com" in url:
        return "gemini.google.com"
    return url
