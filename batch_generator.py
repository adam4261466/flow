"""One-click batch factory: invent stories, render them, then schedule uploads.

Each prompt is written to ``input/batch/<story>.txt`` and rendered by
``main.py``.  Finished videos are copied to ``outbox/<story>/`` so the YouTube
uploader can find them without touching the live ``output/`` folders.

Examples::

    py -3.13 batch_generator.py --count 30
    py -3.13 batch_generator.py --count 30 --upload
    py -3.13 batch_generator.py --count 6 --per-day 3 --ramp-days 2 --upload
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from agents.batch_state import BatchStateManager
from agents.limits import LimitLedger, parse_reset_minutes
from agents.prompt_architect import PromptArchitect
from agents.stage_scheduler import (
    StageLayout,
    StageResult,
    StageScheduler,
    StoryJob,
)
from utils.console import setup_console
from utils.status import update_status
from ai.claude_browser import (
    ClaudeLimitError,
    generate_story_with_claude,
    story_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTBOX_DIR = PROJECT_ROOT / "outbox"
OUTPUT_DIR = PROJECT_ROOT / "output"
INPUT_DIR = PROJECT_ROOT / "input"
BATCH_INPUT_DIR = INPUT_DIR / "batch"
STATE_FILE = PROJECT_ROOT / "batch_state.json"
LEDGER_FILE = PROJECT_ROOT / "factory_state.json"

setup_console()

_RESERVED_STORY_IDS: set[str] = set()


def build_story_prompt(premise: str, *, kids: bool = False) -> str:
    """Insert an architect premise into Claude's full script-format prompt."""
    premise = premise.strip()
    base = story_prompt(kids=kids)
    if not premise:
        return base

    marker = (
        "Now generate the complete children's story script."
        if kids
        else "Now generate the complete cinematic story script."
    )
    idea = (
        "Use this premise as the core idea (expand it into a complete story):\n\n"
        f"{premise}\n\n"
    )
    if marker in base:
        return base.replace(marker, idea + marker, 1)
    return f"{base}\n\n{idea}"


def make_story_id() -> str:
    """Return a unique, sortable story folder name like ``story_1762512345``."""
    candidate = int(time.time())
    while (
        f"story_{candidate}" in _RESERVED_STORY_IDS
        or (OUTPUT_DIR / f"story_{candidate}").exists()
        or (OUTBOX_DIR / f"story_{candidate}").exists()
    ):
        candidate += 1
    story_id = f"story_{candidate}"
    _RESERVED_STORY_IDS.add(story_id)
    return story_id


# ----------------------------------------------------------------------
# Usage-limit helpers
# ----------------------------------------------------------------------

def parse_limit_minutes(message: str, *, default: float = 60.0) -> float:
    """Best-effort cooldown (minutes) parsed from a Claude limit message.

    Claude usually says "resets at 5:00 PM" or "in 2 hours". When neither is
    present we fall back to an hour so the batch still backs off instead of
    hammering the account.
    """
    return parse_reset_minutes(message, default=default)


def wait_for_resource(
    ledger: LimitLedger,
    resource: str,
    *,
    label: str,
    max_wait_minutes: float,
    announced: dict,
) -> bool:
    """Wait quietly until ``resource`` is usable again.

    The ledger is re-read from disk every iteration because ``main.py`` runs as
    a subprocess and records cooldowns in the same file.  The cooldown is
    announced once; nothing is printed again until it clears.  Returns False
    only when the wait exceeds ``max_wait_minutes``.
    """
    ledger.reload()
    remaining = ledger.remaining(resource)
    if remaining <= 0.0:
        if announced.pop(resource, None):
            print(f"{label} limit cleared. Resuming.")
        return True

    info = ledger.info(resource)
    until = info.get("blocked_until_text", "?")
    token = (info.get("last_hit_at"), until)
    if announced.get(resource) != token:
        print(
            f"{label} limit reached. Pausing for {remaining / 60.0:.0f} min "
            f"(until {until}). This message will not repeat."
        )
        announced[resource] = token

    started = time.monotonic()
    while True:
        ledger.reload()
        remaining = ledger.remaining(resource)
        if remaining <= 0.0:
            break

        wait = min(30.0, max(1.0, remaining))
        time.sleep(wait)
        if (
            max_wait_minutes > 0
            and time.monotonic() - started >= max_wait_minutes * 60.0
        ):
            print(
                f"{label} is still limited after {max_wait_minutes:.0f} min. "
                "Stopping; rerun the batch to continue."
            )
            return False

    announced.pop(resource, None)
    print(f"{label} limit cleared. Resuming.")
    return True


def block_on_limit(
    ledger: LimitLedger,
    resource: str,
    message: str,
    *,
    default_minutes: float = 60.0,
) -> float:
    """Record a limit in the shared ledger and return the cooldown."""
    minutes = parse_limit_minutes(message, default=default_minutes)
    ledger.block(resource, minutes, message)
    return minutes


def run_single_story(
    prompt: str,
    story_id: str,
    *,
    no_agents: bool = False,
    kids: bool = False,
    resume: bool = False,
    max_wait_minutes: float | None = None,
) -> bool:
    """Render one prompt through the main.py pipeline and copy it to outbox/.

    When ``resume`` is true the story folder and script file already exist, so
    the script is reused and main.py is asked to repair/continue the existing
    assets instead of starting over.
    """
    print(f"\n{'=' * 60}")
    print(f"STARTING STORY: {story_id} ({'resume' if resume else 'new'})")
    print(f"PROMPT: {prompt}")
    print(f"{'=' * 60}\n")

    BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    script_file = BATCH_INPUT_DIR / f"{story_id}.txt"

    if script_file.exists() and script_file.stat().st_size > 0:
        print(f"Reusing saved script: {script_file}")
    else:
        try:
            script_text = generate_story_with_claude(
                prompt=build_story_prompt(prompt, kids=kids)
            )
            script_file.write_text(script_text, encoding="utf-8")
        except ClaudeLimitError:
            # Let the batch loop record the cooldown and wait quietly.
            raise
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            print(f"FAILED to generate story script: {exc}")
            return False

    command = [
        sys.executable,
        "main.py",
        "--story",
        story_id,
        "--script",
        str(script_file),
        "--after-finish",
        "none",
    ]
    if no_agents:
        command.append("--no-agents")
    if kids:
        command.append("--kids")
    if resume:
        command.append("--recap")
    if max_wait_minutes is not None:
        command.extend(["--max-wait-minutes", str(max_wait_minutes)])

    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"Pipeline exited with code {result.returncode}.")
            return False
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline execution error: {exc}")
        return False

    try:
        story_output = OUTPUT_DIR / story_id
        final_video = story_output / "video" / "final_video.mp4"
        if not final_video.exists():
            print(f"No finished video at {final_video}.")
            return False

        outbox_story = OUTBOX_DIR / story_id
        outbox_story.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_video, outbox_story / "video.mp4")

        thumbnail = story_output / "thumbnail" / "thumbnail.png"
        if thumbnail.exists():
            shutil.copy2(thumbnail, outbox_story / "thumbnail.png")

        for name in ("title.txt", "description.txt"):
            source = story_output / name
            if source.exists():
                shutil.copy2(source, outbox_story / name)

        print(f"Copied to outbox: {outbox_story}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Error copying files to outbox: {exc}")
        return False


def ensure_script(
    state: BatchStateManager,
    prompt: str,
    *,
    kids: bool,
    ledger: LimitLedger,
    announced: dict,
    max_wait_minutes: float,
):
    """Return ``(story_id, script_path)`` for a prompt, writing the script if needed.

    ``script_path`` is None when Claude could not produce a script yet. The
    prompt keeps its story folder either way, so the next pass resumes it.
    """
    info = state.get_story(prompt)
    story_id = info.get("story_id") or make_story_id()
    script_file = BATCH_INPUT_DIR / f"{story_id}.txt"
    state.set_story(prompt, story_id, script_file)

    if script_file.exists() and script_file.stat().st_size > 0:
        return story_id, script_file

    while True:
        if not wait_for_resource(
            ledger,
            "claude",
            label="Claude",
            max_wait_minutes=max_wait_minutes,
            announced=announced,
        ):
            return story_id, None
        try:
            script_text = generate_story_with_claude(
                prompt=build_story_prompt(prompt, kids=kids)
            )
        except ClaudeLimitError as exc:
            minutes = block_on_limit(ledger, "claude", str(exc))
            print(
                f"Claude limit reached while writing {story_id} "
                f"(cooldown {minutes:.0f} min)."
            )
            continue
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            print(f"FAILED to generate script for {story_id}: {exc}")
            return story_id, None

        script_file.write_text(script_text, encoding="utf-8")
        return story_id, script_file


def voice_first_pass(
    state: BatchStateManager,
    *,
    kids: bool,
    no_agents: bool,
    ledger: LimitLedger,
    announced: dict,
    max_wait_minutes: float,
) -> None:
    """Voice every queued story in one continuous pass.

    For each pending prompt we write the script (if needed) and immediately run
    ``main.py --voices-only``. This keeps the TTS server busy from one story to
    the next instead of idling while images and clips render. The later render
    pass finds the WAVs already on disk and skips them.
    """
    pending = list(state.state["pending"])
    if not pending:
        return

    BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print(f"VOICE-FIRST PASS: {len(pending)} queued stories")
    print("=" * 60)

    voiced = 0
    prepared = 0

    for index, prompt in enumerate(pending, start=1):
        print()
        print(f"[{index}/{len(pending)}] Preparing script...")
        story_id, script_file = ensure_script(
            state,
            prompt,
            kids=kids,
            ledger=ledger,
            announced=announced,
            max_wait_minutes=max_wait_minutes,
        )
        if script_file is None:
            print(f"Skipping {story_id}: no script available yet.")
            continue

        prepared += 1
        print()
        print("=" * 60)
        print(f"VOICE PASS {index}/{len(pending)}: {story_id}")
        print("=" * 60)

        command = [
            sys.executable,
            "main.py",
            "--story",
            story_id,
            "--script",
            str(script_file),
            "--voices-only",
            "--after-finish",
            "none",
        ]
        if kids:
            command.append("--kids")
        if no_agents:
            command.append("--no-agents")

        try:
            # Update status for GUI polling
            update_status({
                "current_story": story_id,
                "stage": "voicing",
                "batch_id": "batch_001",
                "pending": len(pending),
                "completed": voiced,
            })
            result = subprocess.run(command, cwd=PROJECT_ROOT)
        except Exception as exc:  # noqa: BLE001 - keep voicing other stories
            print(f"Voice pass failed for {story_id}: {exc}")
            continue

        if result.returncode == 0:
            voiced += 1
        elif result.returncode == 2:
            # main.py uses exit code 2 when the TTS server itself fails.
            # Stop here instead of hammering it once per queued story.
            print(
                "TTS server failed during the voice-first pass. Stopping so "
                "the render pass can retry later."
            )
            break
        else:
            print(
                f"Voice pass for {story_id} exited with {result.returncode}; "
                "continuing with the next story."
            )

    print()
    print(
        f"Voice-first pass finished: {voiced}/{prepared} prepared stories "
        "have narration."
    )


_UPLOAD_PAUSE_ANNOUNCED: list = []


def schedule_uploads(args: argparse.Namespace) -> None:
    """Send every finished outbox/output video to the YouTube uploader."""
    from youtube_uploader import upload_all_finished_videos

    # Do not reopen YouTube after every story while the daily limit is active.
    paused = LimitLedger(LEDGER_FILE)
    if paused.is_blocked("youtube"):
        if not _UPLOAD_PAUSE_ANNOUNCED:
            _UPLOAD_PAUSE_ANNOUNCED.append(True)
            print(f"YouTube uploads paused ({paused.describe('youtube')}). "
                  "Rendering continues; uploads resume automatically after the cooldown.")
        return
    _UPLOAD_PAUSE_ANNOUNCED.clear()

    upload_all_finished_videos(
        videos_per_day=args.per_day,
        ramp_days=args.ramp_days,
        start_videos_per_day=args.start_per_day,
        start_date=args.start_date,
        publish_time=args.publish_time,
        dry_run=args.dry_run_upload,
    )


# ----------------------------------------------------------------------
# Stage-parallel factory
# ----------------------------------------------------------------------

STAGE_FLAGS = {
    "images": "--images-only",
    "voices": "--voices-only",
    "clips": "--clips-only",
    "final": "--final-only",
}

# Stages that share a single, account/process-wide resource.
STAGE_RESOURCE = {
    "images": ("gemini", "Gemini"),
    "voices": ("tts", "TTS"),
}


class FactoryStageRunner:
    """Execute one stage for one story as a ``main.py`` subprocess.

    The script stage calls Claude in-process because it already owns the Claude
    browser helper.  Every other stage is a stage-only ``main.py`` run, which
    reuses the saved script, visual plan, title and thumbnail plan.
    """

    def __init__(
        self,
        *,
        kids: bool,
        no_agents: bool,
        max_wait_minutes: float,
        no_wait: bool,
        ledger: LimitLedger,
        stop_event: threading.Event,
    ) -> None:
        self.kids = kids
        self.no_agents = no_agents
        self.max_wait_minutes = max_wait_minutes
        self.no_wait = no_wait
        self.ledger = ledger
        self.stop_event = stop_event
        self.announced: dict = {}
        self._story_locks: dict[str, threading.Lock] = {}
        self._story_locks_guard = threading.Lock()

    # -- helpers ---------------------------------------------------------

    def _story_lock(self, story_id: str) -> threading.Lock:
        with self._story_locks_guard:
            return self._story_locks.setdefault(story_id, threading.Lock())

    def _command(self, job: StoryJob, stage: str) -> list[str]:
        command = [
            sys.executable,
            "main.py",
            "--story",
            job.story_id,
            "--script",
            str(job.script),
            STAGE_FLAGS[stage],
            "--after-finish",
            "none",
        ]
        if self.kids:
            command.append("--kids")
        if self.no_agents:
            command.append("--no-agents")
        if self.no_wait:
            command.append("--no-wait")
        if self.max_wait_minutes is not None:
            command.extend(["--max-wait-minutes", str(self.max_wait_minutes)])
        return command

    def _wait_for_shared_resource(self, stage: str) -> StageResult | None:
        resource = STAGE_RESOURCE.get(stage)
        if resource is None:
            return None
        name, label = resource

        if self.no_wait:
            # ``--no-wait`` must not sit through a cooldown inside a worker.
            # Defer the stage instead; the scheduler keeps other stages busy
            # and retries this one once the ledger clears.
            self.ledger.reload()
            if self.ledger.is_blocked(name):
                info = self.ledger.info(name)
                token = (info.get("last_hit_at"), info.get("blocked_until_text"))
                if self.announced.get(name) != token:
                    print(
                        f"{label} limit reached. Not waiting (--no-wait); "
                        "this stage will retry after the cooldown."
                    )
                    self.announced[name] = token
                return StageResult.blocked_stage(f"{label} is limited")
            if self.announced.pop(name, None):
                print(f"{label} limit cleared. Resuming.")
            return None

        ok = wait_for_resource(
            self.ledger,
            name,
            label=label,
            max_wait_minutes=self.max_wait_minutes,
            announced=self.announced,
        )
        if ok:
            return None
        return StageResult.failed(
            f"{label} is still limited after "
            f"{self.max_wait_minutes:.0f} min; stopping the factory.",
            fatal=True,
        )

    # -- stage entry point ----------------------------------------------

    def __call__(self, stage: str, job: StoryJob) -> StageResult:
        if stage == "script":
            lock = self._story_lock(job.story_id)
            if not lock.acquire(blocking=False):
                return StageResult.blocked_stage(
                    "another stage is running for this story"
                )
            try:
                return self._run_script(job)
            finally:
                lock.release()

        wait_result = self._wait_for_shared_resource(stage)
        if wait_result is not None:
            return wait_result

        # Two stages must never drive the same story folder at the same time:
        # both write checkpoint.json/production.json and could collide.  The
        # scheduler releases the claim and tries another story instead of
        # stalling every stage behind this one.
        lock = self._story_lock(job.story_id)
        if not lock.acquire(blocking=False):
            return StageResult.blocked_stage(
                "another stage is running for this story"
            )
        try:
            command = self._command(job, stage)
            print()
            print("=" * 60)
            print(f"STAGE {stage.upper()}: {job.story_id}")
            print("=" * 60)

            try:
                result = subprocess.run(command, cwd=PROJECT_ROOT)
            except Exception as exc:  # noqa: BLE001 - keep other stages alive
                return StageResult.failed(f"{type(exc).__name__}: {exc}")

            if result.returncode == 0:
                return StageResult.success()

            if result.returncode == 2:
                self.ledger.reload()
                if stage == "images":
                    minutes = parse_reset_minutes(
                        self.ledger.info("gemini").get("message", ""),
                        default=60.0,
                    )
                    if not self.ledger.is_blocked("gemini") and not self.no_wait:
                        self.ledger.block(
                            "gemini",
                            minutes,
                            "Gemini usage limit reached",
                        )
                    return StageResult.blocked_stage("Gemini usage limit")
                if stage == "voices":
                    if not self.ledger.is_blocked("tts"):
                        self.ledger.block(
                            "tts",
                            10.0,
                            "Chatterbox TTS server failed; retrying shortly.",
                        )
                    return StageResult.blocked_stage("TTS server failure")

            return StageResult.failed(
                f"{stage} stage exited with code {result.returncode}"
            )
        finally:
            lock.release()

    # -- script stage ----------------------------------------------------

    def _run_script(self, job: StoryJob) -> StageResult:
        if job.script.exists() and job.script.stat().st_size > 0:
            return StageResult.success()

        while not self.stop_event.is_set():
            if self.no_wait:
                self.ledger.reload()
                if self.ledger.is_blocked("claude"):
                    info = self.ledger.info("claude")
                    token = (
                        info.get("last_hit_at"),
                        info.get("blocked_until_text"),
                    )
                    if self.announced.get("claude") != token:
                        print(
                            "Claude limit reached. Not waiting (--no-wait); "
                            "the script stage will retry after the cooldown."
                        )
                        self.announced["claude"] = token
                    return StageResult.blocked_stage("Claude is limited")
                if self.announced.pop("claude", None):
                    print("Claude limit cleared. Resuming.")
            else:
                ok = wait_for_resource(
                    self.ledger,
                    "claude",
                    label="Claude",
                    max_wait_minutes=self.max_wait_minutes,
                    announced=self.announced,
                )
                if not ok:
                    return StageResult.failed(
                        "Claude is still limited; stopping the factory.",
                        fatal=True,
                    )

            try:
                script_text = generate_story_with_claude(
                    prompt=build_story_prompt(job.prompt, kids=self.kids)
                )
            except ClaudeLimitError as exc:
                minutes = block_on_limit(self.ledger, "claude", str(exc))
                print(
                    f"Claude limit reached while writing {job.story_id} "
                    f"(cooldown {minutes:.0f} min)."
                )
                continue
            except Exception as exc:  # noqa: BLE001 - keep the batch alive
                return StageResult.failed(
                    f"{type(exc).__name__}: {exc}"
                )

            job.script.parent.mkdir(parents=True, exist_ok=True)
            job.script.write_text(script_text, encoding="utf-8")
            return StageResult.success()

        return StageResult.failed("factory stopped")


def _build_stage_jobs(
    state: BatchStateManager,
    state_lock: threading.RLock,
) -> list[StoryJob]:
    """Create a job for every pending prompt, reusing saved story folders."""
    jobs: list[StoryJob] = []
    with state_lock:
        pending = list(state.state["pending"])

    for prompt in pending:
        with state_lock:
            info = state.get_story(prompt)
        story_id = info.get("story_id") or make_story_id()
        script_file = BATCH_INPUT_DIR / f"{story_id}.txt"
        with state_lock:
            state.set_story(prompt, story_id, script_file)
        jobs.append(
            StoryJob(
                prompt=prompt,
                story_id=story_id,
                script=script_file,
                attempts=int(info.get("attempts", 0)),
            )
        )
    return jobs


def copy_story_to_outbox(story_id: str) -> Path:
    """Copy a finished story's deliverables where the uploader finds them."""
    story_output = OUTPUT_DIR / story_id
    final_video = story_output / "video" / "final_video.mp4"

    if not final_video.exists() or final_video.stat().st_size <= 0:
        raise RuntimeError(f"No finished video at {final_video}.")

    outbox_story = OUTBOX_DIR / story_id
    outbox_story.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_video, outbox_story / "video.mp4")

    thumbnail = story_output / "thumbnail" / "thumbnail.png"
    if thumbnail.exists():
        shutil.copy2(thumbnail, outbox_story / "thumbnail.png")

    for name in ("title.txt", "description.txt"):
        source = story_output / name
        if source.exists():
            shutil.copy2(source, outbox_story / name)

    return outbox_story


def run_stage_parallel(
    state: BatchStateManager,
    *,
    args: argparse.Namespace,
    kids: bool,
    ledger: LimitLedger,
) -> dict:
    """Run every stage as an independent worker over the whole queue."""
    state_lock = threading.RLock()
    jobs = _build_stage_jobs(state, state_lock)
    if not jobs:
        print("No pending stories to process.")
        return {"completed": 0, "failed": 0, "pending": 0}

    layout = StageLayout(
        batch_input=BATCH_INPUT_DIR,
        output_root=OUTPUT_DIR,
        outbox_root=OUTBOX_DIR,
    )

    stop_event = threading.Event()
    runner = FactoryStageRunner(
        kids=kids,
        no_agents=args.no_agents,
        max_wait_minutes=args.max_wait_minutes,
        no_wait=args.no_wait,
        ledger=ledger,
        stop_event=stop_event,
    )

    upload_event = threading.Event()
    upload_stop = threading.Event()

    def deliver(job: StoryJob) -> None:
        outbox_story = copy_story_to_outbox(job.story_id)
        with state_lock:
            state.mark_complete(job.prompt)
        print(f"Copied to outbox: {outbox_story}")
        if args.upload_each:
            upload_event.set()

    def failed(job: StoryJob) -> None:
        with state_lock:
            attempts = state.bump_attempt(job.prompt)
            if attempts >= args.max_story_attempts:
                state.mark_complete(job.prompt)
                print(
                    f"Story {job.story_id} failed {attempts} time(s); "
                    "skipping this prompt."
                )
            else:
                state.requeue(job.prompt)
                print(
                    f"Story {job.story_id} failed (attempt {attempts}/"
                    f"{args.max_story_attempts}); it will be retried on the "
                    "next factory run in the same folder."
                )

    def publish(status: dict) -> None:
        status.setdefault("pending", 0)
        status.setdefault("completed", 0)
        status.setdefault("failed", 0)
        status.setdefault("batch_id", "stage_parallel")
        update_status(status)

    def upload_worker() -> None:
        while not upload_stop.is_set():
            if not upload_event.wait(1.0):
                continue
            upload_event.clear()
            try:
                schedule_uploads(args)
            except Exception as exc:  # noqa: BLE001 - never stop rendering
                print(f"YouTube upload skipped: {exc}")

    upload_thread = None
    if args.upload_each or args.upload:
        upload_thread = threading.Thread(
            target=upload_worker,
            name="stage-upload",
            daemon=True,
        )
        upload_thread.start()

    runners = {
        "script": lambda job: runner("script", job),
        "images": lambda job: runner("images", job),
        "voices": lambda job: runner("voices", job),
        "clips": lambda job: runner("clips", job),
        "final": lambda job: runner("final", job),
    }

    scheduler = StageScheduler(
        jobs,
        layout=layout,
        runners=runners,
        clip_workers=args.clip_workers,
        max_stage_attempts=args.stage_retries,
        poll_seconds=args.stage_poll_seconds,
        retry_seconds=args.stage_retry_seconds,
        on_complete=deliver,
        on_failed=failed,
        on_status=publish,
        log=print,
        stop_event=stop_event,
    )

    try:
        summary = scheduler.run()
    except KeyboardInterrupt:
        stop_event.set()
        summary = scheduler.status()
        print("Factory interrupted; finished assets are checkpointed.")

    upload_stop.set()
    upload_event.set()
    if upload_thread is not None:
        upload_thread.join(timeout=10)

    # A final sweep catches anything the per-story uploader skipped because
    # another uploader held the Studio lock.
    if args.upload or args.upload_each:
        try:
            schedule_uploads(args)
        except Exception as exc:  # noqa: BLE001 - rendering is already done
            print(f"YouTube upload failed: {exc}")

    return summary


def run_sequential_batch(
    state: BatchStateManager,
    *,
    args: argparse.Namespace,
    kids: bool,
    ledger: LimitLedger,
) -> None:
    """Legacy one-story-at-a-time loop (``--no-stage-parallel``)."""
    announced: dict = {}

    if args.voices_first:
        voice_first_pass(
            state,
            kids=kids,
            no_agents=args.no_agents,
            ledger=ledger,
            announced=announced,
            max_wait_minutes=args.max_wait_minutes,
        )

    while True:
        prompt = state.get_next()
        if not prompt:
            print("\nAll stories in the queue have been processed!")
            break

        # main.py records Gemini cooldowns in the same ledger. Wait here, once,
        # instead of letting every story's worker print the same message.
        if not wait_for_resource(
            ledger,
            "gemini",
            label="Gemini",
            max_wait_minutes=args.max_wait_minutes,
            announced=announced,
        ):
            break
        if not wait_for_resource(
            ledger,
            "claude",
            label="Claude",
            max_wait_minutes=args.max_wait_minutes,
            announced=announced,
        ):
            break

        info = state.get_story(prompt)
        story_id = info.get("story_id") or make_story_id()
        script_file = BATCH_INPUT_DIR / f"{story_id}.txt"
        resuming = bool(info.get("story_id"))
        state.set_story(prompt, story_id, script_file)

        if resuming:
            print(f"\nResuming unfinished story {story_id}.")

        try:
            # Update status for GUI polling
            update_status({
                "current_story": story_id,
                "stage": "rendering",
                "batch_id": "batch_001",
                "pending": len(state.state["pending"]),
                "completed": len(state.state["completed"]),
            })
            success = run_single_story(
                prompt,
                story_id,
                no_agents=args.no_agents,
                kids=kids,
                resume=resuming,
                max_wait_minutes=args.max_wait_minutes,
            )
        except ClaudeLimitError as exc:  # noqa: BLE001 - keep the batch alive
            minutes = block_on_limit(ledger, "claude", str(exc))
            print(
                f"Claude limit reached while writing {story_id} "
                f"(cooldown {minutes:.0f} min); story requeued."
            )
            state.requeue(prompt)
            continue

        if not success:
            attempts = state.bump_attempt(prompt)
            if attempts >= args.max_story_attempts:
                print(
                    f"Story {story_id} failed {attempts} time(s); "
                    "skipping this prompt."
                )
                state.mark_complete(prompt)
            else:
                print(
                    f"Story {story_id} failed (attempt {attempts}/"
                    f"{args.max_story_attempts}); it will be retried later "
                    "in the same folder."
                )
                state.requeue(prompt)
            continue

        state.mark_complete(prompt)

        if args.upload_each:
            try:
                schedule_uploads(args)
            except Exception as exc:  # noqa: BLE001 - never stop rendering
                print(f"YouTube upload skipped after {story_id}: {exc}")

    if args.upload:
        try:
            schedule_uploads(args)
        except Exception as exc:  # noqa: BLE001 - rendering is already done
            print(f"YouTube upload failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Video Batch Generator")
    parser.add_argument(
        "--prompts-file",
        default="prompts.txt",
        help="File containing story prompts",
    )
    parser.add_argument(
        "--theme",
        default=None,
        help=(
            "Theme for the Architect if prompts are missing. Default: dark "
            "cinematic mysteries, or gentle kids bedtime stories with --kids."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Number of prompts to generate if missing",
    )
    parser.add_argument(
        "--kids",
        action="store_true",
        help=(
            "Generate gentle children's stories and storybook images. The "
            "setting is remembered in batch_state.json across restarts."
        ),
    )
    parser.add_argument(
        "--max-story-attempts",
        type=int,
        default=5,
        help=(
            "How many times a failing story is retried before its prompt is "
            "skipped (default: 5). Retries resume the same folder."
        ),
    )
    parser.add_argument(
        "--voices-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Voice every queued story in one continuous pass before rendering "
            "(default: on). Use --no-voices-first for the old per-story order."
        ),
    )
    parser.add_argument(
        "--max-wait-minutes",
        type=float,
        default=240.0,
        help=(
            "How long to wait for a Gemini/Claude limit before stopping the "
            "batch (default: 240; 0 waits forever)."
        ),
    )
    parser.add_argument(
        "--no-agents",
        action="store_true",
        help="Disable the Level-2 critic loops in main.py.",
    )
    parser.add_argument(
        "--stage-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run script/image/voice/clip/final as independent workers across "
            "all queued stories (default: on). Use --no-stage-parallel for the "
            "old one-story-at-a-time render loop."
        ),
    )
    parser.add_argument(
        "--clip-workers",
        type=int,
        default=1,
        help=(
            "How many stories may render clips at the same time in "
            "stage-parallel mode (default: 1)."
        ),
    )
    parser.add_argument(
        "--stage-retries",
        type=int,
        default=3,
        help="Retries per stage per story before it is deferred (default: 3).",
    )
    parser.add_argument(
        "--stage-poll-seconds",
        type=float,
        default=5.0,
        help="How often a waiting stage checks for new work (default: 5).",
    )
    parser.add_argument(
        "--stage-retry-seconds",
        type=float,
        default=30.0,
        help="Base delay before retrying a failed stage (default: 30).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help=(
            "Do not sit through usage-limit cooldowns; save checkpoints and "
            "stop/pause instead."
        ),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Schedule finished videos on YouTube when the batch ends.",
    )
    parser.add_argument(
        "--upload-each",
        action="store_true",
        help=(
            "Schedule finished videos after every successful story instead of "
            "waiting for the whole batch to finish."
        ),
    )
    parser.add_argument(
        "--dry-run-upload",
        action="store_true",
        help="With --upload, print the schedule without opening YouTube.",
    )
    parser.add_argument(
        "--per-day",
        type=int,
        default=3,
        help="Videos to schedule per day on YouTube (default: 3).",
    )
    parser.add_argument(
        "--start-per-day",
        type=int,
        default=None,
        help="Videos per day at the start of the ramp (default: 1).",
    )
    parser.add_argument(
        "--ramp-days",
        type=int,
        default=0,
        help="Ramp from --start-per-day to --per-day over this many days.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="First YouTube publish date (YYYY-MM-DD). Default: tomorrow.",
    )
    parser.add_argument(
        "--publish-time",
        default="18:00",
        help="Last publish time of each day in 24-hour format (default: 18:00).",
    )
    args = parser.parse_args()

    OUTBOX_DIR.mkdir(exist_ok=True)
    prompts_path = PROJECT_ROOT / args.prompts_file

    # The audience is remembered in batch_state.json so a restart keeps making
    # the same kind of videos even if --kids is forgotten.
    state = BatchStateManager(str(PROJECT_ROOT / "batch_state.json"))
    if args.kids:
        state.set_setting("kids", True)
    kids = bool(state.get_setting("kids", False))

    if not prompts_path.exists() or prompts_path.stat().st_size == 0:
        print("\nNo prompts found. Activating Prompt Architect...")
        architect = PromptArchitect()
        theme = args.theme or (
            "gentle bedtime stories for young children: animal friends, "
            "gentle magic, kindness, curiosity, family and cozy adventures"
            if kids
            else "dark cinematic mysteries"
        )
        new_prompts = architect.generate_batch(
            theme=theme,
            count=args.count,
            kids=kids,
        )

        if not new_prompts:
            print("Architect failed to generate prompts. Exiting.")
            return

        print(f"Architect generated {len(new_prompts)} new story ideas.")
        prompts_path.write_text("\n".join(new_prompts), encoding="utf-8")

    current_prompts = [
        line.strip()
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    state.add_prompts(current_prompts)

    print(f"Batch initialized. {len(state.state['pending'])} stories pending.")
    print(f"Audience: {'kids / family' if kids else 'general'}")

    ledger = LimitLedger(LEDGER_FILE)

    if args.stage_parallel:
        print()
        print("=" * 60)
        print("STAGE-PARALLEL FACTORY")
        print("=" * 60)
        print("Each stage runs continuously across stories:")
        print("  script -> images / voices -> clips -> final -> outbox")
        print(
            "Images no longer wait for a story's video: when a story's images "
            "finish, the image worker starts the next story immediately."
        )
        summary = run_stage_parallel(
            state,
            args=args,
            kids=kids,
            ledger=ledger,
        )
        print()
        print(
            "Stage-parallel run finished: "
            f"{summary.get('completed', 0)} complete, "
            f"{summary.get('failed', 0)} deferred, "
            f"{summary.get('pending', 0)} still pending."
        )
        return

    run_sequential_batch(
        state,
        args=args,
        kids=kids,
        ledger=ledger,
    )


if __name__ == "__main__":
    main()
