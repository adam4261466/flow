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
import time
from pathlib import Path

from agents.batch_state import BatchStateManager
from agents.limits import LimitLedger, parse_reset_minutes
from agents.prompt_architect import PromptArchitect
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
    while (OUTPUT_DIR / f"story_{candidate}").exists() or (
        OUTBOX_DIR / f"story_{candidate}"
    ).exists():
        candidate += 1
    return f"story_{candidate}"


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


def schedule_uploads(args: argparse.Namespace) -> None:
    """Send every finished outbox/output video to the YouTube uploader."""
    from youtube_uploader import upload_all_finished_videos

    upload_all_finished_videos(
        videos_per_day=args.per_day,
        ramp_days=args.ramp_days,
        start_videos_per_day=args.start_per_day,
        start_date=args.start_date,
        publish_time=args.publish_time,
        dry_run=args.dry_run_upload,
    )


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
    announced: dict = {}

    # Continuous voice pass: write each script, voice it, then move straight to
    # the next story. Rendering afterwards reuses those WAVs.
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


if __name__ == "__main__":
    main()
