import json
import threading
import time
import argparse
from datetime import datetime
from pathlib import Path
import os
import subprocess

from utils.console import setup_console

setup_console()

from parser.script_parser import parse_script
from utils.validator import validate_clips

from ai.gemini_browser import (
    GeminiBrowser,
    GeminiLimitError,
    GeminiTextResponseError,
)
from ai.ollama_video import OllamaVideoPlanner

from audio.chatterbox_tts import ChatterboxTTS

from editor.video_editor import VideoEditor
from PIL import Image, ImageOps

from agents.config import AgentConfig
from agents.critics import VoiceCritic
from agents.loops import (
    run_image_qc_loop,
    run_thumbnail_qc_loop,
    run_voice_qc_loop,
)
from agents.runtime import AgentRuntime
from agents.state import ProductionState


# Resolve every project-relative path against this file, not the process
# working directory. When gui.py is launched by double-click the working
# directory can be a protected folder (e.g. System32), which made
# Path("output").mkdir() fail with "[WinError 5] Access is denied: 'output'".
PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# GLOBALS (set by GUI or CLI)
# ============================================================

PRE_SELECTED_AFTER_FINISH = None  # "none" | "sleep" | "shutdown" | None

# The parallel workers can save progress from several threads at once.
CHECKPOINT_LOCK = threading.Lock()


# ============================================================
# CLI
# ============================================================

def default_script_path():
    """Pick input/script.txt, or the first script in input/ if it is gone."""
    input_dir = PROJECT_ROOT / "input"
    default = input_dir / "script.txt"
    if default.exists():
        return str(default)

    if input_dir.is_dir():
        candidates = sorted(input_dir.glob("*.txt"))
        if candidates:
            return str(candidates[0])

    return str(default)


def parse_args():
    parser = argparse.ArgumentParser(description="AI Video Creator")
    parser.add_argument(
        "--story",
        default="story_001",
        help="Story/output folder name (default: story_001)",
    )
    parser.add_argument(
        "--script",
        default=default_script_path(),
        help="Path to script file (default: input/script.txt)",
    )
    parser.add_argument(
        "--after-finish",
        choices=["none", "sleep", "shutdown"],
        default=None,
        help="Action after successful completion (default: prompt interactively)",
    )
    parser.add_argument(
        "--recap",
        action="store_true",
        help=(
            "Check every expected image, voice and video by filename and "
            "regenerate only the ones that are missing or stale."
        ),
    )
    parser.add_argument(
        "--generate-story",
        action="store_true",
        help=(
            "Ask Claude for a new story first, save it as the next "
            "input/scriptN.txt, and process it into output/story_NNN."
        ),
    )
    parser.add_argument(
        "--regenerate-thumbnail",
        action="store_true",
        help=(
            "Regenerate the thumbnail with Gemini even if thumbnail.png "
            "already exists."
        ),
    )
    parser.add_argument(
        "--thumbnail-only",
        action="store_true",
        help=(
            "Stop after generating the thumbnail. Useful with "
            "--regenerate-thumbnail to refresh an existing story's thumbnail."
        ),
    )
    parser.add_argument(
        "--voices-only",
        action="store_true",
        help=(
            "Only generate the missing narration for this story and exit. "
            "The batch factory uses this to voice every queued story in one "
            "continuous pass before rendering starts."
        ),
    )
    parser.add_argument(
        "--kids",
        action="store_true",
        help=(
            "Family-safe mode: gentle children's stories, storybook images "
            "and bright, friendly thumbnails instead of dark cinematic ones."
        ),
    )
    parser.add_argument(
        "--no-agents",
        action="store_true",
        help=(
            "Disable the Level-2 critic loops and use the previous "
            "generate-only pipeline."
        ),
    )
    parser.add_argument(
        "--vision-backend",
        choices=["auto", "gemini", "ollama", "openai", "none"],
        default=None,
        help=(
            "Critic vision backend (default: auto). 'gemini' reuses the "
            "existing Chrome/Gemini session."
        ),
    )
    parser.add_argument(
        "--vision-model",
        default=None,
        help="Ollama vision model name, e.g. qwen2.5vl:7b.",
    )
    parser.add_argument(
        "--image-threshold",
        type=float,
        default=None,
        help="Minimum image score from 0-10 to accept (default: 7.0).",
    )
    parser.add_argument(
        "--max-image-retries",
        type=int,
        default=None,
        help="Extra image attempts after the first (default: 2).",
    )
    parser.add_argument(
        "--max-thumbnail-retries",
        type=int,
        default=None,
        help="Extra thumbnail attempts after the first (default: 2).",
    )
    parser.add_argument(
        "--max-voice-retries",
        type=int,
        default=None,
        help="Extra voice attempts after the first (default: 1).",
    )
    parser.add_argument(
        "--max-critic-calls",
        type=int,
        default=None,
        help="Semantic critic calls allowed per story (default: 60).",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help=(
            "Run the old stage-by-stage pipeline instead of the parallel "
            "stage workers."
        ),
    )
    parser.add_argument(
        "--gemini-cooldown-minutes",
        type=float,
        default=None,
        help=(
            "How long the image worker waits after a Gemini usage limit "
            "before retrying (default: 60)."
        ),
    )
    parser.add_argument(
        "--max-wait-minutes",
        type=float,
        default=None,
        help=(
            "Maximum time the parallel workers wait for a Gemini limit to "
            "clear before pausing (default: 240)."
        ),
    )
    parser.add_argument(
        "--parallel-poll-seconds",
        type=float,
        default=None,
        help=(
            "How often each stage worker re-checks readiness "
            "(default: 5)."
        ),
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help=(
            "When a usage limit is hit, save a checkpoint and stop instead "
            "of waiting for the cooldown."
        ),
    )
    return parser.parse_args()


# ============================================================
# PATHS (populated from args in main)
# ============================================================

SCRIPT_PATH = None  # set in main()
STORY_NAME = None   # set in main()

OUTPUT_ROOT = PROJECT_ROOT / "output"
STORY_OUTPUT = None
IMAGE_OUTPUT = None
AUDIO_OUTPUT = None
VIDEO_OUTPUT = None
THUMBNAIL_OUTPUT = None

VISUAL_PLAN_FILE = None
THUMBNAIL_PLAN_FILE = None
TITLE_FILE = None
CHECKPOINT_FILE = None
GEMINI_CONVERSATION_FILE = None


# ============================================================
# SETTINGS
# ============================================================

VIDEO_FPS = 30
VIDEO_RESOLUTION = (1920, 1080)
VIDEO_BITRATE = "8M"

FADE_DURATION = 0.4

# Narration is generated by the local Chatterbox TTS Server.
# CHATTERBOX_VOICE is a filename from that server's voices/ folder.
CHATTERBOX_SERVER_URL = "http://127.0.0.1:8004"
CHATTERBOX_VOICE = (
    "ElevenLabs_2026-11-03T20_43_25_Amara_-_Calm__Elegant_and_"
    "Refined_pvc_sp95_s20_sb9_v4.mp3"
)
CHATTERBOX_LANGUAGE = "en"

# Chatterbox's "Standard Narration" preset.
CHATTERBOX_TEMPERATURE = 0.8
CHATTERBOX_EXAGGERATION = 0.4
CHATTERBOX_CFG_WEIGHT = 0.5
CHATTERBOX_SEED = 42
CHATTERBOX_SPEED_FACTOR = 1.0

INTRO_DURATION = 3.0
OUTRO_DURATION = 4.0

THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT = 720

AFTER_FINISH_COUNTDOWN = 15  # seconds before sleep/shutdown (Ctrl+C cancels)

# ------------------------------------------------------------------
# Resume / checkpoint behaviour
# ---------------------------------------------------------------------------
# RESUME_EXISTING = True means every stage skips work whose output already
# exists on disk, and reloads the saved title / visual plan / thumbnail plan
# instead of regenerating them. This is what lets a run "continue from where
# it left off" after a Gemini account limit.
#
# Set to False to force a full regeneration (fresh images, voices, plans).
RESUME_EXISTING = True

# RECAP_MISSING is set by --recap (and the GUI "Recap missing" button). It
# makes the pipeline behave like a repair pass: every expected asset is looked
# up by filename and only the missing/stale ones are regenerated.
RECAP_MISSING = False

# FORCE_THUMBNAIL makes the image phase regenerate thumbnail.png even when
# RESUME_EXISTING finds an old one. THUMBNAIL_ONLY stops the pipeline right
# after the thumbnail is ready, so an existing story can be refreshed without
# re-rendering its video.
FORCE_THUMBNAIL = False
THUMBNAIL_ONLY = False
VOICES_ONLY = False

# KIDS_MODE is set by --kids. It swaps the image and thumbnail art direction
# for bright, child-friendly storybook illustration. The story text itself is
# written with KIDS_STORY_PROMPT (see ai/claude_browser.py).
KIDS_MODE = False

# When a Gemini usage limit is detected the image phase pauses instead of
# aborting: the pipeline then runs voice generation (which does not need
# Gemini) and saves a checkpoint, so the next run only has to finish the
# remaining images. See GeminiLimitError.

# Auto-resume: when the Gemini limit is hit, keep the process alive and retry
# image generation every LIMIT_POLL_SECONDS until the limit clears (or
# LIMIT_MAX_WAIT_MINUTES elapses). Set to False to instead exit and resume by
# re-running main.py later.
#
# Disabled on purpose: when the Gemini limit is hit we now STOP and wait for a
# manual re-run instead of polling in the background.
AUTO_RESUME_AFTER_LIMIT = False
LIMIT_POLL_SECONDS = 300          # how often to retry while limited (5 min)
LIMIT_MAX_WAIT_MINUTES = 240      # give up auto-resume after this (4 hours)

# Transient browser/UI hiccups (a truncated composer, a stale tab) should not
# kill a long run. Each image is retried this many times before giving up.
IMAGE_MAX_ATTEMPTS = 3

# Level-2 critic layer. AGENTS is created in main() once the story paths are
# known. When agents are disabled it stays None and every stage uses the
# previous generate-only code path.
AGENT_CONFIG = None
AGENT_STATE = None
AGENTS = None


# ============================================================
# SCRIPT
# ============================================================

def load_script():
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"Script file not found:\n{SCRIPT_PATH.resolve()}\n\n"
            "Put your script in input/script.txt, choose another file "
            "in the GUI, or run with --script <file>."
        )

    with open(
        SCRIPT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return file.read()


# ============================================================
# DIRECTORIES
# ============================================================

def create_output_directories():
    for directory in [
        IMAGE_OUTPUT,
        AUDIO_OUTPUT,
        VIDEO_OUTPUT,
        THUMBNAIL_OUTPUT,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# GEMINI STORY IMAGE PROMPT
# ============================================================

# --kids swaps the cinematic/photoreal art direction for a storybook look.
# The replacements are applied only when KIDS_MODE is on, so the original
# pipeline text stays byte-for-byte unchanged for normal stories.

_KIDS_CLIP_STYLE_REPLACEMENTS = (
    (
        "Generate ONE high-quality cinematic image for this video clip.",
        "Generate ONE high-quality, child-friendly storybook illustration for this video clip.",
    ),
    (
        "Visual requirements:\n"
        "- Photorealistic cinematic film still.\n"
        "- High visual detail.\n"
        "- Sharp textures and realistic materials.\n"
        "- Natural cinematic lighting.\n"
        "- Realistic shadows.\n"
        "- Professional composition.",
        "Visual requirements:\n"
        "- Bright, colorful, child-friendly 3D-animated storybook illustration.\n"
        "- Soft rounded shapes, friendly faces and expressive characters.\n"
        "- Cheerful, gentle lighting with warm colors.\n"
        "- Simple, uncluttered composition a young child can read at a glance.\n"
        "- Wholesome, cozy atmosphere. Nothing scary, dark or threatening.\n"
        "- Professional children's picture-book quality.",
    ),
    (
        "color palette, and overall cinematic look.",
        "color palette, and overall look.",
    ),
)

_KIDS_THUMBNAIL_STYLE_REPLACEMENTS = (
    (
        "for a cinematic story video",
        "for a children's story video",
    ),
    (
        "premium movie-poster typography",
        "playful, chunky storybook lettering",
    ),
    (
        "Professional cinematic movie-poster / YouTube-thumbnail quality.",
        "Bright, colorful children's storybook / YouTube-thumbnail quality.",
    ),
    (
        "Cinematic lighting with depth and atmosphere.",
        "Warm, cheerful lighting with soft depth and a sunny mood.",
    ),
    (
        "Dramatic but believable visual storytelling.",
        "Delightful, friendly visual storytelling that feels safe and welcoming.",
    ),
    (
        "High detail and photorealistic cinematic quality.",
        "High detail, colorful 3D-animated storybook quality.",
    ),
)


def _apply_style_variants(text, pairs):
    for old, new in pairs:
        if old in text:
            text = text.replace(old, new, 1)
    return text


def build_clip_prompt(
    clip,
    visual_plan,
    story_text=None,
    has_reference=False,
    strict_image_only=False,
    revision_notes=None,
):
    """Build the prompt pasted into a fresh Gemini conversation for one clip.

    ``story_text`` gives Gemini the whole story so it understands what happened
    before this moment. ``has_reference`` tells it that the previous clip's
    image is attached and must be used for visual continuity. ``revision_notes``
    carries the critic's concrete fix into a retry.
    """
    story_block = ""

    if story_text:
        story_block = f"""
FULL STORY (for context and continuity):
{story_text}
"""

    reference_block = ""

    if has_reference:
        reference_block = """
REFERENCE IMAGE:
The image attached to this message is the previous visual beat of this exact
story (if a clip had to be skipped, it is the most recent successful frame).
Use it as the visual reference for how everything looks. Keep the same
characters, faces, hairstyles, clothing, environment, props, lighting style,
color palette, and overall cinematic look. Continue the story forward from
that moment. Do NOT copy the attached frame literally: create the NEXT beat
as a new image.
"""

    strict_block = ""

    if strict_image_only:
        strict_block = """
CRITICAL OUTPUT RULE:
You MUST generate an image now. Do NOT reply with text, an explanation, a
description, a question, an apology, or a summary. If any detail is unclear,
make a reasonable cinematic choice and still generate exactly one image.
"""

    revision_block = ""

    if revision_notes and str(revision_notes).strip():
        revision_block = f"""
REVISION NOTES (from the previous attempt):
{str(revision_notes).strip()}
Apply these corrections in this new image.
"""

    prompt = f"""
Generate ONE high-quality cinematic image for this video clip.

Clip:
{clip.id}

Characters:
{clip.characters}

Moment:
{clip.moment}

Vibe:
{clip.vibe}

Style:
{clip.style}

Camera:
{visual_plan["camera"]}

Composition:
{visual_plan["composition"]}
{story_block}{reference_block}{strict_block}{revision_block}
Visual requirements:
- Photorealistic cinematic film still.
- High visual detail.
- Sharp textures and realistic materials.
- Natural cinematic lighting.
- Realistic shadows.
- Professional composition.
- Fill the entire frame.
- No borders.
- No UI.
- No captions.
- No subtitles.
- No text.
- No collage.
- One scene only.
- Maintain character continuity.
- Maintain clothing continuity.
- Maintain environment continuity.
- Maintain important object continuity.
""".strip()

    if KIDS_MODE:
        prompt = _apply_style_variants(prompt, _KIDS_CLIP_STYLE_REPLACEMENTS)

    return prompt


# ============================================================
# THUMBNAIL PROMPT
# ============================================================

def build_thumbnail_prompt(title, thumbnail_plan):
    """Build the full Gemini thumbnail prompt, including the rendered title."""
    title_position = thumbnail_plan["title_position"].replace("_", " ")
    return f"""
Create ONE complete, finished, ready-to-upload YouTube thumbnail image for a cinematic story video. The thumbnail must be generated entirely by you, including the title lettering. Do not leave a blank area for a later text overlay.

THE EXACT TITLE TEXT TO RENDER INSIDE THE IMAGE IS:
"{title}"

TEXT RULES — CRITICAL:
- Render the title exactly as written above, word for word, with correct spelling, capitalization and punctuation.
- Make the title large, bold and cinematic, using premium movie-poster typography with a strong outline, shadow or subtle glow so it stays readable at small thumbnail size.
- Place the title in the {title_position} area of the frame.
- Use one or two lines maximum. Keep every letter inside the safe margins and away from the outer edges.
- Use ONLY the title words above. Do not add subtitles, captions, credits, channel names, logos, watermarks, UI, borders or any other text.
- If the attached reference image contains text, ignore that text completely.

VISUAL DIRECTION:
Main subject:
{thumbnail_plan["subject"]}

Scene:
{thumbnail_plan["scene"]}

Emotion:
{thumbnail_plan["emotion"]}

Camera:
{thumbnail_plan["camera"]}

Lighting:
{thumbnail_plan["lighting"]}

Color palette:
{thumbnail_plan["color_palette"]}

Composition:
{thumbnail_plan["composition"]}

REFERENCE IMAGE:
Use the attached clip image as the visual reference for the main character(s), costume, creature design, environment, props, color palette and lighting continuity. Do NOT copy the reference image's framing or camera angle. Build a new, stronger movie-poster composition that clearly belongs to the same story and characters.

THUMBNAIL REQUIREMENTS:
- Professional cinematic movie-poster / YouTube-thumbnail quality.
- 16:9 landscape composition framed for 1280x720.
- One dominant subject with a clear silhouette and an expressive, readable face or focal point.
- Strong focal point, high contrast and strong foreground/background separation.
- Cinematic lighting with depth and atmosphere.
- Dramatic but believable visual storytelling.
- High detail and photorealistic cinematic quality.
- Keep the subject large enough to be readable on a phone at small size.
- Integrated title typography that feels designed into the artwork, not pasted on.
- One single image only. No collage, no split panels, no borders, no mockup, no UI.
- No watermark, no logo, no signature.
""".strip()

    if KIDS_MODE:
        prompt = _apply_style_variants(prompt, _KIDS_THUMBNAIL_STYLE_REPLACEMENTS)

    return prompt


def choose_thumbnail_reference(thumbnail_plan=None):
    """Pick one of the clip images as Gemini's visual reference."""
    plan = thumbnail_plan or {}

    explicit = plan.get("reference_image")
    if explicit:
        path = Path(str(explicit))
        if not path.is_absolute():
            path = IMAGE_OUTPUT / path
        if path.is_file() and path.stat().st_size > 0:
            return path

    try:
        clip_id = int(plan.get("reference_clip", 1))
    except (TypeError, ValueError):
        clip_id = 1

    candidate = IMAGE_OUTPUT / f"image_{clip_id:03d}.png"
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate

    first = IMAGE_OUTPUT / "image_001.png"
    if first.is_file() and first.stat().st_size > 0:
        return first

    images = sorted(IMAGE_OUTPUT.glob("image_*.png"))
    for image in reversed(images):
        if image.stat().st_size > 0:
            return image
    return None


def normalize_thumbnail(path):
    """Crop/resize Gemini's output to YouTube's 1280x720 thumbnail size."""
    path = Path(path)
    with Image.open(path) as image:
        fitted = ImageOps.fit(
            image.convert("RGB"),
            (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT),
            method=Image.Resampling.LANCZOS,
        )

    fitted.save(path, "PNG", optimize=True)

    if path.stat().st_size > 2_000_000:
        # YouTube's custom-thumbnail limit is 2 MB. Quantizing a PNG keeps the
        # same filename that the uploader expects while shrinking the file.
        fitted.quantize(colors=256).save(path, "PNG", optimize=True)

    return path


# ============================================================
# ACTION AFTER FINISH
# ============================================================

def choose_after_finish_action(pre_selected: str | None = None):
    """
    Ask what Windows should do after the video pipeline
    finishes successfully. Called once, at startup.

    If ``pre_selected`` is provided, return it immediately without prompting.
    """
    if pre_selected in ("none", "sleep", "shutdown"):
        return pre_selected

    print()
    print("========================================")
    print("AFTER FINISH")
    print("========================================")

    print("1. Do nothing")
    print("2. Sleep")
    print("3. Shutdown")

    while True:

        choice = input(
            "\nSelect action [1/2/3]: "
        ).strip()

        if choice == "1":
            return "none"

        if choice == "2":
            return "sleep"

        if choice == "3":
            return "shutdown"

        print(
            "Invalid choice. Enter 1, 2, or 3."
        )


def countdown_before_action(action, seconds=AFTER_FINISH_COUNTDOWN):
    """
    Give the user a chance to cancel with Ctrl+C.
    Returns True if the action should proceed.
    """

    print()
    print(
        f"Action '{action}' in {seconds}s... "
        f"Press Ctrl+C to cancel."
    )

    try:
        for remaining in range(seconds, 0, -1):
            print(
                f"  {remaining}...  ",
                end="\r",
                flush=True,
            )
            time.sleep(1)

    except KeyboardInterrupt:
        print()
        print("Action cancelled.")
        return False

    print()
    return True


def perform_after_finish_action(action):
    """
    Perform the selected Windows action.

    This must only be called after the complete pipeline
    has successfully finished, as the LAST step of main().
    """

    if action == "none":

        print()
        print(
            "No automatic action selected."
        )

        return

    if not countdown_before_action(action):
        return

    if action == "sleep":

        print()
        print(
            "Putting Windows to sleep..."
        )

        subprocess.run(
            [
                "rundll32.exe",
                "powrprof.dll,SetSuspendState",
                "0,1,0",
            ],
            check=False,
        )

        return

    if action == "shutdown":

        print()
        print(
            "Shutting down Windows..."
        )

        subprocess.run(
            [
                "shutdown",
                "/s",
                "/t",
                "0",
            ],
            check=False,
        )

        return

    raise ValueError(
        f"Unknown post-finish action: {action}"
    )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(path, data):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def load_json(path):
    """Return parsed JSON, or None when the file is missing/unreadable."""
    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
    except (FileNotFoundError, OSError, ValueError):
        return None


def build_progress(clips):
    """Inspect the output folders and report which assets already exist."""
    def present(path):
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    progress = {
        "images": [],
        "images_pending": [],
        "voices": [],
        "voices_pending": [],
        "clips": [],
        "clips_pending": [],
        "video_rendered": False,
    }

    for clip in clips:
        image_path = IMAGE_OUTPUT / f"image_{clip.id:03d}.png"
        audio_path = AUDIO_OUTPUT / f"audio_{clip.id:03d}.wav"
        clip_video_path = VIDEO_OUTPUT / f"clip_{clip.id:03d}.mp4"

        (progress["images"] if present(image_path) else progress["images_pending"]).append(clip.id)
        (progress["voices"] if present(audio_path) else progress["voices_pending"]).append(clip.id)
        (progress["clips"] if present(clip_video_path) else progress["clips_pending"]).append(clip.id)

    # Gemini now renders the complete thumbnail (artwork + title) directly.
    progress["thumbnail"] = present(
        THUMBNAIL_OUTPUT / "thumbnail.png"
    )

    progress["final_video"] = present(
        VIDEO_OUTPUT / "final_video.mp4"
    )
    progress["video_rendered"] = progress["final_video"]

    return progress


def save_checkpoint(clips, stage, blocked=False, limit_message=None):
    """Write checkpoint.json so the next run can report/resume progress."""
    with CHECKPOINT_LOCK:
        progress = build_progress(clips)

        data = {
            "story": STORY_NAME,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage,
            "gemini_blocked": blocked,
            "gemini_limit_message": limit_message,
            "progress": progress,
        }

        save_json(CHECKPOINT_FILE, data)

        if AGENT_STATE is not None:
            AGENT_STATE.set_stage(stage)
            AGENT_STATE.set_artifact("progress", progress)
            AGENT_STATE.save()

        return data


def print_progress(progress):
    """Print a short human-readable resume summary."""
    images_total = len(progress["images"]) + len(progress["images_pending"])
    voices_total = len(progress["voices"]) + len(progress["voices_pending"])
    clips_total = len(progress["clips"]) + len(progress["clips_pending"])

    print()
    print("========================================")
    print("PROGRESS")
    print("========================================")
    print(f"  Images:   {len(progress['images'])}/{images_total}")
    print(f"  Voices:   {len(progress['voices'])}/{voices_total}")
    print(f"  Clips:    {len(progress['clips'])}/{clips_total}")
    print(f"  Thumbnail: {'yes' if progress['thumbnail'] else 'no'}")
    print(f"  Final video: {'yes' if progress['final_video'] else 'no'}")

    if progress["images_pending"]:
        print(
            "  Pending images: "
            + ", ".join(str(i) for i in progress["images_pending"])
        )

    if progress["voices_pending"]:
        print(
            "  Pending voices: "
            + ", ".join(str(i) for i in progress["voices_pending"])
        )

    if progress["clips_pending"]:
        print(
            "  Pending clips: "
            + ", ".join(str(i) for i in progress["clips_pending"])
        )


def print_recap_report(progress):
    """Print exactly what a recap would regenerate."""
    missing_images = progress["images_pending"]
    missing_voices = progress["voices_pending"]
    missing_clips = progress["clips_pending"]

    print()
    print("========================================")
    print("RECAP — MISSING ASSETS")
    print("========================================")

    print(
        "  Images missing: "
        + (", ".join(str(i) for i in missing_images) if missing_images else "none")
    )
    print(
        "  Voices missing: "
        + (", ".join(str(i) for i in missing_voices) if missing_voices else "none")
    )
    print(
        "  Clips missing:  "
        + (", ".join(str(i) for i in missing_clips) if missing_clips else "none")
    )
    print(f"  Thumbnail:      {'present' if progress['thumbnail'] else 'MISSING'}")
    print(
        "  Final video:    "
        f"{'present' if progress['final_video'] else 'MISSING'}"
    )


# ============================================================
# GEMINI IMAGE PHASE (reusable, supports auto-resume)
# ============================================================

def generate_images_and_thumbnail(
    clips,
    visual_plans,
    title,
    thumbnail_plan,
    story_text=None,
):
    """Generate any missing clip images and the thumbnail.

    Every clip is created in a brand-new Gemini conversation. The story and the
    clip prompt are pasted fresh, and the previous clip's image is attached as
    the visual reference so characters, clothing and environments stay
    consistent.

    Returns ``(blocked, limit_message)``. ``blocked`` is True when Gemini's
    account limit stopped generation; in that case the caller can do other
    work and retry later.
    """
    gemini = GeminiBrowser(
        conversation_file=GEMINI_CONVERSATION_FILE,
    )
    blocked = False
    limit_message = None
    thumbnail_path = THUMBNAIL_OUTPUT / "thumbnail.png"

    # The previous clip's image used as the reference for the next clip.
    last_image = None
    failed_images = []
    reuse = RESUME_EXISTING or RECAP_MISSING

    try:
        print()
        print("Connecting to existing Chrome...")

        gemini.connect()
        gemini.open_gemini()

        if AGENTS is not None and AGENTS.enabled:
            AGENTS.attach_gemini(gemini)

        for clip in clips:
            print()
            print("========================================")
            print(f"CLIP {clip.id} — IMAGE")
            print("========================================")

            image_path = (
                IMAGE_OUTPUT
                / f"image_{clip.id:03d}.png"
            )

            if (
                reuse
                and image_path.exists()
                and image_path.stat().st_size > 0
            ):
                print()
                print("Image already exists, skipping:")
                print(image_path)
                # Keep continuity for the next clip even when this one was
                # already generated by an earlier run.
                last_image = image_path
                continue

            visual_plan = visual_plans.get(clip.id)

            if visual_plan is None:
                raise RuntimeError(
                    f"No visual plan found for clip {clip.id}."
                )

            reference = (
                last_image
                if last_image is not None and last_image.exists()
                else None
            )

            if reference is not None:
                print()
                print(f"Using previous clip as visual reference: {reference.name}")

            use_agents = AGENTS is not None and AGENTS.enabled

            if use_agents:
                critic = AGENTS.image_critic()
                max_attempts = 1 + AGENTS.config.max_image_retries

                def generate_candidate(attempt, revision):
                    image_prompt = build_clip_prompt(
                        clip,
                        visual_plan,
                        story_text=story_text,
                        has_reference=reference is not None,
                        strict_image_only=attempt > 1,
                        revision_notes=revision,
                    )

                    print()
                    print(
                        f"Generating Gemini image with critic "
                        f"(attempt {attempt}/{max_attempts})..."
                    )

                    if AGENT_STATE is not None:
                        AGENT_STATE.budget.spend_image()

                    return gemini.generate_image(
                        prompt=image_prompt,
                        output_directory=IMAGE_OUTPUT,
                        filename=(
                            f"image_{clip.id:03d}_candidate_{attempt}.png"
                        ),
                        reference_image=reference,
                        new_conversation=True,
                    )

                def evaluate_candidate(path, attempt):
                    review = critic.evaluate(
                        path,
                        clip,
                        visual_plan,
                        story_text=story_text,
                        same_conversation=True,
                    )
                    if AGENT_STATE is not None:
                        AGENT_STATE.record_attempt(
                            "image",
                            attempt,
                            score=review.score,
                            accepted=(
                                review.score
                                >= AGENTS.config.image_threshold
                            ),
                            issues=review.issues,
                            clip=clip.id,
                        )
                    return review

                try:
                    result = run_image_qc_loop(
                        generate=generate_candidate,
                        evaluate=evaluate_candidate,
                        final_path=image_path,
                        max_attempts=max_attempts,
                        threshold=AGENTS.config.image_threshold,
                        fatal_exceptions=(GeminiLimitError,),
                    )
                except GeminiLimitError as error:
                    blocked = True
                    limit_message = str(error)

                    print()
                    print("========================================")
                    print("GEMINI LIMIT REACHED")
                    print("========================================")
                    print(limit_message)
                    print()
                    print(
                        "Pausing image generation and continuing with "
                        "voice generation so no time is wasted."
                    )
                    break

                if result.path is None:
                    print()
                    print(
                        f"Giving up on clip {clip.id} for now. "
                        "It will be retried by the next run or recap."
                    )
                    failed_images.append(clip.id)
                    continue

                if AGENT_STATE is not None:
                    AGENT_STATE.record_decision(
                        "image_critic",
                        "promote",
                        clip=clip.id,
                        reason=(
                            f"best score {result.best_score:.1f} "
                            f"(threshold "
                            f"{AGENTS.config.image_threshold:.1f})"
                        ),
                    )

            else:
                for attempt in range(1, IMAGE_MAX_ATTEMPTS + 1):
                    image_prompt = build_clip_prompt(
                        clip,
                        visual_plan,
                        story_text=story_text,
                        has_reference=reference is not None,
                        strict_image_only=attempt > 1,
                    )

                    print()
                    print(
                        f"Generating Gemini image "
                        f"(attempt {attempt}/{IMAGE_MAX_ATTEMPTS})..."
                    )

                    try:
                        gemini.generate_image(
                            prompt=image_prompt,
                            output_directory=IMAGE_OUTPUT,
                            filename=f"image_{clip.id:03d}.png",
                            reference_image=reference,
                            new_conversation=True,
                        )
                        break
                    except GeminiLimitError as error:
                        blocked = True
                        limit_message = str(error)

                        print()
                        print("========================================")
                        print("GEMINI LIMIT REACHED")
                        print("========================================")
                        print(limit_message)
                        print()
                        print(
                            "Pausing image generation and continuing with "
                            "voice generation so no time is wasted."
                        )
                        break
                    except Exception as error:  # noqa: BLE001 - retry transient UI errors
                        if isinstance(error, GeminiTextResponseError):
                            print()
                            print(
                                "Gemini answered with text instead of an image. "
                                "Retrying with a stricter image-only prompt."
                            )
                        print()
                        print(
                            f"Image attempt {attempt}/{IMAGE_MAX_ATTEMPTS} for "
                            f"clip {clip.id} failed: {error}"
                        )
                        if attempt >= IMAGE_MAX_ATTEMPTS:
                            print()
                            print(
                                f"Giving up on clip {clip.id} for now. "
                                "It will be retried by the next run or recap."
                            )
                            failed_images.append(clip.id)
                            break
                        print("Retrying this clip in a fresh conversation...")
                        time.sleep(2)

            if blocked:
                break

            if not (image_path.exists() and image_path.stat().st_size > 0):
                # This clip failed; keep the previous successful image as the
                # reference for the next clip and keep going.
                continue

            last_image = image_path

            print()
            print("Image saved:")
            print(image_path)

        if failed_images:
            print()
            print("========================================")
            print("IMAGES STILL MISSING")
            print("========================================")
            print(
                "Clips: "
                + ", ".join(str(clip_id) for clip_id in failed_images)
            )
            print(
                "Voice generation will continue, then the pipeline will pause. "
                "Run recap when Gemini is ready."
            )

        # ====================================================
        # THUMBNAIL ARTWORK + TITLE (entirely Gemini)
        # ====================================================

        if blocked:
            print()
            print("Skipping thumbnail generation (Gemini limit).")
        else:
            print()
            print("========================================")
            print("GENERATING THUMBNAIL WITH GEMINI")
            print("========================================")

            if (
                reuse
                and not FORCE_THUMBNAIL
                and thumbnail_path.exists()
                and thumbnail_path.stat().st_size > 0
            ):
                print()
                print("Thumbnail already exists, skipping:")
                print(thumbnail_path)
            else:
                thumbnail_prompt = build_thumbnail_prompt(
                    title=title,
                    thumbnail_plan=thumbnail_plan,
                )
                reference = choose_thumbnail_reference(thumbnail_plan)

                print()
                if reference is not None:
                    print(
                        "Using clip image as visual reference for the "
                        f"thumbnail: {reference.name}"
                    )
                else:
                    print(
                        "Warning: no clip image was found; generating the "
                        "thumbnail without a visual reference."
                    )

                use_agents = AGENTS is not None and AGENTS.enabled

                if use_agents:
                    critic = AGENTS.thumbnail_critic(title)
                    max_attempts = 1 + AGENTS.config.max_thumbnail_retries

                    def generate_thumbnail(attempt, revision):
                        prompt_text = build_thumbnail_prompt(
                            title=title,
                            thumbnail_plan=thumbnail_plan,
                        )
                        if revision:
                            prompt_text += "\n\n" + str(revision).strip()
                        if attempt > 1:
                            prompt_text += (
                                "\n\nCRITICAL: Generate the image now. "
                                "Render the exact title text inside the "
                                "image. Do NOT reply with text."
                            )

                        candidate = THUMBNAIL_OUTPUT / (
                            f"thumbnail_candidate_{attempt}.png"
                        )

                        print()
                        print(
                            f"Generating Gemini thumbnail with critic "
                            f"(attempt {attempt}/{max_attempts})..."
                        )

                        gemini.generate_image(
                            prompt=prompt_text,
                            output_directory=THUMBNAIL_OUTPUT,
                            filename=candidate.name,
                            reference_image=reference,
                            new_conversation=True,
                        )

                        # Critique the exact 1280x720 asset that would ship,
                        # not Gemini's uncropped source image.
                        normalize_thumbnail(candidate)
                        return candidate

                    def check_thumbnail(path, attempt):
                        review = critic.check(
                            path,
                            same_conversation=False,
                        )
                        if AGENT_STATE is not None:
                            AGENT_STATE.record_attempt(
                                "thumbnail",
                                attempt,
                                score=review.score,
                                accepted=review.accepted,
                                issues=review.issues,
                                clip="thumbnail",
                            )
                        return review

                    try:
                        result = run_thumbnail_qc_loop(
                            generate=generate_thumbnail,
                            check=check_thumbnail,
                            final_path=thumbnail_path,
                            max_attempts=max_attempts,
                            fatal_exceptions=(GeminiLimitError,),
                        )
                    except GeminiLimitError as error:
                        blocked = True
                        limit_message = str(error)
                        print()
                        print("Gemini limit reached during thumbnail.")
                        print(limit_message)
                        result = None

                    if result is not None and result.path is not None:
                        print()
                        print("Gemini thumbnail created:")
                        print(thumbnail_path)
                        if AGENT_STATE is not None:
                            AGENT_STATE.record_decision(
                                "thumbnail_critic",
                                "promote",
                                clip="thumbnail",
                                reason=(
                                    "thumbnail accepted"
                                    if result.accepted
                                    else "best available thumbnail"
                                ),
                            )
                    elif not blocked:
                        print()
                        print("Thumbnail was not generated.")

                else:
                    # Generate to a temporary file so a failed Gemini attempt
                    # never destroys a previously working thumbnail.
                    temp_thumbnail = (
                        THUMBNAIL_OUTPUT / "thumbnail_gemini.png"
                    )

                    def render_thumbnail(prompt_text: str) -> None:
                        gemini.generate_image(
                            prompt=prompt_text,
                            output_directory=THUMBNAIL_OUTPUT,
                            filename=temp_thumbnail.name,
                            reference_image=reference,
                            new_conversation=True,
                        )

                    try:
                        render_thumbnail(thumbnail_prompt)
                    except GeminiLimitError as error:
                        blocked = True
                        limit_message = str(error)

                        print()
                        print("Gemini limit reached during thumbnail.")
                        print(limit_message)
                    except GeminiTextResponseError:
                        print()
                        print(
                            "Gemini answered the thumbnail prompt with text. "
                            "Retrying with a stricter image-only prompt."
                        )
                        try:
                            render_thumbnail(
                                thumbnail_prompt
                                + "\n\nCRITICAL: Generate the image now. "
                                "Render the exact title text inside the image. "
                                "Do NOT reply with text."
                            )
                        except GeminiLimitError as error:
                            blocked = True
                            limit_message = str(error)

                            print()
                            print("Gemini limit reached during thumbnail.")
                            print(limit_message)
                        except Exception as error:  # noqa: BLE001
                            print()
                            print(f"Thumbnail generation failed: {error}")
                    except Exception as error:  # noqa: BLE001
                        print()
                        print(f"Thumbnail generation failed: {error}")

                    if (
                        temp_thumbnail.exists()
                        and temp_thumbnail.stat().st_size > 0
                    ):
                        normalize_thumbnail(temp_thumbnail)
                        temp_thumbnail.replace(thumbnail_path)
                        print()
                        print("Gemini thumbnail created:")
                        print(thumbnail_path)
                    elif not blocked:
                        print()
                        print("Thumbnail was not generated.")

    finally:
        gemini.close()

    return blocked, limit_message


def wait_for_limit_reset(
    clips,
    visual_plans,
    title,
    thumbnail_plan,
    limit_message,
    story_text=None,
):
    """Retry image generation until Gemini's limit clears or we give up.

    Returns the new ``(blocked, limit_message)`` after the final attempt.
    """
    print()
    print("========================================")
    print("AUTO-RESUME ARMED")
    print("========================================")

    if limit_message:
        print()
        print(limit_message)

    print()
    print(
        "Waiting for Gemini's limit to reset. Retrying every "
        f"{LIMIT_POLL_SECONDS // 60} minute(s), up to "
        f"{LIMIT_MAX_WAIT_MINUTES} minute(s)."
    )
    print("Press Ctrl+C to stop; progress is checkpointed.")

    waited = 0

    while True:
        try:
            time.sleep(LIMIT_POLL_SECONDS)
        except KeyboardInterrupt:
            print()
            print("Auto-resume cancelled. Progress is checkpointed.")
            return True, limit_message

        waited += LIMIT_POLL_SECONDS

        print()
        print("========================================")
        print(
            f"RETRYING GEMINI IMAGES "
            f"(waited {waited // 60} minute(s))"
        )
        print("========================================")

        blocked, message = generate_images_and_thumbnail(
            clips,
            visual_plans,
            title,
            thumbnail_plan,
            story_text=story_text,
        )

        save_checkpoint(
            clips,
            stage="images",
            blocked=blocked,
            limit_message=message,
        )

        if not blocked:
            print()
            print("Gemini is available again. Resuming pipeline.")
            return False, None

        if (
            LIMIT_MAX_WAIT_MINUTES
            and waited >= LIMIT_MAX_WAIT_MINUTES * 60
        ):
            print()
            print(
                "Still limited after "
                f"{LIMIT_MAX_WAIT_MINUTES} minute(s). "
                "Stopping auto-resume; re-run main.py later."
            )
            return True, message


# ============================================================
# MAIN
# ============================================================

def generate_all_voices(clips, tts, blocked=False, limit_message=None):
    """Generate any missing narration WAVs and save the voice checkpoint.

    This is the same loop the serial pipeline always used; it is a separate
    function so the parallel voice worker can call it directly.
    """
    AUDIO_OUTPUT.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        print()
        print("========================================")
        print(f"CLIP {clip.id} — VOICE")
        print("========================================")

        audio_path = (
            AUDIO_OUTPUT
            / f"audio_{clip.id:03d}.wav"
        )

        if (
            RESUME_EXISTING
            and audio_path.exists()
            and audio_path.stat().st_size > 0
        ):
            print()
            print("Audio already exists, skipping:")
            print(audio_path)
            continue

        print()
        print("Generating voice...")

        use_agents = AGENTS is not None and AGENTS.enabled

        if use_agents:
            critic = AGENTS.voice_critic()
            max_attempts = 1 + AGENTS.config.max_voice_retries

            def generate_voice(attempt, overrides):
                candidate = AUDIO_OUTPUT / (
                    f"audio_{clip.id:03d}_candidate_{attempt}.wav"
                )
                print(
                    f"  Voice attempt {attempt}/{max_attempts}"
                    + (f" ({overrides})" if overrides else "")
                )
                if AGENT_STATE is not None:
                    AGENT_STATE.budget.spend_tts()
                tts.generate_to_file(
                    text=clip.script,
                    output_file=candidate,
                    **overrides,
                )
                return candidate

            def inspect_voice(path):
                return critic.inspect(path, clip.script)

            result = run_voice_qc_loop(
                generate=generate_voice,
                inspect=inspect_voice,
                final_path=audio_path,
                max_attempts=max_attempts,
                retry_overrides=(
                    lambda attempt, health: VoiceCritic.retry_overrides(
                        attempt,
                        health,
                        base_speed=CHATTERBOX_SPEED_FACTOR,
                    )
                ),
                fatal_exceptions=(RuntimeError, OSError),
            )

            if AGENT_STATE is not None:
                for attempt in result.attempts:
                    AGENT_STATE.record_attempt(
                        "voice",
                        attempt.index,
                        accepted=attempt.ok,
                        issues=attempt.issues,
                        clip=clip.id,
                    )

            if not (
                audio_path.exists()
                and audio_path.stat().st_size > 0
            ):
                print()
                print(
                    f"Voice for clip {clip.id} was not generated."
                )
                continue
        else:
            tts.generate_to_file(
                text=clip.script,
                output_file=audio_path,
            )

        print()
        print("Audio saved:")
        print(audio_path)

    save_checkpoint(
        clips,
        stage="voices",
        blocked=blocked,
        limit_message=limit_message,
    )


def finish_story(title, final_video, thumbnail_path, after_finish_action):
    """Print the completion summary and run the after-finish action."""
    print()
    print("========================================")
    print("STORY COMPLETE")
    print("========================================")

    print()
    print(f"Title:\n{title}")

    print()
    print(f"Final video:\n{final_video.resolve()}")

    print()
    print(f"Thumbnail:\n{thumbnail_path.resolve()}")

    print()
    print(f"Visual plan:\n{VISUAL_PLAN_FILE.resolve()}")

    print()
    print(f"Thumbnail plan:\n{THUMBNAIL_PLAN_FILE.resolve()}")

    print()
    print("Story generation and rendering completed successfully.")

    # MUST STAY LAST.
    perform_after_finish_action(after_finish_action)


def run_parallel_story(
    clips,
    visual_plans,
    title,
    thumbnail_plan,
    script_text,
    tts,
    editor,
    args,
):
    """Run the dependency-gated parallel workers for one story."""

    from agents.limits import LimitLedger
    from agents.orchestrator import ParallelPipeline, StagePaths

    def env_float(name, default):
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def env_int(name, default):
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    poll_seconds = getattr(args, "parallel_poll_seconds", None)
    if poll_seconds is None:
        poll_seconds = env_float("FLOW_PARALLEL_POLL_SECONDS", 5.0)

    cooldown_minutes = getattr(args, "gemini_cooldown_minutes", None)
    if cooldown_minutes is None:
        cooldown_minutes = env_float("FLOW_GEMINI_COOLDOWN_MINUTES", 60.0)

    max_wait_minutes = getattr(args, "max_wait_minutes", None)
    if max_wait_minutes is None:
        max_wait_minutes = env_float("FLOW_MAX_WAIT_MINUTES", 240.0)

    stage_retries = env_int("FLOW_STAGE_RETRIES", 3)
    stage_retry_seconds = env_float("FLOW_STAGE_RETRY_SECONDS", 30.0)

    ledger = LimitLedger(PROJECT_ROOT / "factory_state.json")
    paths = StagePaths(
        images=IMAGE_OUTPUT,
        audio=AUDIO_OUTPUT,
        video=VIDEO_OUTPUT,
        thumbnail=THUMBNAIL_OUTPUT,
    )
    plans = {
        int(key): value
        for key, value in visual_plans.items()
    }

    def generate_images():
        return generate_images_and_thumbnail(
            clips,
            visual_plans,
            title,
            thumbnail_plan,
            story_text=script_text,
        )

    def generate_voices():
        generate_all_voices(
            clips,
            tts,
            blocked=False,
            limit_message=None,
        )

    def render_clip(clip_id, plan):
        return editor.render_clip(
            clip_id,
            plan,
            fps=VIDEO_FPS,
            resolution=VIDEO_RESOLUTION,
            bitrate=VIDEO_BITRATE,
            fade_duration=FADE_DURATION,
            reuse_existing=(RESUME_EXISTING or RECAP_MISSING),
        )

    def assemble_final():
        clip_paths = [
            VIDEO_OUTPUT / f"clip_{clip.id:03d}.mp4"
            for clip in clips
        ]
        return editor.assemble_final_video(
            title=title,
            clip_paths=clip_paths,
            fps=VIDEO_FPS,
            resolution=VIDEO_RESOLUTION,
            bitrate=VIDEO_BITRATE,
            intro_duration=INTRO_DURATION,
            outro_duration=OUTRO_DURATION,
            reuse_existing=(RESUME_EXISTING or RECAP_MISSING),
        )

    def checkpoint(stage, blocked, message):
        save_checkpoint(
            clips,
            stage=stage,
            blocked=blocked,
            limit_message=message,
        )

    pipeline = ParallelPipeline(
        clips=clips,
        paths=paths,
        plans=plans,
        generate_images=generate_images,
        generate_voices=generate_voices,
        render_clip=render_clip,
        assemble_final=assemble_final,
        ledger=ledger,
        checkpoint=checkpoint,
        poll_seconds=poll_seconds,
        cooldown_minutes=cooldown_minutes,
        max_wait_minutes=max_wait_minutes,
        wait_for_limit=not getattr(args, "no_wait", False),
        max_stage_retries=stage_retries,
        stage_retry_seconds=stage_retry_seconds,
        log=print,
    )
    return pipeline.run()
def main():
    args = parse_args()


    global SCRIPT_PATH, STORY_NAME, STORY_OUTPUT, IMAGE_OUTPUT, AUDIO_OUTPUT
    global VIDEO_OUTPUT, THUMBNAIL_OUTPUT
    global VISUAL_PLAN_FILE, THUMBNAIL_PLAN_FILE
    global TITLE_FILE, CHECKPOINT_FILE, GEMINI_CONVERSATION_FILE
    global PRE_SELECTED_AFTER_FINISH, RECAP_MISSING, RESUME_EXISTING
    global FORCE_THUMBNAIL, THUMBNAIL_ONLY, KIDS_MODE
    global VOICES_ONLY

    RECAP_MISSING = bool(getattr(args, "recap", False))
    FORCE_THUMBNAIL = bool(getattr(args, "regenerate_thumbnail", False))
    THUMBNAIL_ONLY = bool(getattr(args, "thumbnail_only", False))
    KIDS_MODE = bool(getattr(args, "kids", False))
    VOICES_ONLY = bool(getattr(args, "voices_only", False))

    # A recap is a repair pass: it must never overwrite assets that already
    # exist, even if the user left the "Resume existing" checkbox unchecked.
    if RECAP_MISSING:
        RESUME_EXISTING = True

    # Optional: let Claude write the next story before the normal pipeline runs.
    if getattr(args, "generate_story", False):
        from ai.claude_browser import generate_story_with_claude, story_prompt
        from utils.story_scripts import save_generated_script

        print()
        print("Generating a new story with Claude...")
        generated_script = generate_story_with_claude(
            prompt=story_prompt(kids=KIDS_MODE)
        )
        generated_path, generated_story = save_generated_script(
            generated_script,
            input_dir=PROJECT_ROOT / "input",
            output_dir=OUTPUT_ROOT,
        )
        print()
        print(f"New script saved: {generated_path}")
        print(f"New story folder: {generated_story}")
        args.script = str(generated_path)
        args.story = generated_story

    SCRIPT_PATH = Path(args.script)

    if not SCRIPT_PATH.is_absolute():
        SCRIPT_PATH = PROJECT_ROOT / SCRIPT_PATH

    STORY_NAME = args.story
    STORY_OUTPUT = OUTPUT_ROOT / STORY_NAME
    IMAGE_OUTPUT = STORY_OUTPUT / "images"
    AUDIO_OUTPUT = STORY_OUTPUT / "audio"
    VIDEO_OUTPUT = STORY_OUTPUT / "video"
    THUMBNAIL_OUTPUT = STORY_OUTPUT / "thumbnail"
    VISUAL_PLAN_FILE = STORY_OUTPUT / "visual_plan.json"
    THUMBNAIL_PLAN_FILE = STORY_OUTPUT / "thumbnail_plan.json"
    TITLE_FILE = STORY_OUTPUT / "title.txt"
    DESCRIPTION_FILE = STORY_OUTPUT / "description.txt"
    CHECKPOINT_FILE = STORY_OUTPUT / "checkpoint.json"
    GEMINI_CONVERSATION_FILE = STORY_OUTPUT / "gemini_conversation.txt"

    # --------------------------------------------------------
    # LEVEL-2 CRITIC LAYER
    # --------------------------------------------------------
    global AGENT_CONFIG, AGENT_STATE, AGENTS

    AGENT_CONFIG = AgentConfig.from_args(args)
    if AGENT_CONFIG.enabled:
        AGENT_STATE = ProductionState.load_or_create(
            STORY_OUTPUT,
            STORY_NAME,
        )
        AGENT_STATE.data["agent_config"] = AGENT_CONFIG.to_dict()
    else:
        AGENT_STATE = None

    AGENTS = AgentRuntime(
        AGENT_CONFIG,
        AGENT_STATE,
        log=print,
    )

    print()
    print("Agent layer:")
    print(f"  {AGENT_CONFIG.describe()}")

    # CLI --after-finish takes precedence over GUI global
    if args.after_finish:
        PRE_SELECTED_AFTER_FINISH = args.after_finish

    print()
    print("========================================")
    print("AI VIDEO CREATOR")
    print("========================================")

    # ========================================================
    # ASK ACTION FIRST (right after launch)
    # ========================================================

    after_finish_action = (
        choose_after_finish_action(PRE_SELECTED_AFTER_FINISH)
    )

    print()
    print(
        f"Selected after-finish action: "
        f"{after_finish_action}"
    )

    # ========================================================
    # DIRECTORIES
    # ========================================================

    create_output_directories()

    print()
    print("Output folder:")
    print(STORY_OUTPUT.resolve())

    # ========================================================
    # LOAD SCRIPT
    # ========================================================

    print()
    print("Loading script...")

    script_text = load_script()

    print("Script loaded.")

    # ========================================================
    # PARSE
    # ========================================================

    print()
    print("Parsing script...")

    clips = parse_script(script_text)

    print(
        f"Found {len(clips)} clips."
    )

    if not clips:
        raise RuntimeError(
            "No clips were found."
        )

    # ========================================================
    # VALIDATE
    # ========================================================

    print()
    print("Validating script...")

    errors = validate_clips(clips)

    if errors:
        print()
        print("SCRIPT ERRORS:")

        for error in errors:
            print("-", error)

        raise SystemExit(1)

    print("Script validation successful.")

    # ========================================================
    # VOICE-ONLY PASS
    # ========================================================

    if VOICES_ONLY:
        print()
        print("========================================")
        print("VOICE-ONLY PASS")
        print("========================================")

        tts = ChatterboxTTS(
            server=CHATTERBOX_SERVER_URL,
            voice=CHATTERBOX_VOICE,
            language=CHATTERBOX_LANGUAGE,
            temperature=CHATTERBOX_TEMPERATURE,
            exaggeration=CHATTERBOX_EXAGGERATION,
            cfg_weight=CHATTERBOX_CFG_WEIGHT,
            seed=CHATTERBOX_SEED,
            speed_factor=CHATTERBOX_SPEED_FACTOR,
        )

        try:
            generate_all_voices(clips, tts)
        except (RuntimeError, OSError) as error:
            print(f"Voice-only pass stopped: {error}")
            raise SystemExit(2)

        print()
        print("Voice-only pass complete.")
        return

    # ========================================================
    # OLLAMA
    # ========================================================

    print()
    print("Starting Ollama visual planner (auto-starting Ollama if needed)...")

    visual_planner = OllamaVideoPlanner()

    # ========================================================
    # TITLE
    # ========================================================

    print()
    print("Generating story title...")

    title = None

    if RESUME_EXISTING and TITLE_FILE.exists():
        saved_title = TITLE_FILE.read_text(
            encoding="utf-8",
        ).strip()

        if saved_title and saved_title.lower() != "untitled":
            title = saved_title
            print(f"Reusing saved title: {title}")

    if title is None:
        title = visual_planner.generate_title(clips)
        title = title.strip()

        if not title or title.lower() == "untitled":
            raise RuntimeError(
                "Ollama failed to generate a valid story title."
            )

        TITLE_FILE.write_text(
            title,
            encoding="utf-8",
        )

    # Generate description (reuse the saved one on resume/thumbnail-only runs)
    print()
    description = None
    if RESUME_EXISTING and DESCRIPTION_FILE.exists():
        saved_description = DESCRIPTION_FILE.read_text(
            encoding="utf-8",
        ).strip()
        if saved_description:
            description = saved_description
            print("Reusing saved description.")

    if description is None:
        print("Generating video description...")
        description = visual_planner.generate_description(clips, title)
        DESCRIPTION_FILE.write_text(
            description,
            encoding="utf-8",
        )
        print(f"Description saved to {DESCRIPTION_FILE}")

    print()
    print(f"Generated title: {title}")
    print(f"Description: {description[:120]}...")

    # ========================================================
    # VISUAL PLAN
    # ========================================================

    print()
    print("Creating visual plan with Ollama...")

    visual_plans = None

    if RESUME_EXISTING and VISUAL_PLAN_FILE.exists():
        cached_plans = load_json(VISUAL_PLAN_FILE)

        if isinstance(cached_plans, dict):
            try:
                cached_plans = {
                    int(key): value
                    for key, value in cached_plans.items()
                }
            except (TypeError, ValueError):
                cached_plans = None

            if (
                isinstance(cached_plans, dict)
                and len(cached_plans) == len(clips)
                and all(clip.id in cached_plans for clip in clips)
            ):
                visual_plans = cached_plans

    if visual_plans is None:
        visual_plans = visual_planner.plan_clips(clips)
        print(
            f"Visual plan created for {len(visual_plans)} clips."
        )
    else:
        print(
            f"Reusing saved visual plan for "
            f"{len(visual_plans)} clips."
        )

    # HARD DISABLE CAMERA SHAKE/VIBRATION.
    # Even if Ollama or an old cached plan returns a shake value,
    # the renderer will receive only "none".
    for _clip_id, _plan in visual_plans.items():
        _plan["shake"] = "none"

    for clip_id in sorted(visual_plans):
        plan = visual_plans[clip_id]

        print()
        print(f"Clip {clip_id} visual plan:")
        print(f"  Camera: {plan['camera']}")
        print(f"  Motion: {plan['motion']}")
        print(f"  Shake: {plan['shake']}")
        print(f"  Transition: {plan['transition_after']}")
        print(f"  Grade: {plan['grade']}")

    save_json(
        VISUAL_PLAN_FILE,
        visual_plans,
    )

    # ========================================================
    # THUMBNAIL PLAN
    # ========================================================

    print()
    print("Creating professional thumbnail plan...")

    thumbnail_plan = None

    if RESUME_EXISTING and THUMBNAIL_PLAN_FILE.exists():
        cached_thumbnail = load_json(THUMBNAIL_PLAN_FILE)

        if (
            isinstance(cached_thumbnail, dict)
            and "subject" in cached_thumbnail
            and "title_position" in cached_thumbnail
        ):
            thumbnail_plan = cached_thumbnail
            print("Reusing saved thumbnail plan.")

    if thumbnail_plan is None:
        thumbnail_plan = visual_planner.generate_thumbnail_plan(
            clips=clips,
            title=title,
        )

    print()
    print("Thumbnail plan:")
    print(f"  Subject: {thumbnail_plan['subject']}")
    print(f"  Scene: {thumbnail_plan['scene']}")
    print(f"  Emotion: {thumbnail_plan['emotion']}")
    print(f"  Camera: {thumbnail_plan['camera']}")
    print(f"  Title position: {thumbnail_plan['title_position']}")

    save_json(
        THUMBNAIL_PLAN_FILE,
        thumbnail_plan,
    )

    # ========================================================
    # SERVICES
    # ========================================================

    tts = ChatterboxTTS(
        server=CHATTERBOX_SERVER_URL,
        voice=CHATTERBOX_VOICE,
        language=CHATTERBOX_LANGUAGE,
        temperature=CHATTERBOX_TEMPERATURE,
        exaggeration=CHATTERBOX_EXAGGERATION,
        cfg_weight=CHATTERBOX_CFG_WEIGHT,
        seed=CHATTERBOX_SEED,
        speed_factor=CHATTERBOX_SPEED_FACTOR,
    )

    print_progress(build_progress(clips))

    # ========================================================
    # PHASE 1 — IMAGES (all clips first)
    # ========================================================

    if RECAP_MISSING:
        print_recap_report(build_progress(clips))

    # ========================================================
    # PARALLEL STAGE WORKERS (default)
    # ========================================================

    if not getattr(args, "no_parallel", False) and not THUMBNAIL_ONLY:
        thumbnail_path = THUMBNAIL_OUTPUT / "thumbnail.png"

        print()
        print("========================================")
        print("PARALLEL STAGE WORKERS")
        print("========================================")

        editor = VideoEditor(
            image_directory=IMAGE_OUTPUT,
            audio_directory=AUDIO_OUTPUT,
            output_directory=VIDEO_OUTPUT,
        )

        result = run_parallel_story(
            clips=clips,
            visual_plans=visual_plans,
            title=title,
            thumbnail_plan=thumbnail_plan,
            script_text=script_text,
            tts=tts,
            editor=editor,
            args=args,
        )

        if result.complete and result.final_video is not None:
            save_checkpoint(
                clips,
                stage="complete",
                blocked=False,
                limit_message=None,
            )
            finish_story(
                title,
                result.final_video,
                thumbnail_path,
                after_finish_action,
            )
            return

        if result.errors:
            print()
            print("WORKER ERRORS:")
            for error in result.errors:
                print(f"  - {error}")

        if result.missing_images:
            pause_stage = "images"
        elif result.missing_voices:
            pause_stage = "voices"
        else:
            pause_stage = "clips"

        save_checkpoint(
            clips,
            stage=pause_stage,
            blocked=result.gemini_blocked,
            limit_message=result.limit_message,
        )

        print()
        print("========================================")
        print(
            "PAUSED — WAITING ON GEMINI"
            if result.gemini_blocked
            else "PAUSED — INCOMPLETE ASSETS"
        )
        print("========================================")

        if result.limit_message:
            print()
            print(result.limit_message)

        print()
        print("Remaining to finish:")

        if result.missing_images:
            print(
                "  - Images: "
                + ", ".join(str(i) for i in result.missing_images)
            )

        if result.missing_voices:
            print(
                "  - Voices: "
                + ", ".join(str(i) for i in result.missing_voices)
            )

        if result.missing_clips:
            print(
                "  - Clips: "
                + ", ".join(str(i) for i in result.missing_clips)
            )

        if not thumbnail_path.exists():
            print("  - Thumbnail")

        print()
        print(f"Checkpoint saved to:\n{CHECKPOINT_FILE.resolve()}")
        print()
        print("Re-run the same command to resume. Completed assets are")
        print("skipped, and the limiter cooldown is remembered in")
        print("factory_state.json.")
        return

    gemini_blocked, gemini_limit_message = (
        generate_images_and_thumbnail(
            clips,
            visual_plans,
            title,
            thumbnail_plan,
            story_text=script_text,
        )
    )

    save_checkpoint(
        clips,
        stage="images",
        blocked=gemini_blocked,
        limit_message=gemini_limit_message,
    )

    thumbnail_path = THUMBNAIL_OUTPUT / "thumbnail.png"

    if THUMBNAIL_ONLY:
        print()
        if thumbnail_path.exists() and thumbnail_path.stat().st_size > 0:
            print("Thumbnail-only run complete:")
            print(thumbnail_path.resolve())
            return
        print("Thumbnail-only run did not produce a thumbnail.")
        if gemini_limit_message:
            print(gemini_limit_message)
        return

    # ========================================================
    # PHASE 2 — VOICE (runs even if Gemini is limited)
    # ========================================================

    generate_all_voices(
        clips,
        tts,
        blocked=gemini_blocked,
        limit_message=gemini_limit_message,
    )

    # ========================================================
    # COMPLETENESS CHECK + AUTO-RESUME
    # ========================================================

    progress = build_progress(clips)

    can_render = (
        not progress["images_pending"]
        and not progress["voices_pending"]
        and progress["thumbnail"]
    )

    # If Gemini's limit is the only thing holding us back, wait for the
    # reset and continue automatically instead of throwing the run away.
    if (
        not can_render
        and gemini_blocked
        and AUTO_RESUME_AFTER_LIMIT
        and (
            progress["images_pending"]
            or not progress["thumbnail"]
        )
    ):
        gemini_blocked, gemini_limit_message = wait_for_limit_reset(
            clips,
            visual_plans,
            title,
            thumbnail_plan,
            gemini_limit_message,
            story_text=script_text,
        )

        progress = build_progress(clips)

        can_render = (
            not progress["images_pending"]
            and not progress["voices_pending"]
            and progress["thumbnail"]
        )

    if not can_render:
        print()
        print("========================================")
        print(
            "PAUSED — WAITING ON GEMINI"
            if gemini_blocked
            else "PAUSED — INCOMPLETE ASSETS"
        )
        print("========================================")

        if gemini_limit_message:
            print()
            print(gemini_limit_message)

        print()
        print("The pipeline did all the work that does not")
        print("need Gemini (voice). Remaining to finish:")

        if progress["images_pending"]:
            print(
                "  - Images: "
                + ", ".join(str(i) for i in progress["images_pending"])
            )

        if progress["voices_pending"]:
            print(
                "  - Voices: "
                + ", ".join(str(i) for i in progress["voices_pending"])
            )

        if not progress["thumbnail"]:
            print("  - Thumbnail")

        print()
        print(f"Checkpoint saved to:\n{CHECKPOINT_FILE.resolve()}")
        print()
        print("When Gemini is ready, run main.py again or use")
        print("Recap missing — it will skip everything already")
        print("finished and regenerate only what is missing.")

        return

    # ========================================================
    # THUMBNAIL (already rendered by Gemini)
    # ========================================================

    if not (
        thumbnail_path.exists()
        and thumbnail_path.stat().st_size > 0
    ):
        raise RuntimeError(
            "Gemini's thumbnail is missing. Re-run with "
            "--regenerate-thumbnail --thumbnail-only."
        )

    print()
    print("Gemini thumbnail ready:")
    print(thumbnail_path.resolve())

    # ========================================================
    # VIDEO EDITOR
    # ========================================================

    print()
    print("========================================")
    print("STARTING VIDEO EDITOR")
    print("========================================")

    editor = VideoEditor(
        image_directory=IMAGE_OUTPUT,
        audio_directory=AUDIO_OUTPUT,
        output_directory=VIDEO_OUTPUT,
    )

    final_video = editor.create_final_video(
        plans=visual_plans,
        title=title,
        fps=VIDEO_FPS,
        resolution=VIDEO_RESOLUTION,
        bitrate=VIDEO_BITRATE,
        fade_duration=FADE_DURATION,
        intro_duration=INTRO_DURATION,
        outro_duration=OUTRO_DURATION,
        reuse_existing=(RESUME_EXISTING or RECAP_MISSING),
    )

    save_checkpoint(
        clips,
        stage="complete",
        blocked=False,
        limit_message=None,
    )

    finish_story(
        title,
        final_video,
        thumbnail_path,
        after_finish_action,
    )


if __name__ == "__main__":
    main()