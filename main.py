import json
import shutil
import time
from pathlib import Path
import os
import subprocess

from parser.script_parser import parse_script
from utils.validator import validate_clips

from ai.gemini_browser import GeminiBrowser
from ai.ollama_video import OllamaVideoPlanner

from audio.voicebox import Voicebox
from audio.sfx_selector import SFXSelector

from editor.video_editor import VideoEditor
from editor.thumbnail import ThumbnailCreator


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path("input/script.txt")

OUTPUT_ROOT = Path("output")
STORY_NAME = "story_001"

STORY_OUTPUT = OUTPUT_ROOT / STORY_NAME

IMAGE_OUTPUT = STORY_OUTPUT / "images"
AUDIO_OUTPUT = STORY_OUTPUT / "audio"
SFX_OUTPUT = STORY_OUTPUT / "sfx"
VIDEO_OUTPUT = STORY_OUTPUT / "video"
THUMBNAIL_OUTPUT = STORY_OUTPUT / "thumbnail"

SFX_SELECTION_FILE = STORY_OUTPUT / "sfx_selection.json"
VISUAL_PLAN_FILE = STORY_OUTPUT / "visual_plan.json"
THUMBNAIL_PLAN_FILE = STORY_OUTPUT / "thumbnail_plan.json"
TITLE_FILE = STORY_OUTPUT / "title.txt"

SFX_LIBRARY = Path("sfx")


# ============================================================
# SETTINGS
# ============================================================

VIDEO_FPS = 30
VIDEO_RESOLUTION = (1920, 1080)
VIDEO_BITRATE = "8M"

FADE_DURATION = 0.4
SFX_VOLUME = 0.15
SFX_START_RATIO = 0.35

INTRO_DURATION = 3.0
OUTRO_DURATION = 4.0

THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT = 720

AFTER_FINISH_COUNTDOWN = 15  # seconds before sleep/shutdown (Ctrl+C cancels)


# ============================================================
# SCRIPT
# ============================================================

def load_script():
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"Script file not found:\n{SCRIPT_PATH.resolve()}"
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
        SFX_OUTPUT,
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

def build_clip_prompt(clip, visual_plan):
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

    return prompt


# ============================================================
# THUMBNAIL PROMPT
# ============================================================

def build_thumbnail_prompt(title, thumbnail_plan):
    return f"""
Create ONE professional cinematic thumbnail image for a film/story video.

Video title:
{title}

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

Important thumbnail requirements:
- Professional movie-poster / YouTube-thumbnail composition.
- 16:9 landscape.
- Strong focal point.
- One dominant subject.
- High contrast.
- Cinematic lighting.
- Strong foreground/background separation.
- Clear silhouette.
- Dramatic but believable visual storytelling.
- High detail.
- Photorealistic cinematic quality.
- Keep the important subject large enough to be visible at small thumbnail size.
- Leave clean negative space on the {thumbnail_plan["title_position"].replace("_", " ")} for the title overlay.
- Do NOT put any text in the image.
- No subtitles.
- No logos.
- No watermark.
- No UI.
- No borders.
- No collage.
- One image only.
""".strip()


# ============================================================
# ACTION AFTER FINISH
# ============================================================

def choose_after_finish_action():
    """
    Ask what Windows should do after the video pipeline
    finishes successfully. Called once, at startup.
    """

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


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("========================================")
    print("AI VIDEO CREATOR")
    print("========================================")

    # ========================================================
    # ASK ACTION FIRST (right after launch)
    # ========================================================

    after_finish_action = (
        choose_after_finish_action()
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
    # OLLAMA
    # ========================================================

    print()
    print("Starting Ollama visual planner...")

    visual_planner = OllamaVideoPlanner()

    # ========================================================
    # TITLE
    # ========================================================

    print()
    print("Generating story title...")

    title = visual_planner.generate_title(clips)
    title = title.strip()

    if not title or title.lower() == "untitled":
        raise RuntimeError(
            "Ollama failed to generate a valid story title."
        )

    print()
    print(f"Generated title: {title}")

    TITLE_FILE.write_text(
        title,
        encoding="utf-8",
    )

    # ========================================================
    # VISUAL PLAN
    # ========================================================

    print()
    print("Creating visual plan with Ollama...")

    visual_plans = visual_planner.plan_clips(clips)

    # HARD DISABLE CAMERA SHAKE/VIBRATION.
    # Even if Ollama or an old cached plan returns a shake value,
    # the renderer will receive only "none".
    for _clip_id, _plan in visual_plans.items():
        _plan["shake"] = "none"

    print(
        f"Visual plan created for {len(visual_plans)} clips."
    )

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
    # SFX LIBRARY
    # ========================================================

    if not SFX_LIBRARY.exists():
        raise FileNotFoundError(
            "SFX library was not found:\n"
            f"{SFX_LIBRARY.resolve()}"
        )

    print()
    print("Loading SFX library...")

    sfx_selector = SFXSelector(
        sfx_directory=SFX_LIBRARY,
    )

    print(
        f"Found {len(sfx_selector.library.files)} SFX files."
    )

    # ========================================================
    # SERVICES
    # ========================================================

    gemini = GeminiBrowser()

    voicebox = Voicebox(
        profile="test1"
    )

    sfx_selection_data = {}

    try:
        # ====================================================
        # GEMINI
        # ====================================================

        print()
        print("Connecting to existing Chrome...")

        gemini.connect()
        gemini.open_gemini()

        # ====================================================
        # STORY CLIPS
        # ====================================================

        for clip in clips:
            print()
            print("========================================")
            print(f"PROCESSING CLIP {clip.id}")
            print("========================================")

            visual_plan = visual_plans.get(clip.id)

            if visual_plan is None:
                raise RuntimeError(
                    f"No visual plan found for clip {clip.id}."
                )

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            print()
            print("Generating Gemini image...")

            image_prompt = build_clip_prompt(
                clip,
                visual_plan,
            )

            image_path = gemini.generate_image(
                prompt=image_prompt,
                output_directory=IMAGE_OUTPUT,
                filename=f"image_{clip.id:03d}.png",
            )

            print()
            print("Image saved:")
            print(image_path)

            # ------------------------------------------------
            # VOICE
            # ------------------------------------------------

            print()
            print("Generating voice...")

            audio_path = (
                AUDIO_OUTPUT
                / f"audio_{clip.id:03d}.wav"
            )

            voicebox.generate_to_file(
                text=clip.script,
                output_file=audio_path,
            )

            print()
            print("Audio saved:")
            print(audio_path)

            # ------------------------------------------------
            # SFX
            # ------------------------------------------------

            print()
            print("Analyzing sound effects...")

            sfx_result = sfx_selector.select_for_clip(clip)
            selected_sfx = sfx_result.get("selected")
            selected_output = None

            if selected_sfx:
                selected_sfx = Path(selected_sfx)

                output_sfx_path = (
                    SFX_OUTPUT
                    / f"sfx_{clip.id:03d}"
                    f"{selected_sfx.suffix}"
                )

                shutil.copy2(
                    selected_sfx,
                    output_sfx_path,
                )

                selected_output = output_sfx_path

                print()
                print("SFX copied:")
                print(output_sfx_path)

            else:
                print()
                print("No SFX selected.")

            sfx_selection_data[str(clip.id)] = {
                "keywords": sfx_result.get(
                    "keywords",
                    [],
                ),
                "candidates": sfx_result.get(
                    "candidates",
                    [],
                ),
                "selected_source": (
                    str(selected_sfx)
                    if selected_sfx
                    else None
                ),
                "selected_output": (
                    str(selected_output)
                    if selected_output
                    else None
                ),
            }

            print()
            print(
                f"Clip {clip.id} completed."
            )

        # ====================================================
        # THUMBNAIL ARTWORK
        # ====================================================

        print()
        print("========================================")
        print("GENERATING THUMBNAIL")
        print("========================================")

        thumbnail_prompt = build_thumbnail_prompt(
            title=title,
            thumbnail_plan=thumbnail_plan,
        )

        thumbnail_background = gemini.generate_image(
            prompt=thumbnail_prompt,
            output_directory=THUMBNAIL_OUTPUT,
            filename="thumbnail_background.png",
        )

        print()
        print("Thumbnail background generated:")
        print(thumbnail_background)

    finally:
        gemini.close()

    # ========================================================
    # THUMBNAIL TITLE OVERLAY
    # ========================================================

    thumbnail_creator = ThumbnailCreator(
        output_directory=THUMBNAIL_OUTPUT,
        width=THUMBNAIL_WIDTH,
        height=THUMBNAIL_HEIGHT,
    )

    thumbnail_path = thumbnail_creator.create(
        source_image=thumbnail_background,
        title=title,
        title_position=thumbnail_plan[
            "title_position"
        ],
    )

    print()
    print("Professional thumbnail created:")
    print(thumbnail_path.resolve())

    # ========================================================
    # SAVE METADATA
    # ========================================================

    save_json(
        SFX_SELECTION_FILE,
        sfx_selection_data,
    )

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
        sfx_directory=SFX_OUTPUT,
        output_directory=VIDEO_OUTPUT,
    )

    final_video = editor.create_final_video(
        plans=visual_plans,
        title=title,
        fps=VIDEO_FPS,
        resolution=VIDEO_RESOLUTION,
        bitrate=VIDEO_BITRATE,
        fade_duration=FADE_DURATION,
        sfx_volume=SFX_VOLUME,
        sfx_start_ratio=SFX_START_RATIO,
        intro_duration=INTRO_DURATION,
        outro_duration=OUTRO_DURATION,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

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
    print(f"SFX metadata:\n{SFX_SELECTION_FILE.resolve()}")

    print()
    print("Story generation and rendering completed successfully.")

    # ========================================================
    # AFTER FINISH ACTION (MUST STAY LAST)
    # ========================================================

    perform_after_finish_action(
        after_finish_action
    )


if __name__ == "__main__":
    main()