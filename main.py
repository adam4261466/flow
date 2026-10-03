import json
import shutil
from pathlib import Path

from parser.script_parser import parse_script
from utils.validator import validate_clips

from ai.gemini_browser import GeminiBrowser
from ai.ollama_video import OllamaVideoPlanner

from audio.voicebox import Voicebox
from audio.sfx_selector import SFXSelector

from editor.video_editor import VideoEditor


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(
    "input/script.txt"
)

OUTPUT_ROOT = Path(
    "output"
)

STORY_NAME = "story_001"

STORY_OUTPUT = (
    OUTPUT_ROOT
    / STORY_NAME
)

IMAGE_OUTPUT = (
    STORY_OUTPUT
    / "images"
)

AUDIO_OUTPUT = (
    STORY_OUTPUT
    / "audio"
)

SFX_OUTPUT = (
    STORY_OUTPUT
    / "sfx"
)

VIDEO_OUTPUT = (
    STORY_OUTPUT
    / "video"
)

SFX_SELECTION_FILE = (
    STORY_OUTPUT
    / "sfx_selection.json"
)

VISUAL_PLAN_FILE = (
    STORY_OUTPUT
    / "visual_plan.json"
)

TITLE_FILE = (
    STORY_OUTPUT
    / "title.txt"
)

SFX_LIBRARY = Path(
    "sfx"
)


# ============================================================
# SCRIPT
# ============================================================

def load_script():

    if not SCRIPT_PATH.exists():

        raise FileNotFoundError(
            f"Script file not found:\n"
            f"{SCRIPT_PATH.resolve()}"
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

    IMAGE_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIO_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SFX_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    VIDEO_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_clip_prompt(
    clip,
    visual_plan,
):

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
# MAIN
# ============================================================

def main():

    print()
    print("========================================")
    print("AI VIDEO CREATOR")
    print("========================================")

    # ========================================================
    # DIRECTORIES
    # ========================================================

    create_output_directories()

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

    clips = parse_script(
        script_text
    )

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

    errors = validate_clips(
        clips
    )

    if errors:

        print()
        print("SCRIPT ERRORS:")

        for error in errors:

            print(
                "-",
                error,
            )

        raise SystemExit(1)

    print(
        "Script validation successful."
    )

    # ========================================================
    # OLLAMA
    # ========================================================

    print()
    print(
        "Starting Ollama visual planner..."
    )

    visual_planner = (
        OllamaVideoPlanner()
    )

    # ========================================================
    # GENERATE STORY TITLE
    # ========================================================

    print()
    print(
        "Generating story title..."
    )

    title = (
        visual_planner.generate_title(
            clips
        )
    )

    title = title.strip()

    if (
        not title
        or title.lower() == "untitled"
    ):
        raise RuntimeError(
            "Ollama failed to generate a valid "
            "story title."
        )

    print()
    print(
        f"Generated title: {title}"
    )

    # Save title for the editor/test editor.
    TITLE_FILE.write_text(
        title,
        encoding="utf-8",
    )

    print()
    print(
        "Title saved:"
    )

    print(
        TITLE_FILE.resolve()
    )

    # ========================================================
    # OLLAMA VISUAL PLAN
    # ========================================================

    print()
    print(
        "Creating visual plan with Ollama..."
    )

    visual_plans = (
        visual_planner.plan_clips(
            clips
        )
    )

    print(
        f"Visual plan created for "
        f"{len(visual_plans)} clips."
    )

    # ========================================================
    # SHOW VISUAL PLAN
    # ========================================================

    for clip_id in sorted(
        visual_plans
    ):

        plan = visual_plans[
            clip_id
        ]

        print()
        print(
            f"Clip {clip_id} visual plan:"
        )

        print(
            f"  Camera: "
            f"{plan['camera']}"
        )

        print(
            f"  Motion: "
            f"{plan['motion']}"
        )

        print(
            f"  Shake: "
            f"{plan['shake']}"
        )

        print(
            f"  Transition: "
            f"{plan['transition_after']}"
        )

        print(
            f"  Grade: "
            f"{plan['grade']}"
        )

    # ========================================================
    # SAVE VISUAL PLAN
    # ========================================================

    with open(
        VISUAL_PLAN_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            visual_plans,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print(
        "Visual plan saved:"
    )

    print(
        VISUAL_PLAN_FILE.resolve()
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
    print(
        "Loading SFX library..."
    )

    sfx_selector = SFXSelector(
        sfx_directory=SFX_LIBRARY,
    )

    print(
        f"Found "
        f"{len(sfx_selector.library.files)} "
        f"SFX files."
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
        print(
            "Connecting to existing Chrome..."
        )

        gemini.connect()

        gemini.open_gemini()

        # ====================================================
        # PROCESS CLIPS
        # ====================================================

        for clip in clips:

            print()
            print(
                "========================================"
            )

            print(
                f"PROCESSING CLIP {clip.id}"
            )

            print(
                "========================================"
            )

            visual_plan = visual_plans.get(
                clip.id
            )

            if visual_plan is None:

                raise RuntimeError(
                    f"No visual plan found "
                    f"for clip {clip.id}."
                )

            # =================================================
            # IMAGE
            # =================================================

            print()
            print(
                "Generating Gemini image..."
            )

            image_prompt = (
                build_clip_prompt(
                    clip,
                    visual_plan,
                )
            )

            image_path = (
                gemini.generate_image(
                    prompt=image_prompt,
                    output_directory=IMAGE_OUTPUT,
                    filename=(
                        f"image_{clip.id:03d}.png"
                    ),
                )
            )

            print()
            print(
                "Image saved:"
            )

            print(
                image_path
            )

            # =================================================
            # VOICE
            # =================================================

            print()
            print(
                "Generating voice..."
            )

            audio_path = (
                AUDIO_OUTPUT
                / f"audio_{clip.id:03d}.wav"
            )

            voicebox.generate_to_file(
                text=clip.script,
                output_file=audio_path,
            )

            print()
            print(
                "Audio saved:"
            )

            print(
                audio_path
            )

            # =================================================
            # SFX
            # =================================================

            print()
            print(
                "Analyzing sound effects..."
            )

            sfx_result = (
                sfx_selector.select_for_clip(
                    clip
                )
            )

            selected_sfx = (
                sfx_result.get(
                    "selected"
                )
            )

            selected_output = None

            if selected_sfx:

                selected_sfx = Path(
                    selected_sfx
                )

                output_sfx_path = (
                    SFX_OUTPUT
                    / f"sfx_{clip.id:03d}"
                    f"{selected_sfx.suffix}"
                )

                shutil.copy2(
                    selected_sfx,
                    output_sfx_path,
                )

                selected_output = (
                    output_sfx_path
                )

                print()
                print(
                    "SFX copied:"
                )

                print(
                    output_sfx_path
                )

            else:

                print()
                print(
                    "No SFX selected."
                )

            # =================================================
            # STORE SFX INFORMATION
            # =================================================

            sfx_selection_data[
                str(clip.id)
            ] = {

                "keywords":
                    sfx_result.get(
                        "keywords",
                        [],
                    ),

                "candidates":
                    sfx_result.get(
                        "candidates",
                        [],
                    ),

                "selected_source":
                    (
                        str(selected_sfx)
                        if selected_sfx
                        else None
                    ),

                "selected_output":
                    (
                        str(selected_output)
                        if selected_output
                        else None
                    ),
            }

            print()
            print(
                f"Clip {clip.id} completed."
            )

    finally:

        gemini.close()

    # ========================================================
    # SAVE SFX METADATA
    # ========================================================

    with open(
        SFX_SELECTION_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            sfx_selection_data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print(
        "SFX metadata saved:"
    )

    print(
        SFX_SELECTION_FILE.resolve()
    )

    # ========================================================
    # RENDER VIDEO
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "STARTING VIDEO EDITOR"
    )

    print(
        "========================================"
    )

    editor = VideoEditor(
        image_directory=IMAGE_OUTPUT,
        audio_directory=AUDIO_OUTPUT,
        sfx_directory=SFX_OUTPUT,
        output_directory=VIDEO_OUTPUT,
    )

    final_video = (
        editor.create_final_video(
            plans=visual_plans,

            title=title,

            fps=30,

            resolution=(
                1920,
                1080,
            ),

            bitrate="8M",

            fade_duration=0.4,

            sfx_volume=0.30,

            sfx_start_ratio=0.35,

            intro_duration=3.0,

            outro_duration=4.0,
        )
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "STORY COMPLETE"
    )

    print(
        "========================================"
    )

    print()
    print(
        f"Title:\n"
        f"{title}"
    )

    print()
    print(
        f"Final video:\n"
        f"{final_video.resolve()}"
    )

    print()
    print(
        "Visual plan:"
    )

    print(
        VISUAL_PLAN_FILE.resolve()
    )

    print()
    print(
        "Title file:"
    )

    print(
        TITLE_FILE.resolve()
    )


if __name__ == "__main__":
    main()