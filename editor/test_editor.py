import json

from parser.script_parser import extract_title
from editor.video_editor import VideoEditor


STORY_OUTPUT = "output/story_001"

SCRIPT_PATH = "input/script.txt"

IMAGE_DIRECTORY = (
    f"{STORY_OUTPUT}/images"
)

AUDIO_DIRECTORY = (
    f"{STORY_OUTPUT}/audio"
)

SFX_DIRECTORY = (
    f"{STORY_OUTPUT}/sfx"
)

VIDEO_DIRECTORY = (
    f"{STORY_OUTPUT}/video"
)

VISUAL_PLAN_FILE = (
    f"{STORY_OUTPUT}/visual_plan.json"
)


def load_title():

    with open(
        SCRIPT_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        script_text = file.read()

    title = extract_title(
        script_text
    )

    print(
        f"Story title: {title}"
    )

    return title


def load_visual_plan():

    with open(
        VISUAL_PLAN_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return {
        int(key): value
        for key, value in data.items()
    }


def main():

    title = load_title()

    plans = load_visual_plan()

    editor = VideoEditor(
        image_directory=IMAGE_DIRECTORY,
        audio_directory=AUDIO_DIRECTORY,
        sfx_directory=SFX_DIRECTORY,
        output_directory=VIDEO_DIRECTORY,
    )

    final_video = editor.create_final_video(
        plans=plans,

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

    print()
    print(
        "Final video:"
    )

    print(
        final_video.resolve()
    )


if __name__ == "__main__":
    main()