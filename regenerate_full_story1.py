#!/usr/bin/env python3
"""Regenerate story_001 visual plan for all 99 clips, then missing images, then video."""
import sys
from pathlib import Path
import json
import shutil

sys.path.insert(0, str(Path(__file__).parent))

from main import (
    parse_script,
    validate_clips,
    OllamaVideoPlanner,
    VideoEditor,
    save_json,
    build_clip_prompt,
    build_thumbnail_prompt,
)

# Explicit paths for story_001
OUTPUT_ROOT = Path("output")
STORY_NAME = "story_001"
STORY_OUTPUT = OUTPUT_ROOT / STORY_NAME
IMAGE_OUTPUT = STORY_OUTPUT / "images"
AUDIO_OUTPUT = STORY_OUTPUT / "audio"
VIDEO_OUTPUT = STORY_OUTPUT / "video"
THUMBNAIL_OUTPUT = STORY_OUTPUT / "thumbnail"
VISUAL_PLAN_FILE = STORY_OUTPUT / "visual_plan.json"
THUMBNAIL_PLAN_FILE = STORY_OUTPUT / "thumbnail_plan.json"
TITLE_FILE = STORY_OUTPUT / "title.txt"
GEMINI_CONVERSATION_FILE = STORY_OUTPUT / "gemini_conversation.txt"

SCRIPT_PATH = Path("input/script1.txt")

# 1) Reload script and parse all 99 clips
print("Loading script...")
script_text = SCRIPT_PATH.read_text(encoding="utf-8")
clips = parse_script(script_text)
print(f"Parsed {len(clips)} clips")

errors = validate_clips(clips)
if errors:
    for e in errors:
        print("ERROR:", e)
    sys.exit(1)

# 2) Regenerate visual plan for ALL clips
print("\nGenerating visual plan for all 99 clips...")
planner = OllamaVideoPlanner()
visual_plans = planner.plan_clips(clips)
print(f"Visual plan created for {len(visual_plans)} clips")

# Disable shake
for _clip_id, _plan in visual_plans.items():
    _plan["shake"] = "none"

save_json(VISUAL_PLAN_FILE, visual_plans)

# 3) Regenerate thumbnail plan
print("\nGenerating thumbnail plan...")
title = TITLE_FILE.read_text(encoding="utf-8").strip()
thumbnail_plan = planner.generate_thumbnail_plan(clips=clips, title=title)
save_json(THUMBNAIL_PLAN_FILE, thumbnail_plan)

# 4) Generate missing images (clips 85-99)
print("\nChecking for missing images...")
from ai.gemini_browser import GeminiBrowser, GeminiLimitError

gemini = GeminiBrowser(conversation_file=GEMINI_CONVERSATION_FILE)
try:
    gemini.connect()
    gemini.open_gemini()

    for clip in clips:
        img_path = IMAGE_OUTPUT / f"image_{clip.id:03d}.png"
        if img_path.exists() and img_path.stat().st_size > 0:
            continue

        visual_plan = visual_plans.get(clip.id)
        if visual_plan is None:
            print(f"  Clip {clip.id}: no visual plan, skipping")
            continue

        image_prompt = build_clip_prompt(clip, visual_plan)

        try:
            print(f"  Generating image {clip.id}...")
            gemini.generate_image(
                prompt=image_prompt,
                output_directory=IMAGE_OUTPUT,
                filename=f"image_{clip.id:03d}.png",
            )
            print(f"    Saved: {img_path}")
        except GeminiLimitError as e:
            print(f"  GEMINI LIMIT at clip {clip.id}: {e}")
            break
        except Exception as e:
            print(f"  Clip {clip.id} failed: {e}")

finally:
    gemini.close()

# 5) Regenerate thumbnail if needed
thumb_bg = THUMBNAIL_OUTPUT / "thumbnail_background.png"
if not thumb_bg.exists() or thumb_bg.stat().st_size == 0:
    print("\nGenerating thumbnail background...")
    try:
        gemini = GeminiBrowser(conversation_file=GEMINI_CONVERSATION_FILE)
        gemini.connect()
        gemini.open_gemini()
        thumb_prompt = build_thumbnail_prompt(title=title, thumbnail_plan=thumbnail_plan)
        gemini.generate_image(
            prompt=thumb_prompt,
            output_directory=THUMBNAIL_OUTPUT,
            filename="thumbnail_background.png",
        )
        gemini.close()
    except Exception as e:
        print(f"Thumbnail generation failed: {e}")

# 6) Regenerate full video
print("\nRegenerating full video with static-only editor...")
# Delete old video files
for f in VIDEO_OUTPUT.glob("*.mp4"):
    f.unlink()

editor = VideoEditor(
    image_directory=IMAGE_OUTPUT,
    audio_directory=AUDIO_OUTPUT,
    output_directory=VIDEO_OUTPUT,
)

final_video = editor.create_final_video(
    plans=visual_plans,
    title=title,
    fps=30,
    resolution=(1920, 1080),
    bitrate="8M",
    fade_duration=0.4,
    intro_duration=3.0,
    outro_duration=4.0,
)

print(f"\n[OK] story_001 final video: {final_video}")