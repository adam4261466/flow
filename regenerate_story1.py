#!/usr/bin/env python3
"""Regenerate video for story_001 using the fixed editor (static only)."""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from editor.video_editor import VideoEditor

story_dir = Path("output/story_001")
if not story_dir.exists():
    print("story_001 folder not found")
    sys.exit(1)

visual_plan_file = story_dir / "visual_plan.json"
title_file = story_dir / "title.txt"

if not visual_plan_file.exists():
    print("no visual_plan.json")
    sys.exit(1)
if not title_file.exists():
    print("no title.txt")
    sys.exit(1)

with open(visual_plan_file, "r") as f:
    visual_plans = json.load(f)

title = title_file.read_text(encoding="utf-8").strip()

print(f"REGENERATING VIDEO FOR story_001")
print(f"Title: {title}")
print(f"Clips: {len(visual_plans)}")

editor = VideoEditor(
    image_directory=story_dir / "images",
    audio_directory=story_dir / "audio",
    output_directory=story_dir / "video",
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