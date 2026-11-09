#!/usr/bin/env python3
"""Regenerate videos for story_002 and story_003 using the fixed editor (static only)."""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from editor.video_editor import VideoEditor

STORIES = [
    ("story_002", "output/story_002"),
    ("story_003", "output/story_003"),
]

for story_name, story_dir in STORIES:
    story_path = Path(story_dir)
    if not story_path.exists():
        print(f"SKIP {story_name}: folder not found")
        continue

    visual_plan_file = story_path / "visual_plan.json"
    title_file = story_path / "title.txt"

    if not visual_plan_file.exists():
        print(f"SKIP {story_name}: no visual_plan.json")
        continue
    if not title_file.exists():
        print(f"SKIP {story_name}: no title.txt")
        continue

    with open(visual_plan_file, "r") as f:
        visual_plans = json.load(f)

    title = title_file.read_text(encoding="utf-8").strip()

    print(f"\n{'='*60}")
    print(f"REGENERATING VIDEO FOR {story_name}")
    print(f"Title: {title}")
    print(f"Clips: {len(visual_plans)}")
    print(f"{'='*60}\n")

    editor = VideoEditor(
        image_directory=story_path / "images",
        audio_directory=story_path / "audio",
        output_directory=story_path / "video",
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

    print(f"\n[OK] {story_name} final video: {final_video}")

print("\nAll done.")