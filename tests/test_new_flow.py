"""Tests for the Claude/numbering/YouTube planning additions.

Run with::

    py -3.13 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date, datetime, time
from pathlib import Path

# Make the project root importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.claude_browser import (  # noqa: E402
    STORY_PROMPT,
    clean_claude_script,
    validate_claude_script,
)
from utils.story_scripts import (  # noqa: E402
    next_number,
    save_generated_script,
    story_name,
)
from youtube_uploader import (  # noqa: E402
    UploadItem,
    YouTubeStudioUploader,
    allocate_publish_slot,
    daily_publish_times,
    discover_all_videos,
    discover_finished_videos,
    discover_outbox_videos,
    load_manifest,
    plan_uploads,
    save_manifest,
    upload_all_finished_videos,
)


def sample_script() -> str:
    clip = """CLIP

Characters:
Mara: a 30-year-old woman with short black hair and a grey coat.

Moment:
Mara stands in an empty train station as a single light flickers above her.

Vibe:
mysterious and tense

Style:
dark photorealistic thriller

Script:
The last train had gone hours ago, but the station lights were still on.
"""
    return "GLOBAL\n\nCharacters:\nMara: a 30-year-old woman with short black hair and a grey coat.\n\nStyle:\ndark photorealistic thriller\n\n$$\n\n" + "\n$$\n\n".join([clip, clip, clip])


class ClaudeCleanupTests(unittest.TestCase):
    def test_prompt_is_verbatim_start_and_end(self):
        self.assertTrue(
            STORY_PROMPT.startswith(
                "You are a professional cinematic storyteller writing scripts"
            )
        )
        self.assertTrue(
            STORY_PROMPT.rstrip().endswith(
                "Now generate the complete cinematic story script."
            )
        )

    def test_kids_prompt_is_safe_and_keeps_format(self):
        from ai.claude_browser import KIDS_STORY_PROMPT

        self.assertIn("children aged 3 to 8", KIDS_STORY_PROMPT)
        self.assertIn("CHILD-SAFETY RULES", KIDS_STORY_PROMPT)
        self.assertIn("Now generate the complete children's story script.", KIDS_STORY_PROMPT)
        self.assertIn("GLOBAL", KIDS_STORY_PROMPT)
        self.assertIn("CLIP", KIDS_STORY_PROMPT)
        self.assertIn("$$", KIDS_STORY_PROMPT)

        forbidden = [
            "horror",
            "gunshot",
            "psychological thriller",
            "dystopian",
            "photorealistic thriller",
            "a door slams",
            "glass breaks",
        ]
        for word in forbidden:
            self.assertNotIn(word, KIDS_STORY_PROMPT)

    def test_removes_preamble_and_code_fence(self):
        raw = (
            "Here is the script you asked for:\n\n```\n"
            + sample_script()
            + "\n```\n"
        )
        cleaned = clean_claude_script(raw)
        self.assertTrue(cleaned.startswith("GLOBAL"))
        self.assertNotIn("Here is the script", cleaned)
        self.assertNotIn("```", cleaned)

    def test_normalizes_markdown_bold(self):
        raw = sample_script().replace("GLOBAL", "**GLOBAL**")
        raw = raw.replace("CLIP", "**CLIP**")
        raw = raw.replace("Characters:", "**Characters:**")
        cleaned = clean_claude_script(raw)
        self.assertTrue(cleaned.startswith("GLOBAL"))
        self.assertIn("\nCLIP\n", cleaned)
        self.assertIn("Characters:", cleaned)

    def test_validate_returns_clips(self):
        clips = validate_claude_script(sample_script())
        self.assertEqual(len(clips), 3)
        self.assertEqual(clips[0].id, 1)

    def test_missing_global_rejected(self):
        with self.assertRaises(Exception):
            clean_claude_script("CLIP\n\nCharacters:\nNobody\n")


class NumberingTests(unittest.TestCase):
    def test_next_number_uses_scripts_and_stories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "script1.txt").write_text("one", encoding="utf-8")
            (input_dir / "script3.txt").write_text("three", encoding="utf-8")
            (output_dir / "story_002").mkdir()
            (output_dir / "story_007").mkdir()

            self.assertEqual(next_number(input_dir, output_dir), 8)
            self.assertEqual(story_name(8), "story_008")

    def test_save_generated_script_uses_next_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            for number in (1, 2, 3, 4):
                (input_dir / f"script{number}.txt").write_text("x", encoding="utf-8")
                (output_dir / f"story_{number:03d}").mkdir(parents=True, exist_ok=True)

            path, story = save_generated_script(sample_script(), input_dir, output_dir)
            self.assertEqual(path.name, "script5.txt")
            self.assertEqual(story, "story_005")
            self.assertTrue(path.is_file())
            self.assertIn("GLOBAL", path.read_text(encoding="utf-8"))

    def test_save_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "script1.txt").write_text("keep me", encoding="utf-8")

            path, story = save_generated_script(sample_script(), input_dir, output_dir)
            self.assertEqual(path.name, "script2.txt")
            self.assertEqual(story, "story_002")
            self.assertEqual(
                (input_dir / "script1.txt").read_text(encoding="utf-8"),
                "keep me",
            )


class YouTubePlanningTests(unittest.TestCase):
    def test_allocate_skips_used_dates_and_past_time(self):
        used = {date(2026, 11, 6)}
        slot = allocate_publish_slot(
            used,
            start_date=date(2026, 11, 6),
            publish_time=time(18, 0),
            now=datetime(2026, 11, 5, 12, 0),
        )
        self.assertEqual(slot, datetime(2026, 11, 7, 18, 0))

    def test_allocate_skips_today_when_time_passed(self):
        slot = allocate_publish_slot(
            set(),
            start_date=date(2026, 11, 5),
            publish_time=time(18, 0),
            now=datetime(2026, 11, 5, 20, 0),
        )
        self.assertEqual(slot, datetime(2026, 11, 6, 18, 0))

    def _item(self, story: str) -> UploadItem:
        return UploadItem(
            story=story,
            video=Path("video.mp4"),
            title=story,
            description="description",
        )

    def test_plan_assigns_one_video_per_day(self):
        items = [self._item("story_001"), self._item("story_002"), self._item("story_003")]
        plan = plan_uploads(
            items,
            manifest={},
            start_date="2026-11-06",
            publish_time="18:00",
            now=datetime(2026, 11, 5, 12, 0),
        )
        self.assertEqual(
            [item.scheduled_at for item in plan],
            [
                datetime(2026, 11, 6, 18, 0),
                datetime(2026, 11, 7, 18, 0),
                datetime(2026, 11, 8, 18, 0),
            ],
        )

    def test_plan_skips_already_scheduled(self):
        manifest = {
            "story_001": {
                "status": "scheduled",
                "scheduled_at": "2026-11-06 18:00",
            }
        }
        items = [self._item("story_001"), self._item("story_002")]
        plan = plan_uploads(
            items,
            manifest=manifest,
            start_date="2026-11-06",
            publish_time="18:00",
            now=datetime(2026, 11, 5, 12, 0),
        )
        self.assertEqual([item.story for item in plan], ["story_002"])
        self.assertEqual(plan[0].scheduled_at, datetime(2026, 11, 7, 18, 0))

    def test_discover_reads_title_and_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            story = output / "story_002"
            (story / "video").mkdir(parents=True)
            (story / "video" / "final_video.mp4").write_bytes(b"video")
            (story / "title.txt").write_text("My Film", encoding="utf-8")
            (story / "description.txt").write_text("A description", encoding="utf-8")

            items = discover_finished_videos(output)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].story, "story_002")
            self.assertEqual(items[0].title, "My Film")
            self.assertEqual(items[0].description, "A description")

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            data = {"story_001": {"status": "scheduled"}}
            save_manifest(data, path)
            self.assertEqual(load_manifest(path), data)


class YouTubeScheduleFieldTests(unittest.TestCase):
    """The new Studio Visibility step renders Public + a Schedule box."""

    def test_date_matches_studio_display(self):
        value = datetime(2026, 11, 10, 18, 0)
        self.assertTrue(YouTubeStudioUploader._date_matches("Nov 10, 2026", value))
        self.assertTrue(
            YouTubeStudioUploader._date_matches("November 10, 2026", value)
        )
        self.assertFalse(
            YouTubeStudioUploader._date_matches("Nov 1, 2026", value)
        )
        self.assertFalse(
            YouTubeStudioUploader._date_matches("Nov 10, 2025", value)
        )

    def test_time_matches_studio_display(self):
        value = datetime(2026, 11, 10, 18, 0)
        self.assertTrue(YouTubeStudioUploader._time_matches("6:00 PM", value))
        self.assertTrue(
            YouTubeStudioUploader._time_matches("6:00\u202fPM", value)
        )
        self.assertTrue(YouTubeStudioUploader._time_matches("18:00", value))
        self.assertFalse(YouTubeStudioUploader._time_matches("12:00 AM", value))


class YouTubeCadenceTests(unittest.TestCase):
    """Three videos per day, with an optional 1 -> 3 ramp."""

    def _item(self, story: str) -> UploadItem:
        return UploadItem(
            story=story,
            video=Path("video.mp4"),
            title=story,
            description="description",
        )

    def test_daily_publish_times(self):
        self.assertEqual(daily_publish_times(1, "18:00"), [time(18, 0)])
        self.assertEqual(
            daily_publish_times(2, "18:00"),
            [time(12, 0), time(18, 0)],
        )
        self.assertEqual(
            daily_publish_times(3, "18:00"),
            [time(10, 0), time(14, 0), time(18, 0)],
        )

    def test_plan_assigns_three_videos_per_day(self):
        items = [self._item(f"story_{index:03d}") for index in range(1, 6)]
        plan = plan_uploads(
            items,
            manifest={},
            start_date="2026-11-06",
            publish_time="18:00",
            videos_per_day=3,
            now=datetime(2026, 11, 5, 12, 0),
        )
        self.assertEqual(
            [item.scheduled_at for item in plan],
            [
                datetime(2026, 11, 6, 10, 0),
                datetime(2026, 11, 6, 14, 0),
                datetime(2026, 11, 6, 18, 0),
                datetime(2026, 11, 7, 10, 0),
                datetime(2026, 11, 7, 14, 0),
            ],
        )

    def test_plan_ramps_from_one_to_three_per_day(self):
        items = [self._item(f"story_{index:03d}") for index in range(1, 7)]
        plan = plan_uploads(
            items,
            manifest={},
            start_date="2026-11-06",
            publish_time="18:00",
            videos_per_day=3,
            ramp_days=2,
            start_videos_per_day=1,
            now=datetime(2026, 11, 5, 12, 0),
        )
        self.assertEqual(
            [item.scheduled_at for item in plan],
            [
                datetime(2026, 11, 6, 18, 0),
                datetime(2026, 11, 7, 12, 0),
                datetime(2026, 11, 7, 18, 0),
                datetime(2026, 11, 8, 10, 0),
                datetime(2026, 11, 8, 14, 0),
                datetime(2026, 11, 8, 18, 0),
            ],
        )

    def test_plan_reuses_only_free_slots(self):
        manifest = {
            "story_001": {
                "status": "scheduled",
                "scheduled_at": "2026-11-06 10:00",
            }
        }
        plan = plan_uploads(
            [self._item("story_002")],
            manifest=manifest,
            start_date="2026-11-06",
            publish_time="18:00",
            videos_per_day=3,
            now=datetime(2026, 11, 5, 12, 0),
        )
        self.assertEqual(plan[0].scheduled_at, datetime(2026, 11, 6, 14, 0))

    def test_legacy_date_blocks_the_whole_day_at_one_per_day(self):
        manifest = {
            "story_001": {
                "status": "scheduled",
                "scheduled_date": "2026-11-06",
            }
        }
        plan = plan_uploads(
            [self._item("story_002")],
            manifest=manifest,
            start_date="2026-11-06",
            publish_time="18:00",
            now=datetime(2026, 11, 5, 12, 0),
        )
        self.assertEqual(plan[0].scheduled_at, datetime(2026, 11, 7, 18, 0))


class OutboxDiscoveryTests(unittest.TestCase):
    def test_outbox_reads_metadata_and_falls_back_to_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outbox = root / "outbox"
            output = root / "output"
            story = outbox / "story_1700000000"
            story.mkdir(parents=True)
            (story / "video.mp4").write_bytes(b"video")
            (story / "thumbnail.png").write_bytes(b"image")

            output_story = output / "story_1700000000"
            output_story.mkdir(parents=True)
            (output_story / "title.txt").write_text(
                "Fallback title",
                encoding="utf-8",
            )
            (output_story / "description.txt").write_text(
                "Fallback description",
                encoding="utf-8",
            )

            items = discover_outbox_videos(outbox, output)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].story, "story_1700000000")
            self.assertEqual(items[0].title, "Fallback title")
            self.assertEqual(items[0].description, "Fallback description")
            self.assertEqual(items[0].source, "outbox")
            self.assertEqual(items[0].thumbnail, story / "thumbnail.png")

    def test_discover_all_prefers_outbox_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outbox = root / "outbox"
            output = root / "output"

            outbox_story = outbox / "story_001"
            outbox_story.mkdir(parents=True)
            (outbox_story / "video.mp4").write_bytes(b"outbox video")
            (outbox_story / "title.txt").write_text(
                "Outbox title",
                encoding="utf-8",
            )

            output_story = output / "story_001"
            (output_story / "video").mkdir(parents=True)
            (output_story / "video" / "final_video.mp4").write_bytes(b"output")
            (output_story / "title.txt").write_text(
                "Output title",
                encoding="utf-8",
            )

            items = discover_all_videos(output, outbox)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].title, "Outbox title")
            self.assertEqual(items[0].source, "outbox")

    def test_discover_all_can_ignore_outbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outbox = root / "outbox"
            output = root / "output"

            outbox_story = outbox / "story_001"
            outbox_story.mkdir(parents=True)
            (outbox_story / "video.mp4").write_bytes(b"outbox video")

            output_story = output / "story_002"
            (output_story / "video").mkdir(parents=True)
            (output_story / "video" / "final_video.mp4").write_bytes(b"output")

            items = discover_all_videos(output, outbox, include_outbox=False)
            self.assertEqual([item.story for item in items], ["story_002"])

    def test_upload_all_dry_run_reads_outbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outbox = root / "outbox"
            output = root / "output"
            output.mkdir()

            story = outbox / "story_001"
            story.mkdir(parents=True)
            (story / "video.mp4").write_bytes(b"video")
            (story / "title.txt").write_text("Outbox title", encoding="utf-8")

            summary = upload_all_finished_videos(
                output_root=output,
                outbox_root=outbox,
                manifest_path=root / "manifest.json",
                start_date="2026-11-06",
                publish_time="18:00",
                videos_per_day=3,
                dry_run=True,
            )

            self.assertEqual(summary["found"], 1)
            self.assertEqual(summary["videos_per_day"], 3)
            self.assertEqual(summary["planned"][0]["story"], "story_001")
            self.assertEqual(summary["planned"][0]["source"], "outbox")
            self.assertEqual(summary["uploaded"], [])


if __name__ == "__main__":
    unittest.main()
