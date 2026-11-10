"""Tests for the local dashboard's command builders.

The dashboard never imports the pipeline; it only translates saved settings
into the exact CLI invocation.  A wrong flag here silently disables the
stage-parallel factory, so keep these assertions tight.

Run with::

    py -3.13 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dashboard  # noqa: E402


class FactoryCommandTests(unittest.TestCase):
    def build(self, **overrides) -> list[str]:
        settings = {**dashboard.DEFAULT_SETTINGS, **overrides}
        return dashboard.build_factory(settings)[1]

    def test_stage_parallel_is_the_default(self):
        command = self.build()
        self.assertIn("--stage-parallel", command)
        self.assertNotIn("--no-stage-parallel", command)
        self.assertEqual(command[command.index("--clip-workers") + 1], "1")
        self.assertEqual(command[command.index("--stage-retries") + 1], "3")

    def test_stage_parallel_can_be_disabled(self):
        command = self.build(stage_parallel=False)
        self.assertIn("--no-stage-parallel", command)
        self.assertNotIn("--stage-parallel", command)

    def test_clip_workers_and_retries_are_bounded_numbers(self):
        command = self.build(clip_workers=99, stage_retries=99)
        self.assertEqual(command[command.index("--clip-workers") + 1], "8")
        self.assertEqual(command[command.index("--stage-retries") + 1], "20")

    def test_story_command_supports_each_stage_only_flag(self):
        for key, flag in (
            ("images_only", "--images-only"),
            ("clips_only", "--clips-only"),
            ("final_only", "--final-only"),
            ("voices_only", "--voices-only"),
        ):
            command = dashboard.build_story({"story": "story_1", key: True})[1]
            self.assertIn(flag, command)

    def test_story_command_rejects_unsafe_ids(self):
        with self.assertRaises(ValueError):
            dashboard.build_story({"story": "../escape"})


if __name__ == "__main__":
    unittest.main()
