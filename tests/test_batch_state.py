"""Tests for the resumable batch queue.

Run with::

    py -3.13 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.batch_state import BatchStateManager  # noqa: E402


class BatchStateResumeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "batch_state.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_old_file_format_still_loads(self):
        self.path.write_text(
            json.dumps({"completed": ["old"], "pending": ["next"]}),
            encoding="utf-8",
        )
        state = BatchStateManager(str(self.path))
        self.assertEqual(state.get_next(), "next")
        self.assertEqual(state.state["stories"], {})
        self.assertEqual(state.state["settings"], {})

    def test_story_folder_survives_restart(self):
        state = BatchStateManager(str(self.path))
        state.add_prompts(["a story"])
        prompt = state.get_next()
        state.set_story(prompt, "story_123", "input/batch/story_123.txt")

        reopened = BatchStateManager(str(self.path))
        info = reopened.get_story(prompt)
        self.assertEqual(info["story_id"], "story_123")
        self.assertEqual(info["script"], "input/batch/story_123.txt")
        self.assertEqual(reopened.get_next(), prompt)

    def test_attempts_and_requeue(self):
        state = BatchStateManager(str(self.path))
        state.add_prompts(["first", "second"])

        self.assertEqual(state.bump_attempt("first"), 1)
        self.assertEqual(state.bump_attempt("first"), 2)
        state.requeue("first")

        self.assertEqual(state.get_next(), "second")
        self.assertEqual(state.state["pending"], ["second", "first"])

    def test_mark_complete_keeps_story_history(self):
        state = BatchStateManager(str(self.path))
        state.add_prompts(["done"])
        state.set_story("done", "story_999")
        state.mark_complete("done")

        reopened = BatchStateManager(str(self.path))
        self.assertEqual(reopened.state["pending"], [])
        self.assertIn("done", reopened.state["completed"])
        self.assertEqual(reopened.get_story("done")["story_id"], "story_999")

    def test_settings_persist(self):
        state = BatchStateManager(str(self.path))
        state.set_setting("kids", True)
        reopened = BatchStateManager(str(self.path))
        self.assertTrue(reopened.get_setting("kids"))
        self.assertEqual(reopened.get_setting("missing", "fallback"), "fallback")


if __name__ == "__main__":
    unittest.main()
