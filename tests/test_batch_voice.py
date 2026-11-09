"""Tests for the continuous voice-first batch pass.

Run with::

    py -3.13 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import batch_generator  # noqa: E402
from agents.batch_state import BatchStateManager  # noqa: E402
from agents.limits import LimitLedger  # noqa: E402
from ai.claude_browser import ClaudeLimitError  # noqa: E402
from batch_generator import ensure_script  # noqa: E402


class EnsureScriptTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.batch_dir = self.tmp / "batch"
        self.batch_dir.mkdir()
        self._patch = mock.patch.object(
            batch_generator, "BATCH_INPUT_DIR", self.batch_dir
        )
        self._patch.start()

        self.state = BatchStateManager(str(self.tmp / "batch_state.json"))
        self.state.add_prompts(["a prompt"])
        self.prompt = self.state.get_next()

        self.ledger = LimitLedger(self.tmp / "factory_state.json")

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _ensure(self):
        return ensure_script(
            self.state,
            self.prompt,
            kids=True,
            ledger=self.ledger,
            announced={},
            max_wait_minutes=240.0,
        )

    def test_existing_script_is_reused_without_calling_claude(self):
        story_id = "story_existing"
        script_file = self.batch_dir / f"{story_id}.txt"
        script_file.write_text("GLOBAL\nScript:\nHi\n$$\n", encoding="utf-8")
        self.state.set_story(self.prompt, story_id, script_file)

        with mock.patch.object(
            batch_generator, "generate_story_with_claude"
        ) as generate:
            got_id, got_script = self._ensure()

        generate.assert_not_called()
        self.assertEqual(got_id, story_id)
        self.assertEqual(got_script, script_file)

    def test_new_script_is_written(self):
        with mock.patch.object(
            batch_generator,
            "generate_story_with_claude",
            return_value="GLOBAL\nScript:\nHello\n$$\n",
        ) as generate:
            story_id, script_file = self._ensure()

        self.assertTrue(script_file.exists())
        self.assertIn("Hello", script_file.read_text(encoding="utf-8"))
        self.assertEqual(self.state.get_story(self.prompt)["story_id"], story_id)
        # The kids prompt is passed through.
        self.assertIn("children's story", generate.call_args.kwargs["prompt"])

    def test_claude_limit_waits_then_retries(self):
        with mock.patch.object(
            batch_generator, "wait_for_resource", return_value=True
        ), mock.patch.object(
            batch_generator,
            "generate_story_with_claude",
            side_effect=[ClaudeLimitError("resets in 2 hours"), "GLOBAL\n$$"],
        ) as generate, mock.patch.object(
            batch_generator.time, "sleep"
        ):
            story_id, script_file = self._ensure()

        self.assertEqual(generate.call_count, 2)
        self.assertIsNotNone(script_file)
        self.assertTrue(
            self.ledger.remaining("claude") > 0.0,
            "the Claude cooldown must be persisted for the next story",
        )

    def test_permanent_failure_returns_no_script(self):
        with mock.patch.object(
            batch_generator,
            "generate_story_with_claude",
            side_effect=RuntimeError("boom"),
        ):
            story_id, script_file = self._ensure()

        self.assertIsNone(script_file)
        # The story folder is still remembered so a restart can resume it.
        self.assertEqual(self.state.get_story(self.prompt)["story_id"], story_id)


class VoiceFirstPassTests(unittest.TestCase):
    def test_empty_queue_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = BatchStateManager(str(Path(tmp) / "batch_state.json"))
            ledger = LimitLedger(Path(tmp) / "factory_state.json")
            with mock.patch.object(batch_generator.subprocess, "run") as run:
                batch_generator.voice_first_pass(
                    state,
                    kids=True,
                    no_agents=False,
                    ledger=ledger,
                    announced={},
                    max_wait_minutes=240.0,
                )
            run.assert_not_called()

    def test_tts_failure_stops_the_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            batch_dir = tmp / "batch"
            batch_dir.mkdir()
            state = BatchStateManager(str(tmp / "batch_state.json"))
            state.add_prompts(["one", "two"])
            ledger = LimitLedger(tmp / "factory_state.json")

            # Pretend every script already exists so no Claude call is needed.
            for prompt in state.state["pending"]:
                state.set_story(prompt, "story_1", batch_dir / "story_1.txt")
            (batch_dir / "story_1.txt").write_text("GLOBAL\n$$", encoding="utf-8")

            failing = mock.Mock(returncode=2)
            with mock.patch.object(
                batch_generator, "BATCH_INPUT_DIR", batch_dir
            ), mock.patch.object(
                batch_generator.subprocess, "run", return_value=failing
            ) as run:
                batch_generator.voice_first_pass(
                    state,
                    kids=True,
                    no_agents=False,
                    ledger=ledger,
                    announced={},
                    max_wait_minutes=240.0,
                )

            # It stops after the first TTS failure instead of hammering it for
            # every remaining story.
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
