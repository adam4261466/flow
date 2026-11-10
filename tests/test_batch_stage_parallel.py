"""Integration test for the stage-parallel batch orchestration.

The real stages shell out to Chrome/TTS/FFmpeg.  Here a fake runner writes the
same artifacts those stages produce, so the scheduler, dependency graph,
outbox delivery and queue bookkeeping can all be exercised in a second.

Run with::

    py -3.13 -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
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
from agents.stage_scheduler import StageResult, read_clip_ids  # noqa: E402
from tests.fixtures import VALID_SCRIPT  # noqa: E402


class FakeStageRunner:
    """Write the same files a real stage would, in dependency order."""

    events: list[tuple[str, str]] = []

    def __init__(self, **_kwargs):
        pass

    def __call__(self, stage: str, job) -> StageResult:
        self.events.append((stage, job.story_id))
        output = batch_generator.OUTPUT_DIR / job.story_id

        if stage == "script":
            job.script.parent.mkdir(parents=True, exist_ok=True)
            job.script.write_text(VALID_SCRIPT, encoding="utf-8")
            return StageResult.success()

        clip_ids = read_clip_ids(job.script)
        if stage == "images":
            (output / "images").mkdir(parents=True, exist_ok=True)
            for clip_id in clip_ids:
                (output / "images" / f"image_{clip_id:03d}.png").write_bytes(b"png")
            (output / "thumbnail").mkdir(parents=True, exist_ok=True)
            (output / "thumbnail" / "thumbnail.png").write_bytes(b"png")
        elif stage == "voices":
            (output / "audio").mkdir(parents=True, exist_ok=True)
            for clip_id in clip_ids:
                (output / "audio" / f"audio_{clip_id:03d}.wav").write_bytes(b"wav")
        elif stage == "clips":
            (output / "video").mkdir(parents=True, exist_ok=True)
            for clip_id in clip_ids:
                (output / "video" / f"clip_{clip_id:03d}.mp4").write_bytes(b"mp4")
        elif stage == "final":
            (output / "video").mkdir(parents=True, exist_ok=True)
            (output / "video" / "final_video.mp4").write_bytes(b"mp4")
            (output / "title.txt").write_text("Fake Story", encoding="utf-8")
            (output / "description.txt").write_text("Fake description", encoding="utf-8")
        else:
            return StageResult.failed(f"unknown stage {stage}")
        return StageResult.success()


def make_args(**overrides) -> argparse.Namespace:
    values = dict(
        no_agents=True,
        max_wait_minutes=0.0,
        no_wait=True,
        clip_workers=1,
        stage_retries=3,
        stage_poll_seconds=0.01,
        stage_retry_seconds=0.01,
        upload_each=False,
        upload=False,
        dry_run_upload=True,
        max_story_attempts=3,
        per_day=3,
        ramp_days=0,
        start_per_day=None,
        start_date=None,
        publish_time="18:00",
        voices_first=False,
        stage_parallel=True,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class StageParallelFactoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.batch_input = self.root / "input" / "batch"
        self.output = self.root / "output"
        self.outbox = self.root / "outbox"
        self.batch_input.mkdir(parents=True)
        self.output.mkdir()
        self.outbox.mkdir()

        self._patches = [
            mock.patch.object(batch_generator, "BATCH_INPUT_DIR", self.batch_input),
            mock.patch.object(batch_generator, "OUTPUT_DIR", self.output),
            mock.patch.object(batch_generator, "OUTBOX_DIR", self.outbox),
            mock.patch.object(batch_generator, "FactoryStageRunner", FakeStageRunner),
            mock.patch.object(batch_generator, "update_status", lambda _status: None),
        ]
        for patch in self._patches:
            patch.start()

        self.state = BatchStateManager(str(self.root / "batch_state.json"))
        self.state.add_prompts(["Story one", "Story two"])
        FakeStageRunner.events = []

    def tearDown(self):
        for patch in self._patches:
            patch.stop()
        self._tmp.cleanup()

    def test_two_stories_move_through_every_stage(self):
        summary = batch_generator.run_stage_parallel(
            self.state,
            args=make_args(),
            kids=True,
            ledger=LimitLedger(self.root / "factory_state.json"),
        )

        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["pending"], 0)
        self.assertEqual(len(self.state.state["pending"]), 0)
        self.assertEqual(len(self.state.state["completed"]), 2)

        # Both stories were delivered to the outbox with their metadata.
        outbox_stories = sorted(p.name for p in self.outbox.iterdir())
        self.assertEqual(len(outbox_stories), 2)
        for name in outbox_stories:
            self.assertTrue((self.outbox / name / "video.mp4").exists())
            self.assertTrue((self.outbox / name / "thumbnail.png").exists())
            self.assertTrue((self.outbox / name / "title.txt").exists())

    def test_stage_dependencies_hold_for_each_story(self):
        batch_generator.run_stage_parallel(
            self.state,
            args=make_args(),
            kids=True,
            ledger=LimitLedger(self.root / "factory_state.json"),
        )

        order = {stage: [] for stage in ("script", "images", "voices", "clips", "final")}
        for index, (stage, story) in enumerate(FakeStageRunner.events):
            order[stage].append((index, story))

        starts = {story: {} for _, story in FakeStageRunner.events}
        for index, (stage, story) in enumerate(FakeStageRunner.events):
            starts[story].setdefault(stage, index)

        for story, positions in starts.items():
            self.assertIn("script", positions)
            self.assertLess(positions["script"], positions["images"])
            self.assertLess(positions["script"], positions["voices"])
            self.assertGreater(positions["clips"], positions["images"])
            self.assertGreater(positions["clips"], positions["voices"])
            self.assertGreater(positions["final"], positions["clips"])

    def test_failed_stage_is_requeued_without_stopping_other_story(self):
        original = FakeStageRunner.__call__

        def flaky(self, stage, job):
            if stage == "images" and job.prompt == "Story one":
                return StageResult.failed("boom")
            return original(self, stage, job)

        with mock.patch.object(FakeStageRunner, "__call__", flaky):
            summary = batch_generator.run_stage_parallel(
                self.state,
                args=make_args(stage_retries=0),
                kids=True,
                ledger=LimitLedger(self.root / "factory_state.json"),
            )

        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["failed"], 1)
        # The failed prompt stays queued for the next factory run.
        self.assertIn("Story one", self.state.state["pending"])


if __name__ == "__main__":
    unittest.main()
