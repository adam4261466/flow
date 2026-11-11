"""Tests for cross-story stage-parallel scheduling."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.stage_scheduler import (  # noqa: E402
    StageLayout,
    StageResult,
    StageScheduler,
    StoryJob,
    read_clip_ids,
)
from tests.fixtures import VALID_SCRIPT  # noqa: E402

CLIP_IDS = [1, 2, 3]
class SchedulerTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.layout = StageLayout(
            batch_input=self.root / "input" / "batch",
            output_root=self.root / "output",
            outbox_root=self.root / "outbox",
        )
        self.layout.batch_input.mkdir(parents=True)
        self.completed: list[str] = []
        self.failed: list[str] = []

    def tearDown(self):
        self._tmp.cleanup()

    def job(self, story: str, *, script: bool = True) -> StoryJob:
        path = self.layout.batch_input / f"{story}.txt"
        if script:
            path.write_text(VALID_SCRIPT, encoding="utf-8")
        return StoryJob(prompt=story, story_id=story, script=path)

    def write_images(self, story: str) -> None:
        image_dir = self.layout.story_output(story) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for clip_id in CLIP_IDS:
            self.layout.image(story, clip_id).write_bytes(b"png")
        self.layout.thumbnail(story).parent.mkdir(parents=True, exist_ok=True)
        self.layout.thumbnail(story).write_bytes(b"png")

    def write_voices(self, story: str) -> None:
        audio_dir = self.layout.story_output(story) / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        for clip_id in CLIP_IDS:
            self.layout.audio(story, clip_id).write_bytes(b"wav")

    def write_clips(self, story: str) -> None:
        video_dir = self.layout.story_output(story) / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        for clip_id in CLIP_IDS:
            self.layout.clip(story, clip_id).write_bytes(b"mp4")

    def write_final(self, story: str) -> None:
        self.layout.final(story).parent.mkdir(parents=True, exist_ok=True)
        self.layout.final(story).write_bytes(b"mp4")

    def scheduler(self, jobs, runners=None, **kwargs) -> StageScheduler:
        return StageScheduler(
            jobs,
            layout=self.layout,
            runners=runners or {},
            poll_seconds=0.01,
            retry_seconds=0.01,
            on_complete=lambda job: self.completed.append(job.story_id),
            on_failed=lambda job: self.failed.append(job.story_id),
            log=lambda _message: None,
            **kwargs,
        )


class ProgressProbeTests(SchedulerTestBase):
    def test_read_clip_ids(self):
        job = self.job("story_a")
        self.assertEqual(read_clip_ids(job.script), CLIP_IDS)

    def test_each_stage_claims_the_next_eligible_story(self):
        a = self.job("story_a")
        b = self.job("story_b")
        c = self.job("story_c")

        # Story A has finished all media/clips and only needs the final video.
        self.write_images("story_a")
        self.write_voices("story_a")
        self.write_clips("story_a")
        # Story B only has images, so it is ready for voices and clips later.
        self.write_images("story_b")

        scheduler = self.scheduler([a, b, c])

        # The image worker skips finished A and B and moves straight to C.
        self.assertEqual(scheduler._claim("images").story_id, "story_c")
        # The final worker picks A without waiting for B.
        self.assertEqual(scheduler._claim("final").story_id, "story_a")
        # Clips for B must wait until B's narration exists.
        self.assertIsNone(scheduler._claim("clips"))
        # Voicing B is independent of A's final assembly.
        self.assertEqual(scheduler._claim("voices").story_id, "story_b")

    def test_voice_retry_does_not_block_images_for_the_same_story(self):
        a = self.job("story_a")
        scheduler = self.scheduler([a])

        self.assertIsNotNone(scheduler._claim("voices"))
        # Simulate a failed voice attempt: only the voice stage backs off.
        a.retry_at["voices"] = scheduler.clock() + 600.0
        scheduler._release("voices", a)

        self.assertEqual(scheduler._claim("images").story_id, "story_a")
        self.assertIsNone(scheduler._claim("voices"))

    def test_clips_need_both_images_and_voices(self):
        a = self.job("story_a")
        scheduler = self.scheduler([a])

        self.write_images("story_a")
        self.assertIsNone(scheduler._claim("clips"))

        self.write_voices("story_a")
        self.assertEqual(scheduler._claim("clips").story_id, "story_a")


class FullRunTests(SchedulerTestBase):
    def test_all_stages_finish_all_stories(self):
        jobs = [self.job("story_a"), self.job("story_b")]

        def script(job):
            if not job.script.exists():
                job.script.write_text(VALID_SCRIPT, encoding="utf-8")
            return StageResult.success()

        def images(job):
            self.write_images(job.story_id)
            return StageResult.success()

        def voices(job):
            self.write_voices(job.story_id)
            return StageResult.success()

        def clips(job):
            self.write_clips(job.story_id)
            return StageResult.success()

        def final(job):
            self.write_final(job.story_id)
            return StageResult.success()

        runners = {
            "script": script,
            "images": images,
            "voices": voices,
            "clips": clips,
            "final": final,
        }
        summary = self.scheduler(jobs, runners=runners).run()

        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["failed"], 0)
        self.assertCountEqual(self.completed, ["story_a", "story_b"])
        self.assertEqual(self.failed, [])

    def test_resume_delivers_already_finished_story(self):
        job = self.job("story_a")
        self.write_images("story_a")
        self.write_voices("story_a")
        self.write_clips("story_a")
        self.write_final("story_a")

        summary = self.scheduler([job]).run()

        self.assertEqual(summary["completed"], 1)
        self.assertEqual(self.completed, ["story_a"])

    def test_stage_failure_is_deferred_without_blocking_other_stories(self):
        a = self.job("story_a")
        b = self.job("story_b")

        def images(job):
            if job.story_id == "story_a":
                return StageResult.failed("boom")
            self.write_images(job.story_id)
            return StageResult.success()

        runners = {"images": images}
        scheduler = self.scheduler(
            [a, b],
            runners=runners,
            max_stage_attempts=0,
        ).run()

        # A is deferred, B can still finish its image stage.
        self.assertEqual(scheduler["failed"], 1)
        self.assertIn("story_a", self.failed)
        self.assertTrue(self.layout.thumbnail("story_b").exists())


    def test_fatal_stage_result_stops_the_factory(self):
        a = self.job("story_a")
        b = self.job("story_b")
        calls: list[str] = []

        def images(job):
            calls.append(job.story_id)
            return StageResult.failed("Gemini still limited", fatal=True)

        summary = self.scheduler([a, b], runners={"images": images}).run()

        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.failed, [calls[0]])


class FactoryStageLockTests(SchedulerTestBase):
    def make_runner(self):
        from agents.limits import LimitLedger
        from batch_generator import FactoryStageRunner

        return FactoryStageRunner(
            kids=True,
            no_agents=True,
            max_wait_minutes=1.0,
            no_wait=True,
            ledger=LimitLedger(self.root / "factory_state.json"),
            stop_event=threading.Event(),
        )

    def test_two_stages_cannot_drive_the_same_story_at_once(self):
        job = self.job("story_a")
        runner = self.make_runner()
        lock = runner._story_lock(job.story_id)
        lock.acquire()
        try:
            result = runner("script", job)
        finally:
            lock.release()

        self.assertTrue(result.blocked)
        self.assertFalse(result.fatal)

    def test_a_busy_story_does_not_block_another_story(self):
        busy = self.job("story_a")
        other = self.job("story_b")
        runner = self.make_runner()
        lock = runner._story_lock(busy.story_id)
        lock.acquire()
        try:
            result = runner("script", other)
        finally:
            lock.release()

        self.assertTrue(result.ok)

    def test_script_stage_labels_the_claude_banner(self):
        import contextlib
        import io
        from unittest import mock

        import batch_generator

        job = self.job("story_new", script=False)
        runner = self.make_runner()
        runner.script_backlog = 4
        seen = {}

        def fake(prompt=None, *, label="", **kwargs):
            seen["label"] = label
            return VALID_SCRIPT

        buffer = io.StringIO()
        with mock.patch.object(
            batch_generator, "generate_story_with_claude", fake
        ), contextlib.redirect_stdout(buffer):
            result = runner("script", job)

        self.assertTrue(result.ok)
        self.assertEqual(seen["label"], "story_new")
        self.assertIn("[script] 1/4 story_new", buffer.getvalue())

    def test_stop_event_terminates_the_stage_process(self):
        runner = self.make_runner()
        runner.stop_event.set()

        started = time.monotonic()
        code = runner._run_stage_command(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        elapsed = time.monotonic() - started

        self.assertIsNone(code)
        self.assertLess(elapsed, 20.0)


if __name__ == "__main__":
    unittest.main()
