"""Tests for the parallel stage workers and the usage-limit ledger."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.limits import LimitLedger  # noqa: E402
from agents.orchestrator import (  # noqa: E402
    ParallelPipeline,
    StagePaths,
)


@dataclass
class FakeClip:
    id: int


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class LimitLedgerTests(unittest.TestCase):
    def test_block_and_expire(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            ledger = LimitLedger(
                Path(tmp) / "factory_state.json",
                clock=clock,
            )

            self.assertFalse(ledger.is_blocked("gemini"))
            ledger.block("gemini", 10.0, "limit reached")

            self.assertTrue(ledger.is_blocked("gemini"))
            self.assertAlmostEqual(
                ledger.remaining("gemini"), 600.0, places=3
            )

            clock.advance(599.0)
            self.assertTrue(ledger.is_blocked("gemini"))

            clock.advance(1.5)
            self.assertFalse(ledger.is_blocked("gemini"))
            self.assertEqual(ledger.remaining("gemini"), 0.0)

    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "factory_state.json"
            clock = FakeClock()
            ledger = LimitLedger(path, clock=clock)
            ledger.block("tts", 5.0, "server busy")

            reloaded = LimitLedger(path, clock=clock)
            self.assertTrue(reloaded.is_blocked("tts"))
            self.assertEqual(
                reloaded.info("tts")["message"], "server busy"
            )
            self.assertTrue(path.exists())
            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("resources", parsed)

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            ledger = LimitLedger(Path(tmp) / "state.json", clock=clock)
            ledger.block("gemini", 5.0)
            ledger.clear("gemini")
            self.assertFalse(ledger.is_blocked("gemini"))


class ParallelPipelineTests(unittest.TestCase):
    def _paths(self, tmp: str) -> StagePaths:
        base = Path(tmp)
        paths = StagePaths(
            images=base / "images",
            audio=base / "audio",
            video=base / "video",
            thumbnail=base / "thumbnail",
        )
        for folder in (
            paths.images,
            paths.audio,
            paths.video,
            paths.thumbnail,
        ):
            folder.mkdir(parents=True, exist_ok=True)
        return paths

    def _ledger(self, tmp: str) -> LimitLedger:
        return LimitLedger(Path(tmp) / "factory_state.json")

    def test_clip_renders_before_all_voices_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(tmp)
            clips = [FakeClip(1), FakeClip(2)]
            clip_one_rendered = threading.Event()
            voice_two_started = threading.Event()
            order: list[str] = []
            order_lock = threading.Lock()

            def record(item: str) -> None:
                with order_lock:
                    order.append(item)

            def generate_images():
                for clip in clips:
                    paths.image(clip.id).write_bytes(b"x" * 2048)
                paths.thumbnail_file.write_bytes(b"x" * 2048)
                record("images")
                return False, None

            def generate_voices():
                paths.audio_file(1).write_bytes(b"x" * 8192)
                # Clip 1 must render while voice 2 is still being produced.
                if not clip_one_rendered.wait(timeout=5):
                    raise AssertionError(
                        "clip 1 did not render before voices finished"
                    )
                voice_two_started.set()
                paths.audio_file(2).write_bytes(b"x" * 8192)
                record("voices")

            def render_clip(clip_id, plan):
                record(f"clip{clip_id}")
                paths.clip(clip_id).write_bytes(b"x" * 4096)
                if clip_id == 1:
                    clip_one_rendered.set()

            def assemble_final():
                record("final")
                paths.final.write_bytes(b"x" * 4096)
                return paths.final

            checkpoints: list[str] = []

            pipeline = ParallelPipeline(
                clips=clips,
                paths=paths,
                plans={1: {}, 2: {}},
                generate_images=generate_images,
                generate_voices=generate_voices,
                render_clip=render_clip,
                assemble_final=assemble_final,
                ledger=self._ledger(tmp),
                checkpoint=lambda stage, blocked, message: checkpoints.append(
                    stage
                ),
                poll_seconds=0.02,
                settle_seconds=0.0,
            )

            result = pipeline.run()

            self.assertTrue(result.complete)
            self.assertEqual(result.missing_images, [])
            self.assertEqual(result.missing_voices, [])
            self.assertEqual(result.missing_clips, [])
            self.assertTrue(voice_two_started.is_set())
            self.assertLess(order.index("clip1"), order.index("voices"))
            self.assertEqual(order[-1], "final")
            self.assertIn("complete", checkpoints)

    def test_voice_failure_reports_incomplete_without_hanging(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(tmp)
            clips = [FakeClip(1), FakeClip(2)]

            def generate_images():
                for clip in clips:
                    paths.image(clip.id).write_bytes(b"x" * 2048)
                paths.thumbnail_file.write_bytes(b"x" * 2048)
                return False, None

            def generate_voices():
                raise RuntimeError("TTS server unreachable")

            def render_clip(clip_id, plan):
                paths.clip(clip_id).write_bytes(b"x" * 4096)

            def assemble_final():
                paths.final.write_bytes(b"x" * 4096)
                return paths.final

            pipeline = ParallelPipeline(
                clips=clips,
                paths=paths,
                plans={1: {}, 2: {}},
                generate_images=generate_images,
                generate_voices=generate_voices,
                render_clip=render_clip,
                assemble_final=assemble_final,
                ledger=self._ledger(tmp),
                poll_seconds=0.02,
                settle_seconds=0.0,
                max_stage_retries=0,
            )

            result = pipeline.run()

            self.assertFalse(result.complete)
            self.assertEqual(result.missing_voices, [1, 2])
            self.assertTrue(
                any("TTS server unreachable" in error for error in result.errors)
            )

    def test_voice_failure_retries_then_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(tmp)
            clips = [FakeClip(1), FakeClip(2)]
            attempts = {"voices": 0}

            def generate_images():
                for clip in clips:
                    paths.image(clip.id).write_bytes(b"x" * 2048)
                paths.thumbnail_file.write_bytes(b"x" * 2048)
                return False, None

            def generate_voices():
                attempts["voices"] += 1
                if attempts["voices"] == 1:
                    raise RuntimeError("TTS server restarting")
                for clip in clips:
                    paths.audio_file(clip.id).write_bytes(b"x" * 8192)

            def render_clip(clip_id, plan):
                paths.clip(clip_id).write_bytes(b"x" * 4096)

            def assemble_final():
                paths.final.write_bytes(b"x" * 4096)
                return paths.final

            pipeline = ParallelPipeline(
                clips=clips,
                paths=paths,
                plans={1: {}, 2: {}},
                generate_images=generate_images,
                generate_voices=generate_voices,
                render_clip=render_clip,
                assemble_final=assemble_final,
                ledger=self._ledger(tmp),
                poll_seconds=0.02,
                settle_seconds=0.0,
                max_stage_retries=2,
                stage_retry_seconds=0.01,
            )

            result = pipeline.run()

            self.assertTrue(result.complete)
            self.assertEqual(attempts["voices"], 2)
            self.assertEqual(result.missing_clips, [])

    def test_image_failure_retries_then_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(tmp)
            clips = [FakeClip(1)]
            attempts = {"images": 0}

            def generate_images():
                attempts["images"] += 1
                if attempts["images"] == 1:
                    raise RuntimeError("Chrome disconnected")
                for clip in clips:
                    paths.image(clip.id).write_bytes(b"x" * 2048)
                paths.thumbnail_file.write_bytes(b"x" * 2048)
                return False, None

            def generate_voices():
                for clip in clips:
                    paths.audio_file(clip.id).write_bytes(b"x" * 8192)

            def render_clip(clip_id, plan):
                paths.clip(clip_id).write_bytes(b"x" * 4096)

            def assemble_final():
                paths.final.write_bytes(b"x" * 4096)
                return paths.final

            pipeline = ParallelPipeline(
                clips=clips,
                paths=paths,
                plans={1: {}},
                generate_images=generate_images,
                generate_voices=generate_voices,
                render_clip=render_clip,
                assemble_final=assemble_final,
                ledger=self._ledger(tmp),
                poll_seconds=0.02,
                settle_seconds=0.0,
                max_stage_retries=2,
                stage_retry_seconds=0.01,
            )

            result = pipeline.run()

            self.assertTrue(result.complete)
            self.assertEqual(attempts["images"], 2)

    def test_gemini_limit_blocks_images_but_voices_still_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(tmp)
            clips = [FakeClip(1)]
            ledger = self._ledger(tmp)

            def generate_images():
                return True, "Gemini limit reached"

            def generate_voices():
                paths.audio_file(1).write_bytes(b"x" * 8192)

            def render_clip(clip_id, plan):
                paths.clip(clip_id).write_bytes(b"x" * 4096)

            def assemble_final():
                paths.final.write_bytes(b"x" * 4096)
                return paths.final

            pipeline = ParallelPipeline(
                clips=clips,
                paths=paths,
                plans={1: {}},
                generate_images=generate_images,
                generate_voices=generate_voices,
                render_clip=render_clip,
                assemble_final=assemble_final,
                ledger=ledger,
                poll_seconds=0.02,
                settle_seconds=0.0,
                cooldown_minutes=30.0,
                wait_for_limit=False,
            )

            started = time.time()
            result = pipeline.run()
            elapsed = time.time() - started

            self.assertFalse(result.complete)
            self.assertTrue(result.gemini_blocked)
            self.assertEqual(result.limit_message, "Gemini limit reached")
            self.assertEqual(result.missing_images, [1])
            self.assertEqual(result.missing_voices, [])
            self.assertTrue(ledger.is_blocked("gemini"))
            self.assertLess(elapsed, 10.0)


if __name__ == "__main__":
    unittest.main()
