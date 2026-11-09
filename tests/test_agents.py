"""Tests for the Level-2 critic loops.

Run with::

    py -3.13 -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import tempfile
import unittest
import wave
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import health  # noqa: E402
from agents.config import AgentConfig  # noqa: E402
from agents.critics import ImageCritic, ThumbnailCritic  # noqa: E402
from agents.loops import (  # noqa: E402
    promote_file,
    run_image_qc_loop,
    run_thumbnail_qc_loop,
    run_voice_qc_loop,
)
from agents.state import Budget, ProductionState  # noqa: E402
from agents.text import (  # noqa: E402
    extract_json_object,
    normalize_for_match,
    title_matches,
)
from agents.vision import (  # noqa: E402
    GeminiWebVision,
    NullVision,
    build_vision_backend,
)


def args(**overrides):
    values = {
        "no_agents": False,
        "vision_backend": None,
        "vision_model": None,
        "image_threshold": None,
        "max_image_retries": None,
        "max_thumbnail_retries": None,
        "max_voice_retries": None,
        "max_critic_calls": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class AgentConfigTests(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = AgentConfig.from_args(args())
        self.assertTrue(config.enabled)
        self.assertEqual(config.vision_backend, "auto")
        self.assertEqual(config.image_threshold, 7.0)
        self.assertEqual(config.max_image_retries, 2)
        self.assertEqual(config.max_thumbnail_retries, 2)
        self.assertEqual(config.max_voice_retries, 1)
        self.assertEqual(config.max_critic_calls, 60)

    def test_no_agents_disables(self):
        with patch.dict(os.environ, {}, clear=True):
            config = AgentConfig.from_args(args(no_agents=True))
        self.assertFalse(config.enabled)
        self.assertIn("disabled", config.describe())

    def test_clamps_threshold_and_retries(self):
        with patch.dict(os.environ, {}, clear=True):
            config = AgentConfig.from_args(
                args(
                    image_threshold=99,
                    max_image_retries=50,
                    max_voice_retries=-4,
                    max_critic_calls=-1,
                )
            )
        self.assertEqual(config.image_threshold, 10.0)
        self.assertEqual(config.max_image_retries, 10)
        self.assertEqual(config.max_voice_retries, 0)
        self.assertEqual(config.max_critic_calls, 0)

    def test_rejects_unknown_backend(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                AgentConfig.from_args(args(vision_backend="magic"))

    def test_env_defaults_are_used(self):
        with patch.dict(
            os.environ,
            {
                "FLOW_AGENTS": "false",
                "FLOW_VISION_BACKEND": "none",
                "FLOW_IMAGE_THRESHOLD": "8.5",
                "FLOW_MAX_IMAGE_RETRIES": "1",
            },
            clear=True,
        ):
            config = AgentConfig.from_args(args())
        self.assertFalse(config.enabled)
        self.assertEqual(config.vision_backend, "none")
        self.assertEqual(config.image_threshold, 8.5)
        self.assertEqual(config.max_image_retries, 1)


class TextTests(unittest.TestCase):
    def test_normalize_ignores_punctuation_and_case(self):
        self.assertEqual(
            normalize_for_match("“The Last Light”—Part 1!"),
            "the last light part 1",
        )

    def test_title_matches_exactly(self):
        self.assertTrue(title_matches("The Last Light", "the last light."))
        self.assertFalse(title_matches("The Last Light", "The Lost Light"))
        self.assertFalse(title_matches("The Last Light", ""))

    def test_extract_json_from_fence(self):
        raw = 'Here you go:\n```json\n{"score": 8, "issues": []}\n```'
        self.assertEqual(extract_json_object(raw), {"score": 8, "issues": []})

    def test_extract_json_with_surrounding_text(self):
        raw = 'Sure. {"text": "A Title"} Thanks.'
        self.assertEqual(extract_json_object(raw), {"text": "A Title"})


class FakeVision:
    name = "fake"
    available = True

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = 0

    def critique_image(self, image_path, brief, *, same_conversation=True):
        self.calls += 1
        return self.replies.pop(0)

    def read_text(self, image_path, *, same_conversation=True):
        self.calls += 1
        return self.replies.pop(0)


@dataclass
class FakeImageReview:
    score: float
    issues: list
    prompt_fix: str = ""
    backend: str = "fake"


@dataclass
class FakeThumbReview:
    score: float
    actual_text: str
    text_matches: bool
    issues: list
    prompt_fix: str = ""

    @property
    def accepted(self):
        return self.text_matches


@dataclass
class FakeHealth:
    ok: bool
    issues: list


class CriticTests(unittest.TestCase):
    def test_image_critic_respects_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ProductionState(Path(tmp), "story_x")
            state.budget = Budget(critic_limit=1)
            vision = FakeVision([FakeImageReview(8.0, [])])
            critic = ImageCritic(vision, state=state, threshold=7.0)

            first = critic.evaluate(
                self._image(Path(tmp)),
                self._clip(),
                {},
            )
            second = critic.evaluate(
                self._image(Path(tmp)),
                self._clip(),
                {},
            )

        self.assertEqual(first.score, 8.0)
        self.assertEqual(vision.calls, 1)
        self.assertEqual(second.score, 10.0)
        self.assertTrue(
            any("budget" in issue for issue in second.issues)
        )

    def test_image_critic_falls_back_when_backend_raises(self):
        class Broken(FakeVision):
            def critique_image(self, *a, **k):
                raise RuntimeError("browser closed")

        with tempfile.TemporaryDirectory() as tmp:
            critic = ImageCritic(Broken(), threshold=7.0)
            review = critic.evaluate(
                self._image(Path(tmp)),
                self._clip(),
                {},
            )

        self.assertEqual(review.score, 7.0)
        self.assertTrue(any("critic failed" in i for i in review.issues))

    def test_thumbnail_critic_flags_wrong_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._image(Path(tmp))
            critic = ThumbnailCritic(
                FakeVision(["The Lost Light"]),
                expected_title="The Last Light",
            )
            review = critic.check(path, same_conversation=False)

        self.assertFalse(review.accepted)
        self.assertEqual(review.score, 3.0)
        self.assertIn("The Last Light", review.prompt_fix)

    def test_thumbnail_critic_accepts_exact_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._image(Path(tmp))
            critic = ThumbnailCritic(
                FakeVision(["The Last Light"]),
                expected_title="The Last Light",
            )
            review = critic.check(path, same_conversation=False)

        self.assertTrue(review.accepted)
        self.assertIs(review.text_matches, True)

    @staticmethod
    def _image(directory: Path) -> Path:
        from PIL import Image

        path = directory / "image.png"
        image = Image.new("RGB", (960, 720))
        for x in range(960):
            for y in range(0, 720, 8):
                image.putpixel((x, y), (x % 255, y % 255, 120))
        image.save(path)
        return path

    @staticmethod
    def _clip():
        return argparse.Namespace(
            id=1,
            characters="Mara",
            moment="She opens the door.",
            vibe="tense",
            style="cinematic",
        )


class ImageLoopTests(unittest.TestCase):
    def test_accepts_first_image_when_critics_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def generate(attempt, revision):
                path = tmp / f"candidate_{attempt}.png"
                path.write_bytes(b"image")
                return path

            result = run_image_qc_loop(
                generate=generate,
                evaluate=None,
                final_path=tmp / "final.png",
                max_attempts=3,
                log=lambda *a, **k: None,
            )

            self.assertTrue(result.accepted)
            self.assertEqual((tmp / "final.png").read_bytes(), b"image")
            self.assertFalse((tmp / "candidate_1.png").exists())

    def test_retries_and_keeps_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            scores = [5.0, 9.0]

            def generate(attempt, revision):
                path = tmp / f"candidate_{attempt}.png"
                path.write_bytes(f"image-{attempt}".encode())
                return path

            def evaluate(path, attempt):
                return FakeImageReview(scores[attempt - 1], [])

            result = run_image_qc_loop(
                generate=generate,
                evaluate=evaluate,
                final_path=tmp / "final.png",
                max_attempts=3,
                threshold=7.0,
                log=lambda *a, **k: None,
            )

            self.assertTrue(result.accepted)
            self.assertEqual(result.best_score, 9.0)
            self.assertEqual((tmp / "final.png").read_bytes(), b"image-2")
            self.assertFalse((tmp / "candidate_1.png").exists())
            self.assertFalse((tmp / "candidate_2.png").exists())

    def test_keeps_best_when_nothing_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            scores = [5.0, 4.0, 3.0]

            def generate(attempt, revision):
                path = tmp / f"candidate_{attempt}.png"
                path.write_bytes(f"image-{attempt}".encode())
                return path

            def evaluate(path, attempt):
                return FakeImageReview(scores[attempt - 1], [])

            result = run_image_qc_loop(
                generate=generate,
                evaluate=evaluate,
                final_path=tmp / "final.png",
                max_attempts=3,
                threshold=7.0,
                log=lambda *a, **k: None,
            )

            self.assertFalse(result.accepted)
            self.assertEqual((tmp / "final.png").read_bytes(), b"image-1")

    def test_critic_exception_does_not_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def generate(attempt, revision):
                path = tmp / f"candidate_{attempt}.png"
                path.write_bytes(b"image")
                return path

            def evaluate(path, attempt):
                raise RuntimeError("model exploded")

            result = run_image_qc_loop(
                generate=generate,
                evaluate=evaluate,
                final_path=tmp / "final.png",
                max_attempts=3,
                log=lambda *a, **k: None,
            )

            self.assertTrue(result.accepted)
            self.assertTrue((tmp / "final.png").exists())

    def test_fatal_exception_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def generate(attempt, revision):
                raise RuntimeError("account limit")

            with self.assertRaises(RuntimeError):
                run_image_qc_loop(
                    generate=generate,
                    evaluate=None,
                    final_path=tmp / "final.png",
                    max_attempts=3,
                    fatal_exceptions=(RuntimeError,),
                    log=lambda *a, **k: None,
                )


class ThumbnailLoopTests(unittest.TestCase):
    def test_retries_until_title_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            reviews = [
                FakeThumbReview(3.0, "Wrong", False, ["bad title"]),
                FakeThumbReview(10.0, "The Last Light", True, []),
            ]

            def generate(attempt, revision):
                path = tmp / f"thumb_{attempt}.png"
                path.write_bytes(f"thumb-{attempt}".encode())
                return path

            def check(path, attempt):
                return reviews[attempt - 1]

            result = run_thumbnail_qc_loop(
                generate=generate,
                check=check,
                final_path=tmp / "thumbnail.png",
                max_attempts=3,
                log=lambda *a, **k: None,
            )

            self.assertTrue(result.accepted)
            self.assertEqual(
                (tmp / "thumbnail.png").read_bytes(), b"thumb-2"
            )
            self.assertFalse((tmp / "thumb_1.png").exists())

    def test_prefers_verified_title_over_higher_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            reviews = [
                FakeThumbReview(9.0, "Wrong", False, ["bad title"]),
                FakeThumbReview(3.0, "The Last Light", True, []),
            ]

            def generate(attempt, revision):
                path = tmp / f"thumb_{attempt}.png"
                path.write_bytes(f"thumb-{attempt}".encode())
                return path

            def check(path, attempt):
                return reviews[attempt - 1]

            result = run_thumbnail_qc_loop(
                generate=generate,
                check=check,
                final_path=tmp / "thumbnail.png",
                max_attempts=2,
                log=lambda *a, **k: None,
            )

            self.assertTrue(result.accepted)
            self.assertEqual(
                (tmp / "thumbnail.png").read_bytes(), b"thumb-2"
            )


class VoiceLoopTests(unittest.TestCase):
    def test_retries_until_audio_is_healthy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            checks = [
                FakeHealth(False, ["audio is clipping"]),
                FakeHealth(True, []),
            ]

            def generate(attempt, overrides):
                path = tmp / f"audio_{attempt}.wav"
                path.write_bytes(f"audio-{attempt}".encode())
                return path

            def inspect(path):
                return checks[0] if "audio_1" in path.name else checks[1]

            result = run_voice_qc_loop(
                generate=generate,
                inspect=inspect,
                final_path=tmp / "audio.wav",
                max_attempts=2,
                retry_overrides=lambda attempt, health: {
                    "speed_factor": 0.98
                },
                log=lambda *a, **k: None,
            )

            self.assertTrue(result.accepted)
            self.assertEqual(
                (tmp / "audio.wav").read_bytes(), b"audio-2"
            )

    def test_keeps_best_effort_when_all_attempts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def generate(attempt, overrides):
                path = tmp / f"audio_{attempt}.wav"
                path.write_bytes(f"audio-{attempt}".encode())
                return path

            def inspect(path):
                if path.name == "audio_1.wav":
                    return FakeHealth(False, ["a", "b", "c"])
                return FakeHealth(False, ["a"])

            result = run_voice_qc_loop(
                generate=generate,
                inspect=inspect,
                final_path=tmp / "audio.wav",
                max_attempts=2,
                log=lambda *a, **k: None,
            )

            self.assertFalse(result.accepted)
            self.assertEqual(
                (tmp / "audio.wav").read_bytes(), b"audio-2"
            )


class StateTests(unittest.TestCase):
    def test_round_trip_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            state = ProductionState(tmp, "story_009")
            state.budget.critic_limit = 2
            state.budget.spend_critic()
            state.record_decision("image_critic", "promote", clip=1)
            state.record_issue("voice_critic", "too quiet", clip=2)
            state.save()

            loaded = ProductionState.load_or_create(tmp, "story_009")

        self.assertEqual(loaded.budget.critic_calls, 1)
        self.assertEqual(loaded.budget.critic_limit, 2)
        self.assertTrue(loaded.budget.can_spend_critic())
        self.assertEqual(len(loaded.data["decisions"]), 1)
        self.assertEqual(len(loaded.data["issues"]), 1)

    def test_promote_file_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "candidate.png"
            target = tmp / "final.png"
            source.write_bytes(b"new")
            target.write_bytes(b"old")

            promote_file(source, target)

            self.assertEqual(target.read_bytes(), b"new")
            self.assertEqual(source.read_bytes(), b"new")


class VisionBackendTests(unittest.TestCase):
    def test_none_backend(self):
        config = AgentConfig(vision_backend="none")
        backend = build_vision_backend(config, log=lambda *a, **k: None)
        self.assertIsInstance(backend, NullVision)

    def test_disabled_backend(self):
        config = AgentConfig(enabled=False)
        backend = build_vision_backend(config, log=lambda *a, **k: None)
        self.assertIsInstance(backend, NullVision)

    def test_gemini_follow_up_uses_same_conversation(self):
        class FakeBrowser:
            def __init__(self):
                self.kwargs = None

            def ask_text(self, prompt, **kwargs):
                self.kwargs = kwargs
                return '{"score": 8, "issues": [], "prompt_fix": ""}'

        browser = FakeBrowser()
        vision = GeminiWebVision(browser)
        review = vision.critique_image(
            "image.png",
            "brief",
            same_conversation=True,
        )

        self.assertEqual(review.score, 8.0)
        self.assertFalse(browser.kwargs["new_conversation"])
        self.assertIsNone(browser.kwargs["reference_image"])

    def test_gemini_new_conversation_uploads_image(self):
        class FakeBrowser:
            def ask_text(self, prompt, **kwargs):
                self.kwargs = kwargs
                return '{"text": "The Last Light"}'

        browser = FakeBrowser()
        vision = GeminiWebVision(browser)
        text = vision.read_text("thumb.png", same_conversation=False)

        self.assertEqual(text, "The Last Light")
        self.assertTrue(browser.kwargs["new_conversation"])
        self.assertEqual(browser.kwargs["reference_image"], "thumb.png")


class HealthTests(unittest.TestCase):
    def test_missing_image_is_unhealthy(self):
        result = health.inspect_image("does-not-exist.png")
        self.assertFalse(result.ok)
        self.assertTrue(result.issues)

    def test_blank_image_is_unhealthy(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blank.png"
            Image.new("RGB", (720, 480), (128, 128, 128)).save(path)
            result = health.inspect_image(path)

        self.assertFalse(result.ok)
        self.assertTrue(any("flat" in issue for issue in result.issues))

    @unittest.skipIf(health.np is None, "numpy is required")
    @unittest.skipIf(
        health._ffmpeg_executable() is None,
        "ffmpeg is required",
    )
    def test_audio_health_detects_silence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "silence.wav"
            self._write_wav(path, amplitude=0.0)
            result = health.inspect_audio(path, "hello world")

        self.assertFalse(result.ok)
        self.assertTrue(
            any("silent" in issue for issue in result.issues)
        )

    @unittest.skipIf(health.np is None, "numpy is required")
    @unittest.skipIf(
        health._ffmpeg_executable() is None,
        "ffmpeg is required",
    )
    def test_audio_health_accepts_normal_speech_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.wav"
            self._write_wav(path, amplitude=0.3)
            result = health.inspect_audio(path, "hello world")

        self.assertTrue(result.ok)

    @staticmethod
    def _write_wav(path: Path, amplitude: float) -> None:
        sample_rate = 16000
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            frames = bytearray()
            for index in range(sample_rate):
                value = amplitude * math.sin(
                    2 * math.pi * 220 * index / sample_rate
                )
                frames += struct.pack("<h", int(value * 32767))
            handle.writeframes(bytes(frames))


if __name__ == "__main__":
    unittest.main()
