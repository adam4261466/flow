"""Critics that turn raw assets into a score plus an actionable fix."""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.health import AudioHealth, ImageHealth, inspect_audio, inspect_image
from agents.text import clamp_score, title_matches


@dataclass
class ImageReview:
    score: float
    issues: list[str] = field(default_factory=list)
    prompt_fix: str = ""
    health: ImageHealth | None = None
    backend: str = "none"


@dataclass
class ThumbnailReview:
    score: float
    actual_text: str = ""
    text_matches: bool | None = None
    issues: list[str] = field(default_factory=list)
    prompt_fix: str = ""
    health: ImageHealth | None = None
    backend: str = "none"

    @property
    def accepted(self) -> bool:
        """Thumbnails are accepted when healthy and not known to be mislabeled."""

        return bool(self.health and self.health.ok) and self.text_matches is not False


class ImageCritic:
    """Technical check + optional semantic critique for one clip image."""

    def __init__(self, vision, state=None, threshold: float = 7.0) -> None:
        self.vision = vision
        self.state = state
        self.threshold = threshold

    @staticmethod
    def _brief(clip, plan: dict, story_text: str | None) -> str:
        plan = plan or {}
        context = (story_text or "")[:1200]
        return (
            f"Clip {clip.id}\n"
            f"Characters: {clip.characters}\n"
            f"Moment: {clip.moment}\n"
            f"Vibe: {clip.vibe}\n"
            f"Style: {clip.style}\n"
            f"Requested camera: {plan.get('camera', '')}\n"
            f"Requested composition: {plan.get('composition', '')}\n"
            f"Story context: {context or '(not provided)'}"
        )

    def evaluate(
        self,
        image_path,
        clip,
        plan: dict,
        *,
        story_text: str | None = None,
        same_conversation: bool = True,
    ) -> ImageReview:
        health = inspect_image(image_path)
        issues = list(health.issues)
        score = 10.0
        fix = ""
        backend = getattr(self.vision, "name", "none")

        if not health.ok:
            score = 2.0
        elif not getattr(self.vision, "available", False):
            issues.append(
                "semantic critic unavailable; accepted on technical checks"
            )
        elif self.state is not None and not self.state.budget.can_spend_critic():
            issues.append(
                "critic call budget exhausted; accepted on technical checks"
            )
        else:
            if self.state is not None:
                self.state.budget.spend_critic()
            try:
                review = self.vision.critique_image(
                    image_path,
                    self._brief(clip, plan, story_text),
                    same_conversation=same_conversation,
                )
                score = clamp_score(review.score)
                issues.extend(review.issues)
                fix = review.prompt_fix
                backend = review.backend or backend
            except Exception as exc:  # noqa: BLE001 - a critic must never break a run
                issues.append(f"critic failed: {exc}")
                score = 7.0

        if not health.ok:
            score = min(score, 2.0)

        return ImageReview(
            score=score,
            issues=issues,
            prompt_fix=fix,
            health=health,
            backend=backend,
        )


class ThumbnailCritic:
    """Check the thumbnail's health and verify the rendered title with OCR."""

    def __init__(
        self,
        vision,
        expected_title: str,
        state=None,
    ) -> None:
        self.vision = vision
        self.expected_title = expected_title
        self.state = state

    def check(
        self,
        image_path,
        *,
        same_conversation: bool = True,
    ) -> ThumbnailReview:
        health = inspect_image(
            image_path,
            min_width=640,
            min_height=360,
            expected_aspect=16 / 9,
            aspect_tolerance=0.35,
        )
        issues = list(health.issues)
        score = 10.0
        actual = ""
        matches: bool | None = None
        fix = ""
        backend = getattr(self.vision, "name", "none")

        if not health.ok:
            score = 2.0
            fix = (
                "Re-render as a 16:9 landscape movie-poster composition "
                "framed for 1280x720. "
            )

        if getattr(self.vision, "available", False):
            if self.state is not None and not self.state.budget.can_spend_critic():
                issues.append(
                    "OCR critic budget exhausted; accepted on technical checks"
                )
            else:
                if self.state is not None:
                    self.state.budget.spend_critic()
                try:
                    actual = self.vision.read_text(
                        image_path,
                        same_conversation=same_conversation,
                    )
                    matches = title_matches(self.expected_title, actual)
                    if not matches:
                        score = min(score, 3.0)
                        issues.append(
                            "title text did not match: "
                            f'expected "{self.expected_title}", read "{actual}"'
                        )
                        fix = (
                            "CRITICAL TITLE CORRECTION: The title text must "
                            f'read EXACTLY "{self.expected_title}". '
                            f'The previous render read "{actual}". '
                            "Re-render the title with correct spelling, "
                            "capitalisation and punctuation, large and bold, "
                            "inside the safe margins. Do not add any other text."
                        )
                except Exception as exc:  # noqa: BLE001
                    issues.append(f"OCR critic failed: {exc}")
                    score = min(score, 7.0)
                    matches = None
        else:
            issues.append(
                "OCR critic unavailable; thumbnail accepted on technical checks"
            )

        return ThumbnailReview(
            score=score,
            actual_text=actual,
            text_matches=matches,
            issues=issues,
            prompt_fix=fix,
            health=health,
            backend=backend,
        )


class VoiceCritic:
    """Technical audio checks plus a small, safe retry policy."""

    def inspect(self, audio_path, script_text: str) -> AudioHealth:
        return inspect_audio(audio_path, script_text)

    @staticmethod
    def retry_overrides(
        attempt: int,
        health: AudioHealth,
        base_speed: float = 1.0,
    ) -> dict:
        issues = " ".join(health.issues).lower()
        overrides: dict = {}
        speed = base_speed

        if "longer" in issues or "too slow" in issues:
            speed = min(1.18, base_speed * (1.0 + 0.04 * attempt))
        elif "shorter" in issues or "too fast" in issues:
            speed = max(0.82, base_speed * (1.0 - 0.04 * attempt))

        if "clipping" in issues:
            speed = max(0.82, speed * 0.98)

        if speed != base_speed:
            overrides["speed_factor"] = round(speed, 3)

        if "silent" in issues or "too short" in issues:
            overrides["temperature"] = min(1.0, 0.8 + 0.05 * attempt)
            overrides["seed"] = 42 + attempt

        return overrides
