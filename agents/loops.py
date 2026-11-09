"""Reusable generate -> critique -> promote loops.

The loops know nothing about Gemini, FFmpeg or main.py.  Callers inject:

- ``generate(attempt, revision)``  -> candidate path
- ``evaluate/check/inspect(path)`` -> review or health object
- ``promote(source, destination)`` -> move the winner into place

That keeps the pipeline behaviour untouched when critics are disabled and
makes the loops easy to test with fakes.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


def promote_file(source, destination) -> Path:
    """Copy a candidate into its final name atomically."""

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.resolve() == destination.resolve():
        return destination

    temporary = destination.with_name(destination.name + ".promote.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return destination


def _remove(path) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


@dataclass
class ImageAttempt:
    index: int
    path: Path | None
    score: float = -1.0
    issues: list[str] = field(default_factory=list)
    prompt_fix: str = ""
    accepted: bool = False


@dataclass
class ImageLoopResult:
    path: Path | None
    best_score: float
    attempts: list[ImageAttempt]
    accepted: bool


def run_image_qc_loop(
    *,
    generate,
    evaluate=None,
    final_path,
    max_attempts: int = 3,
    threshold: float = 7.0,
    fatal_exceptions: tuple[type[BaseException], ...] = (),
    promote=promote_file,
    log=print,
) -> ImageLoopResult:
    """Generate up to ``max_attempts`` images and keep the best one.

    ``evaluate`` may return ``None`` when critics are disabled, in which case
    the first successfully generated image is accepted.
    """

    attempts: list[ImageAttempt] = []
    best_path: Path | None = None
    best_score = -1.0
    revision: str | None = None

    for index in range(1, max(1, max_attempts) + 1):
        path: Path | None = None
        try:
            path = Path(generate(index, revision))
        except fatal_exceptions:
            raise
        except Exception as exc:  # noqa: BLE001 - retry transient failures
            log(f"[agents] image attempt {index} failed: {exc}")
            attempts.append(
                ImageAttempt(index, None, -1.0, [str(exc)], "", False)
            )
            continue

        if not path.is_file() or path.stat().st_size <= 0:
            log(f"[agents] image attempt {index} produced no usable file")
            attempts.append(
                ImageAttempt(index, path, -1.0, ["empty image"], "", False)
            )
            continue

        review = None
        if evaluate is not None:
            try:
                review = evaluate(path, index)
            except fatal_exceptions:
                raise
            except Exception as exc:  # noqa: BLE001
                log(f"[agents] image critic attempt {index} failed: {exc}")

        score = float(review.score) if review is not None else 10.0
        issues = list(review.issues) if review is not None else []
        fix = review.prompt_fix if review is not None else ""
        accepted = score >= threshold

        attempts.append(
            ImageAttempt(index, path, score, issues, fix, accepted)
        )

        if score > best_score:
            best_path = path
            best_score = score

        if accepted:
            log(f"[agents] image attempt {index} accepted (score {score:.1f})")
            break

        log(
            f"[agents] image attempt {index} rejected "
            f"(score {score:.1f} < {threshold:.1f})"
        )
        if issues:
            log("[agents] issues: " + "; ".join(issues[:4]))
        if fix:
            revision = fix

    if best_path is None:
        log("[agents] no usable image was produced")
        return ImageLoopResult(None, best_score, attempts, False)

    promote(best_path, final_path)

    for attempt in attempts:
        if attempt.path and attempt.path.resolve() != best_path.resolve():
            _remove(attempt.path)
    if best_path.resolve() != Path(final_path).resolve():
        _remove(best_path)

    accepted = best_score >= threshold
    if not accepted:
        log(
            "[agents] keeping the best available image despite the low score "
            f"({best_score:.1f})"
        )

    return ImageLoopResult(
        Path(final_path), best_score, attempts, accepted
    )


@dataclass
class ThumbnailAttempt:
    index: int
    path: Path | None
    score: float = -1.0
    actual_text: str = ""
    text_matches: bool | None = None
    issues: list[str] = field(default_factory=list)
    accepted: bool = False


@dataclass
class ThumbnailLoopResult:
    path: Path | None
    attempts: list[ThumbnailAttempt]
    accepted: bool


def run_thumbnail_qc_loop(
    *,
    generate,
    check=None,
    final_path,
    max_attempts: int = 3,
    fatal_exceptions: tuple[type[BaseException], ...] = (),
    promote=promote_file,
    log=print,
) -> ThumbnailLoopResult:
    """Generate thumbnails until the title text is verified or budget runs out."""

    attempts: list[ThumbnailAttempt] = []
    best_path: Path | None = None
    best_key = (-1, -1.0)
    revision: str | None = None

    for index in range(1, max(1, max_attempts) + 1):
        path: Path | None = None
        try:
            path = Path(generate(index, revision))
        except fatal_exceptions:
            raise
        except Exception as exc:  # noqa: BLE001
            log(f"[agents] thumbnail attempt {index} failed: {exc}")
            attempts.append(
                ThumbnailAttempt(index, None, -1.0, "", None, [str(exc)], False)
            )
            continue

        if not path.is_file() or path.stat().st_size <= 0:
            attempts.append(
                ThumbnailAttempt(
                    index, path, -1.0, "", None, ["empty thumbnail"], False
                )
            )
            continue

        review = None
        if check is not None:
            try:
                review = check(path, index)
            except fatal_exceptions:
                raise
            except Exception as exc:  # noqa: BLE001
                log(f"[agents] thumbnail critic attempt {index} failed: {exc}")

        if review is None:
            score = 10.0
            matches = None
            issues: list[str] = []
            fix = ""
            accepted = True
        else:
            score = float(review.score)
            matches = review.text_matches
            issues = list(review.issues)
            fix = review.prompt_fix
            accepted = bool(review.accepted)

        attempts.append(
            ThumbnailAttempt(
                index, path, score, getattr(review, "actual_text", ""),
                matches, issues, accepted
            )
        )

        key = (1 if matches is True else 0, score)
        if key > best_key:
            best_path = path
            best_key = key

        if accepted:
            log(
                f"[agents] thumbnail attempt {index} accepted "
                f"(score {score:.1f})"
            )
            break

        log(
            f"[agents] thumbnail attempt {index} rejected "
            f"(score {score:.1f})"
        )
        if issues:
            log("[agents] issues: " + "; ".join(issues[:4]))
        if fix:
            revision = fix

    if best_path is None:
        log("[agents] no usable thumbnail was produced")
        return ThumbnailLoopResult(None, attempts, False)

    promote(best_path, final_path)

    for attempt in attempts:
        if attempt.path and attempt.path.resolve() != best_path.resolve():
            _remove(attempt.path)
    if best_path.resolve() != Path(final_path).resolve():
        _remove(best_path)

    accepted = best_key[0] == 1 or best_key[1] >= 10.0
    return ThumbnailLoopResult(Path(final_path), attempts, accepted)


@dataclass
class VoiceAttempt:
    index: int
    path: Path | None
    ok: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class VoiceLoopResult:
    path: Path | None
    attempts: list[VoiceAttempt]
    accepted: bool


def run_voice_qc_loop(
    *,
    generate,
    inspect,
    final_path,
    max_attempts: int = 2,
    retry_overrides=None,
    fatal_exceptions: tuple[type[BaseException], ...] = (),
    promote=promote_file,
    log=print,
) -> VoiceLoopResult:
    """Generate narration and retry once when the audio is clearly wrong."""

    attempts: list[VoiceAttempt] = []
    best_path: Path | None = None
    best_issue_count = 10**9
    overrides: dict = {}

    for index in range(1, max(1, max_attempts) + 1):
        path: Path | None = None
        try:
            path = Path(generate(index, overrides))
        except fatal_exceptions:
            raise
        except Exception as exc:  # noqa: BLE001
            log(f"[agents] voice attempt {index} failed: {exc}")
            attempts.append(VoiceAttempt(index, None, False, [str(exc)]))
            continue

        if not path.is_file() or path.stat().st_size <= 0:
            attempts.append(
                VoiceAttempt(index, path, False, ["empty audio"])
            )
            continue

        try:
            health = inspect(path)
        except Exception as exc:  # noqa: BLE001
            log(f"[agents] voice check attempt {index} failed: {exc}")
            attempts.append(
                VoiceAttempt(index, path, False, [f"check failed: {exc}"])
            )
            continue

        attempts.append(
            VoiceAttempt(index, path, bool(health.ok), list(health.issues))
        )

        if health.ok:
            promote(path, final_path)
            for attempt in attempts:
                if attempt.path and attempt.path.resolve() != path.resolve():
                    _remove(attempt.path)
            if path.resolve() != Path(final_path).resolve():
                _remove(path)
            log(f"[agents] voice attempt {index} accepted")
            return VoiceLoopResult(Path(final_path), attempts, True)

        log(
            f"[agents] voice attempt {index} rejected: "
            + "; ".join(health.issues[:4])
        )
        if len(health.issues) < best_issue_count:
            best_path = path
            best_issue_count = len(health.issues)

        if retry_overrides is not None:
            overrides = retry_overrides(index, health) or {}

    if best_path is None:
        log("[agents] no usable narration was produced")
        return VoiceLoopResult(None, attempts, False)

    promote(best_path, final_path)
    for attempt in attempts:
        if attempt.path and attempt.path.resolve() != best_path.resolve():
            _remove(attempt.path)
    if best_path.resolve() != Path(final_path).resolve():
        _remove(best_path)

    log("[agents] keeping the best available narration despite issues")
    return VoiceLoopResult(Path(final_path), attempts, False)
