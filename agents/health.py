"""Cheap technical checks that run even when no semantic critic is available.

These catch the failure modes that waste the most time downstream: blank or
corrupt images, near-silent narration, clipped audio and wildly wrong pacing.
They are intentionally forgiving; the semantic critic handles taste.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a normal install
    np = None


@dataclass
class ImageHealth:
    ok: bool
    width: int = 0
    height: int = 0
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    issues: list[str] = field(default_factory=list)


@dataclass
class AudioHealth:
    ok: bool
    duration: float = 0.0
    expected_duration: float = 0.0
    peak: float = 0.0
    rms: float = 0.0
    clipping_ratio: float = 0.0
    issues: list[str] = field(default_factory=list)


def inspect_image(
    path,
    *,
    min_width: int = 512,
    min_height: int = 512,
    expected_aspect: float | None = None,
    aspect_tolerance: float = 0.25,
) -> ImageHealth:
    """Return basic quality facts for a generated still or thumbnail."""

    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return ImageHealth(False, issues=["image file is missing or empty"])

    try:
        with Image.open(path) as source:
            width, height = source.size
            rgb = source.convert("RGB")
            stats = ImageStat.Stat(rgb)
            brightness = sum(stats.mean) / 3.0
            contrast = sum(stats.stddev) / 3.0
            edges = rgb.convert("L").filter(ImageFilter.FIND_EDGES)
            sharpness = ImageStat.Stat(edges).stddev[0]
    except Exception as exc:  # noqa: BLE001 - a corrupt image is just unhealthy
        return ImageHealth(False, issues=[f"could not open image: {exc}"])

    issues: list[str] = []

    if width < min_width or height < min_height:
        issues.append(
            f"resolution {width}x{height} is below {min_width}x{min_height}"
        )

    if brightness < 3.0:
        issues.append("image is almost completely black")
    elif brightness > 252.0:
        issues.append("image is almost completely white")

    if contrast < 5.0:
        issues.append("image looks flat or blank (very low contrast)")

    if sharpness < 2.5:
        issues.append("image looks blurry or lacks detail")

    if expected_aspect and height:
        aspect = width / height
        difference = abs(aspect - expected_aspect) / expected_aspect
        if difference > aspect_tolerance:
            issues.append(
                f"aspect ratio {aspect:.2f} is far from {expected_aspect:.2f}"
            )

    return ImageHealth(
        ok=not issues,
        width=width,
        height=height,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        issues=issues,
    )


def _ffmpeg_executable() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


def _decode_samples(path: Path) -> tuple["np.ndarray", int]:
    """Decode any audio file to mono float32 at 16 kHz."""

    if np is None:
        raise RuntimeError("numpy is not installed")

    ffmpeg = _ffmpeg_executable()
    if ffmpeg:
        command = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            timeout=120,
        )
        if result.stdout:
            return (
                np.frombuffer(result.stdout, dtype=np.float32),
                16000,
            )

    # WAV fallback when FFmpeg is unavailable.
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((raw.shape[0], 4), dtype=np.uint8)
        padded[:, 1:] = raw
        data = padded.view("<i4").reshape(-1).astype(np.float32)
        data = data / 2147483648.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32)
        data = data / 2147483648.0
    else:
        raise RuntimeError(f"unsupported WAV sample width: {width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return data, rate


def inspect_audio(path, script_text: str = "") -> AudioHealth:
    """Check duration, loudness, silence and clipping for one narration clip."""

    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return AudioHealth(False, issues=["audio file is missing or empty"])

    if np is None:
        return AudioHealth(
            True,
            issues=["audio analysis skipped (numpy is not installed)"],
        )

    try:
        samples, sample_rate = _decode_samples(path)
    except Exception as exc:  # noqa: BLE001
        return AudioHealth(False, issues=[f"could not decode audio: {exc}"])

    if sample_rate <= 0 or len(samples) == 0:
        return AudioHealth(False, issues=["audio contains no samples"])

    duration = float(len(samples)) / float(sample_rate)
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
    clipping_ratio = (
        float(np.mean(np.abs(samples) >= 0.995)) if len(samples) else 0.0
    )

    words = len(re.findall(r"[A-Za-z0-9']+", script_text or ""))
    expected = max(0.6, words / 2.6) if words else 0.0

    issues: list[str] = []

    if duration < 0.35:
        issues.append("audio is too short")

    if expected:
        ratio = duration / expected
        if ratio < 0.45:
            issues.append(
                f"audio is much shorter than expected ({ratio:.2f}x)"
            )
        elif ratio > 2.6:
            issues.append(
                f"audio is much longer than expected ({ratio:.2f}x)"
            )

    if rms < 0.0025:
        issues.append("audio is near-silent")

    if clipping_ratio > 0.02:
        issues.append(f"audio is clipping ({clipping_ratio:.1%} of samples)")

    return AudioHealth(
        ok=not issues,
        duration=duration,
        expected_duration=expected,
        peak=peak,
        rms=rms,
        clipping_ratio=clipping_ratio,
        issues=issues,
    )
