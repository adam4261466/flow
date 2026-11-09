"""Dependency-gated parallel stage workers.

The serial pipeline runs every stage to completion before starting the next
one: all images, then all voices, then every clip, then the final assembly.
That leaves the Gemini browser idle while TTS works and leaves FFmpeg idle
while images are still generating.

This module keeps the same artifacts and file names but lets four workers run
at once:

* ``images``  – Gemini images (and the thumbnail at the end)
* ``voices``  – local Chatterbox narration
* ``clips``   – one clip as soon as *that clip's* image and audio both exist
* ``final``   – intro/outro + concatenation once every clip exists

Each worker continuously re-checks its own readiness instead of waiting for a
global stage boundary.  A shared :class:`agents.limits.LimitLedger` records
usage-limit hits so the image worker parks itself until the cooldown expires,
while the other workers keep finishing whatever they can.

The orchestration is intentionally dependency-injected: ``generate_images``,
``generate_voices``, ``render_clip`` and ``assemble_final`` are callables
supplied by ``main.py``.  That keeps this module unit-testable without Chrome,
TTS, or FFmpeg.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from agents.limits import parse_reset_minutes


@dataclass
class StagePaths:
    """Where each stage writes its output."""

    images: Path
    audio: Path
    video: Path
    thumbnail: Path

    def image(self, clip_id: int) -> Path:
        return self.images / f"image_{int(clip_id):03d}.png"

    def audio_file(self, clip_id: int) -> Path:
        return self.audio / f"audio_{int(clip_id):03d}.wav"

    def clip(self, clip_id: int) -> Path:
        return self.video / f"clip_{int(clip_id):03d}.mp4"

    @property
    def final(self) -> Path:
        return self.video / "final_video.mp4"

    @property
    def thumbnail_file(self) -> Path:
        return self.thumbnail / "thumbnail.png"


@dataclass
class ParallelResult:
    complete: bool
    gemini_blocked: bool = False
    limit_message: str | None = None
    missing_images: list[int] = field(default_factory=list)
    missing_voices: list[int] = field(default_factory=list)
    missing_clips: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    waited_seconds: float = 0.0
    final_video: Path | None = None


def _present(path: Path, min_bytes: int = 1024) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


class ParallelPipeline:
    """Run the four stage workers for one story."""

    def __init__(
        self,
        *,
        clips,
        paths: StagePaths,
        plans,
        generate_images,
        generate_voices,
        render_clip,
        assemble_final,
        ledger,
        checkpoint=None,
        poll_seconds: float = 5.0,
        cooldown_minutes: float = 60.0,
        max_wait_minutes: float = 240.0,
        wait_for_limit: bool = True,
        settle_seconds: float = 1.5,
        max_clip_attempts: int = 2,
        max_stage_retries: int = 3,
        stage_retry_seconds: float = 30.0,
        log=print,
        clock=time.time,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.clips = list(clips)
        self.ids = [int(clip.id) for clip in self.clips]
        self.paths = paths
        self.plans = plans
        self.generate_images = generate_images
        self.generate_voices = generate_voices
        self.render_clip = render_clip
        self.assemble_final = assemble_final
        self.ledger = ledger
        self.checkpoint = checkpoint
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.cooldown_minutes = max(0.0, float(cooldown_minutes))
        self.max_wait_minutes = max(0.0, float(max_wait_minutes))
        self.wait_for_limit = bool(wait_for_limit)
        self.settle_seconds = max(0.0, float(settle_seconds))
        self.max_clip_attempts = max(1, int(max_clip_attempts))
        self.max_stage_retries = max(0, int(max_stage_retries))
        self.stage_retry_seconds = max(0.0, float(stage_retry_seconds))
        self.log = log
        self.clock = clock
        self.stop_event = stop_event or threading.Event()

        self._lock = threading.Lock()
        self._errors: list[str] = []
        self._image_done = False
        self._voice_done = False
        self._clip_done = threading.Event()
        self._final_done = threading.Event()
        self._gemini_blocked = False
        self._limit_message: str | None = None
        self._last_checkpoint_stage: str | None = None
        self._last_checkpoint_key: tuple | None = None
        self._started_at = self.clock()

    # ------------------------------------------------------------
    # SMALL HELPERS
    # ------------------------------------------------------------

    def _record_error(self, stage: str, error: Exception) -> None:
        message = f"{stage}: {error}"
        with self._lock:
            self._errors.append(message)
        self.log(f"[parallel] {message}")

    def _stable(self, path: Path, min_bytes: int = 1024) -> bool:
        """True when ``path`` looks complete and has stopped being written.

        Producers write straight to the final filename, so a freshly appeared
        file may still be growing.  Requiring a minimum age avoids handing a
        half-written PNG/WAV to FFmpeg.
        """

        if not _present(path, min_bytes):
            return False
        if self.settle_seconds <= 0:
            return True
        try:
            age = self.clock() - path.stat().st_mtime
        except OSError:
            return False
        return age >= self.settle_seconds

    def _clip_ready(self, clip_id: int) -> bool:
        min_audio = 4096
        return self._stable(
            self.paths.image(clip_id)
        ) and self._stable(self.paths.audio_file(clip_id), min_audio)

    def _all_clips_present(self) -> bool:
        return all(
            _present(self.paths.clip(clip_id), 1024)
            for clip_id in self.ids
        )

    def _deadline_passed(self) -> bool:
        if self.max_wait_minutes <= 0:
            return False
        return (
            self.clock() - self._started_at
            >= self.max_wait_minutes * 60.0
        )

    def _safe_checkpoint(self, stage: str) -> None:
        key = (stage, self._gemini_blocked, bool(self._limit_message))
        if self.checkpoint is None or key == self._last_checkpoint_key:
            return
        self._last_checkpoint_key = key
        self._last_checkpoint_stage = stage
        try:
            self.checkpoint(stage, self._gemini_blocked, self._limit_message)
        except Exception as error:  # checkpointing must never kill the run
            self._record_error("checkpoint", error)

    def _current_stage(self) -> str:
        if _present(self.paths.final, 1024):
            return "complete"
        if self.paths.video.is_dir() and any(
            self.paths.video.glob("clip_*.mp4")
        ) and self._all_clips_present():
            return "final"
        if self.paths.video.is_dir() and any(
            self.paths.video.glob("clip_*.mp4")
        ):
            return "clips"
        if self.paths.audio.is_dir() and any(
            self.paths.audio.glob("audio_*.wav")
        ) or self.paths.images.is_dir() and any(
            self.paths.images.glob("image_*.png")
        ):
            return "media"
        return "starting"

    def _all_images_present(self) -> bool:
        return all(
            _present(self.paths.image(clip_id))
            for clip_id in self.ids
        )

    def _all_voices_present(self) -> bool:
        return all(
            _present(self.paths.audio_file(clip_id), 4096)
            for clip_id in self.ids
        )

    def _media_complete(self) -> bool:
        return (
            self._all_images_present()
            and _present(self.paths.thumbnail_file)
            and self._all_voices_present()
        )

    def _stage_retry_wait(self, failures: int) -> float:
        """Back off between stage retries, capped so resumes stay responsive."""
        delay = self.stage_retry_seconds * max(1, failures)
        return min(delay, 300.0)

    def _retry_stage(self, stage: str, failures: int, error: Exception) -> bool:
        """Record a stage error and return True when another attempt is allowed.

        A transient Chrome/TTS failure used to end the stage (and therefore the
        whole story) immediately, leaving finished images/narration without any
        clips.  Retry a few times so one hiccup no longer strands a render.
        """
        self._record_error(stage, error)
        if failures > self.max_stage_retries or self._deadline_passed():
            self.log(
                f"[parallel] {stage} worker: giving up after "
                f"{failures} attempt(s)"
            )
            return False
        wait = self._stage_retry_wait(failures)
        self.log(
            f"[parallel] {stage} worker: transient error; retrying in "
            f"{wait:.0f}s"
        )
        self.stop_event.wait(wait)
        return True

    # ------------------------------------------------------------
    # WORKERS
    # ------------------------------------------------------------

    def _image_worker(self) -> None:
        """Generate missing images + thumbnail, honouring limiter cooldowns."""

        try:
            if (
                self._all_images_present()
                and _present(self.paths.thumbnail_file)
            ):
                self.log(
                    "[parallel] image worker: all images and thumbnail exist"
                )
                return

            waiting_logged = False
            failures = 0
            while not self.stop_event.is_set():
                remaining = self.ledger.remaining("gemini")
                if remaining > 0.0:
                    if not self.wait_for_limit or self._deadline_passed():
                        with self._lock:
                            self._gemini_blocked = True
                        self.log(
                            "[parallel] images paused: "
                            + self.ledger.describe("gemini")
                        )
                        return
                    if not waiting_logged:
                        until = self.ledger.info("gemini").get(
                            "blocked_until_text", "?"
                        )
                        self.log(
                            f"[parallel] Gemini limit reached. Images pause "
                            f"until {until}. This message will not repeat."
                        )
                        waiting_logged = True
                    # Keep waking in small chunks so Ctrl+C stays responsive,
                    # but stay quiet until the cooldown actually clears.
                    wait = min(max(self.poll_seconds, 5.0), remaining)
                    self.stop_event.wait(wait)
                    continue

                if waiting_logged:
                    self.log(
                        "[parallel] Gemini limit cleared; resuming images."
                    )
                    waiting_logged = False

                self.log("[parallel] image worker: generating missing images")
                try:
                    blocked, message = self.generate_images()
                except Exception as error:  # noqa: BLE001 - retry transient errors
                    failures += 1
                    if not self._retry_stage("images", failures, error):
                        return
                    continue

                if blocked:
                    with self._lock:
                        self._gemini_blocked = True
                        self._limit_message = message
                    # Gemini's banner often contains the real reset time. Use
                    # it so the factory waits exactly as long as needed
                    # instead of retrying (and printing again) every hour.
                    cooldown = parse_reset_minutes(
                        message or "",
                        default=self.cooldown_minutes,
                    )
                    self.ledger.block(
                        "gemini",
                        cooldown,
                        message or "Gemini usage limit reached",
                    )
                    self.log("[parallel] Gemini limit hit; cooldown saved.")
                    if not self.wait_for_limit:
                        return
                    if self.cooldown_minutes <= 0:
                        # Avoid a tight retry loop when the cooldown is zero.
                        self.stop_event.wait(self.poll_seconds)
                    continue

                failures = 0
                if (
                    self._all_images_present()
                    and _present(self.paths.thumbnail_file)
                ):
                    self.ledger.clear("gemini")
                    return

                # Generation finished without a limit banner but some files are
                # still missing (a single failed clip, Chrome hiccup, ...).
                failures += 1
                if not self._retry_stage(
                    "images",
                    failures,
                    RuntimeError("images still missing after generation"),
                ):
                    return
                continue
        finally:
            with self._lock:
                self._image_done = True
            self.log("[parallel] image worker finished")

    def _voice_worker(self) -> None:
        try:
            if self._all_voices_present():
                self.log("[parallel] voice worker: all narration exists")
                return
            self.log("[parallel] voice worker: generating missing narration")

            failures = 0
            while not self.stop_event.is_set():
                try:
                    self.generate_voices()
                except Exception as error:  # noqa: BLE001 - retry TTS failures
                    failures += 1
                    if not self._retry_stage("voices", failures, error):
                        return
                    continue

                if self._all_voices_present():
                    return

                # The loop returned without raising but narration is still
                # missing (for example the TTS server dropped one clip).
                failures += 1
                if not self._retry_stage(
                    "voices",
                    failures,
                    RuntimeError("narration still missing after generation"),
                ):
                    return
        finally:
            with self._lock:
                self._voice_done = True
            self.log("[parallel] voice worker finished")

    def _clip_worker(self) -> None:
        """Render each clip the moment its own image and audio are ready."""

        attempted: set[int] = set()
        failures: dict[int, int] = {}
        try:
            while not self.stop_event.is_set():
                rendered_any = False

                for clip_id in self.ids:
                    if clip_id in attempted:
                        continue
                    if not self._clip_ready(clip_id):
                        continue

                    try:
                        self.render_clip(
                            clip_id,
                            self.plans.get(clip_id),
                        )
                    except FileNotFoundError:
                        # A producer may still be settling the file.
                        continue
                    except Exception as error:  # noqa: BLE001
                        count = failures.get(clip_id, 0) + 1
                        failures[clip_id] = count
                        self._record_error(
                            f"clip {clip_id}", error
                        )
                        if count >= self.max_clip_attempts:
                            attempted.add(clip_id)
                        else:
                            self.stop_event.wait(
                                min(self.poll_seconds, 1.0)
                            )
                        continue

                    if not _present(self.paths.clip(clip_id), 1024):
                        count = failures.get(clip_id, 0) + 1
                        failures[clip_id] = count
                        self._record_error(
                            f"clip {clip_id}",
                            RuntimeError("renderer produced no output"),
                        )
                        if count >= self.max_clip_attempts:
                            attempted.add(clip_id)
                        continue

                    attempted.add(clip_id)
                    rendered_any = True
                    self._safe_checkpoint("clips")
                    break

                if rendered_any:
                    continue

                with self._lock:
                    producers_done = self._image_done and self._voice_done

                if producers_done:
                    ready = [
                        clip_id
                        for clip_id in self.ids
                        if self._clip_ready(clip_id)
                        and clip_id not in attempted
                    ]
                    if not ready:
                        return

                self.stop_event.wait(self.poll_seconds)
        finally:
            self._clip_done.set()
            self.log("[parallel] clip worker finished")

    def _final_worker(self) -> None:
        """Assemble the final video once every clip is on disk."""

        try:
            while not self.stop_event.is_set():
                if (
                    self._clip_done.is_set()
                    and self._all_clips_present()
                    and all(
                        self._stable(self.paths.clip(clip_id), 1024)
                        for clip_id in self.ids
                    )
                ):
                    try:
                        self.log("[parallel] final worker: assembling video")
                        self.assemble_final()
                    except Exception as error:  # noqa: BLE001
                        self._record_error("final", error)
                        return

                    self._safe_checkpoint("complete")
                    self._final_done.set()
                    return

                if self._clip_done.is_set() and not self._all_clips_present():
                    return

                self.stop_event.wait(self.poll_seconds)
        finally:
            self.log("[parallel] final worker finished")

    # ------------------------------------------------------------
    # COORDINATOR
    # ------------------------------------------------------------

    def run(self) -> ParallelResult:
        self.log("[parallel] starting stage workers")
        self._safe_checkpoint("starting")

        threads = [
            threading.Thread(
                target=self._image_worker,
                name="stage-images",
                daemon=True,
            ),
            threading.Thread(
                target=self._voice_worker,
                name="stage-voices",
                daemon=True,
            ),
            threading.Thread(
                target=self._clip_worker,
                name="stage-clips",
                daemon=True,
            ),
            threading.Thread(
                target=self._final_worker,
                name="stage-final",
                daemon=True,
            ),
        ]

        for thread in threads:
            thread.start()

        # Keep refreshing the checkpoint while the workers run, and stop the
        # whole graph cleanly if an unexpected caller-side stop is requested.
        try:
            while any(thread.is_alive() for thread in threads):
                self._safe_checkpoint(self._current_stage())
                self.stop_event.wait(min(self.poll_seconds, 1.0))
        except KeyboardInterrupt:
            self.log("[parallel] interrupted; stopping workers")
            self.stop_event.set()

        for thread in threads:
            thread.join(timeout=30)

        missing_images = [
            clip_id
            for clip_id in self.ids
            if not _present(self.paths.image(clip_id))
        ]
        missing_voices = [
            clip_id
            for clip_id in self.ids
            if not _present(self.paths.audio_file(clip_id), 4096)
        ]
        missing_clips = [
            clip_id
            for clip_id in self.ids
            if not _present(self.paths.clip(clip_id))
        ]
        thumbnail_ready = _present(self.paths.thumbnail_file)

        complete = (
            _present(self.paths.final, 1024)
            and thumbnail_ready
            and not missing_images
            and not missing_voices
            and not missing_clips
        )
        final_video = self.paths.final if complete else None
        result = ParallelResult(
            complete=complete,
            gemini_blocked=self._gemini_blocked,
            limit_message=self._limit_message,
            missing_images=missing_images,
            missing_voices=missing_voices,
            missing_clips=missing_clips,
            errors=list(self._errors),
            waited_seconds=max(0.0, self.clock() - self._started_at),
            final_video=final_video,
        )

        if complete:
            self._safe_checkpoint("complete")
        self.log(
            "[parallel] done: "
            + ("complete" if complete else "incomplete")
            + f" in {result.waited_seconds / 60.0:.1f} min"
        )
        return result
