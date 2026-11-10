"""Cross-story, stage-parallel scheduling for the video factory.

The old batch loop rendered one story at a time:

    script A -> images A -> voices A -> clips A -> final A
    script B -> images B -> ...

That leaves the Gemini browser idle while TTS works and leaves FFmpeg idle
while Gemini works.  This scheduler keeps one worker per production stage, each
continuously claiming the next *eligible story*:

    script worker  : story A, then story B, then story C, ...
    image worker   : story A images, then story B images, ...
    voice worker   : story A voices, then story B voices, ...
    clip worker(s) : any story whose images + voices are done
    final worker   : any story whose clips are all done

A story never has to finish before an unrelated stage moves on.  The only
ordering is the real dependency graph: ``script -> images/voices -> clips ->
final``.  Completion is detected from the artifacts on disk, so a restart
resumes the same folders without a fragile in-memory-only state.

The module is deliberately free of Chrome/TTS/FFmpeg imports.  ``batch_generator``
injects the actual stage runners and a :class:`StageLayout`; tests can inject
fakes and temporary folders.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

STAGE_ORDER = ("script", "images", "voices", "clips", "final")

# A stage may only start when these stages are complete for the same story.
DEFAULT_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "script": (),
    "images": ("script",),
    "voices": ("script",),
    "clips": ("images", "voices"),
    "final": ("clips",),
}


@dataclass
class StoryJob:
    """One queued prompt and the artifacts produced for it."""

    prompt: str
    story_id: str
    script: Path
    clip_ids: list[int] = field(default_factory=list)
    attempts: int = 0
    stage_attempts: dict[str, int] = field(default_factory=dict)
    retry_at: dict[str, float] = field(default_factory=dict)
    failed: bool = False
    completed: bool = False
    last_error: str = ""


@dataclass
class StageResult:
    """Outcome of one stage run for one story."""

    ok: bool = False
    blocked: bool = False
    fatal: bool = False
    message: str = ""

    @classmethod
    def success(cls) -> "StageResult":
        return cls(ok=True)

    @classmethod
    def blocked_stage(cls, message: str = "") -> "StageResult":
        return cls(blocked=True, message=message)

    @classmethod
    def failed(cls, message: str = "", *, fatal: bool = False) -> "StageResult":
        return cls(ok=False, fatal=fatal, message=message)


@dataclass
class StageLayout:
    """Where each stage writes its files (per story id)."""

    batch_input: Path
    output_root: Path
    outbox_root: Path

    def story_output(self, story_id: str) -> Path:
        return self.output_root / story_id

    def image(self, story_id: str, clip_id: int) -> Path:
        return self.story_output(story_id) / "images" / f"image_{int(clip_id):03d}.png"

    def audio(self, story_id: str, clip_id: int) -> Path:
        return self.story_output(story_id) / "audio" / f"audio_{int(clip_id):03d}.wav"

    def clip(self, story_id: str, clip_id: int) -> Path:
        return self.story_output(story_id) / "video" / f"clip_{int(clip_id):03d}.mp4"

    @property
    def thumbnail_name(self) -> str:
        return "thumbnail.png"

    def thumbnail(self, story_id: str) -> Path:
        return self.story_output(story_id) / "thumbnail" / self.thumbnail_name

    def final(self, story_id: str) -> Path:
        return self.story_output(story_id) / "video" / "final_video.mp4"

    def outbox_video(self, story_id: str) -> Path:
        return self.outbox_root / story_id / "video.mp4"


def _present(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def read_clip_ids(script: Path) -> list[int]:
    """Return the clip ids declared by a saved script (empty when unreadable)."""

    try:
        text = Path(script).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    try:
        from parser.script_parser import parse_script

        return [int(clip.id) for clip in parse_script(text)]
    except Exception:  # a malformed script must not kill the scheduler
        return []


class StageScheduler:
    """Run one worker per stage over a shared list of :class:`StoryJob`."""

    def __init__(
        self,
        jobs: list[StoryJob],
        *,
        layout: StageLayout,
        runners: dict[str, Callable[[StoryJob], StageResult]],
        prerequisites: dict[str, tuple[str, ...]] | None = None,
        clip_workers: int = 1,
        max_stage_attempts: int = 3,
        poll_seconds: float = 5.0,
        retry_seconds: float = 30.0,
        on_complete: Callable[[StoryJob], None] | None = None,
        on_failed: Callable[[StoryJob], None] | None = None,
        on_status: Callable[[dict], None] | None = None,
        parse_clip_ids: Callable[[Path], list[int]] | None = None,
        log: Callable[[str], None] = print,
        stop_event: threading.Event | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.jobs = list(jobs)
        self.layout = layout
        self.runners = dict(runners)
        self.prerequisites = prerequisites or DEFAULT_PREREQUISITES
        self.clip_workers = max(1, int(clip_workers))
        self.max_stage_attempts = max(0, int(max_stage_attempts))
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.retry_seconds = max(0.0, float(retry_seconds))
        self.on_complete = on_complete
        self.on_failed = on_failed
        self.on_status = on_status
        self.parse_clip_ids = parse_clip_ids or read_clip_ids
        self.log = log
        self.stop_event = stop_event or threading.Event()
        self.clock = clock

        self._lock = threading.RLock()
        self._claims: dict[str, set[str]] = {stage: set() for stage in STAGE_ORDER}
        self._active: dict[str, str] = {}
        self._deliver_lock = threading.Lock()
        self._delivering: set[str] = set()

    # ------------------------------------------------------------------
    # FILESYSTEM PROBES
    # ------------------------------------------------------------------

    def clip_ids(self, job: StoryJob) -> list[int]:
        if not job.clip_ids and _present(job.script):
            job.clip_ids = self.parse_clip_ids(job.script)
        return job.clip_ids

    def stage_done(self, stage: str, job: StoryJob) -> bool:
        ids = self.clip_ids(job)

        if stage == "script":
            return _present(job.script)

        if stage in ("images", "voices", "clips") and not ids:
            return False

        if stage == "images":
            return all(
                _present(self.layout.image(job.story_id, clip_id))
                for clip_id in ids
            ) and _present(self.layout.thumbnail(job.story_id))

        if stage == "voices":
            return all(
                _present(self.layout.audio(job.story_id, clip_id))
                for clip_id in ids
            )

        if stage == "clips":
            return all(
                _present(self.layout.clip(job.story_id, clip_id))
                for clip_id in ids
            )

        if stage == "final":
            return _present(self.layout.final(job.story_id))

        return False

    def _prerequisites_done(self, stage: str, job: StoryJob) -> bool:
        return all(
            self.stage_done(dependency, job)
            for dependency in self.prerequisites.get(stage, ())
        )

    def _stage_enabled(self, stage: str) -> bool:
        return stage in self.runners

    def _stage_reachable(self, stage: str, job: StoryJob, seen: set[str] | None = None) -> bool:
        """True when this stage is already done or can eventually run."""
        if self.stage_done(stage, job):
            return True
        if not self._stage_enabled(stage):
            return False
        seen = set() if seen is None else seen
        if stage in seen:  # defensive: never recurse forever on a bad graph
            return False
        seen.add(stage)
        return all(
            self._stage_reachable(dependency, job, seen)
            for dependency in self.prerequisites.get(stage, ())
        )

    def _work_left(self, job: StoryJob) -> bool:
        """True when an enabled stage could still produce work for this job."""
        if job.completed or job.failed:
            return False
        return any(
            not self.stage_done(stage, job) and self._stage_reachable(stage, job)
            for stage in self.runners
        )

    # ------------------------------------------------------------------
    # CLAIMING
    # ------------------------------------------------------------------

    def _claim(self, stage: str) -> StoryJob | None:
        now = self.clock()
        with self._lock:
            busy = set()
            for other_stage, claims in self._claims.items():
                if other_stage != stage:
                    busy.update(claims)

            for job in self.jobs:
                if job.completed or job.failed:
                    continue
                if job.prompt in self._claims[stage]:
                    continue
                # Do not run two stages on the same story folder at once.
                # Another story usually has work for this stage, so prefer it
                # instead of waiting on the story a different worker owns.
                if job.prompt in busy:
                    continue
                if now < job.retry_at.get(stage, 0.0):
                    continue
                if self.stage_done(stage, job):
                    continue
                if not self._prerequisites_done(stage, job):
                    continue

                self._claims[stage].add(job.prompt)
                self._active[stage] = job.story_id
                return job
        return None

    def _release(self, stage: str, job: StoryJob) -> None:
        with self._lock:
            self._claims[stage].discard(job.prompt)
            if self._active.get(stage) == job.story_id:
                self._active.pop(stage, None)

    # ------------------------------------------------------------------
    # COMPLETION / FAILURE
    # ------------------------------------------------------------------

    def _deliver(self, job: StoryJob) -> bool:
        """Mark a job complete once its final video exists."""
        with self._deliver_lock:
            if job.completed or job.failed or job.story_id in self._delivering:
                return False
            if not self.stage_done("final", job):
                return False
            self._delivering.add(job.story_id)

        try:
            if self.on_complete is not None:
                self.on_complete(job)
            job.completed = True
        except Exception as error:  # noqa: BLE001 - delivery must not kill workers
            self.log(f"[stages] delivery failed for {job.story_id}: {error}")
            job.completed = False
            job.failed = True
            job.last_error = str(error)
            if self.on_failed is not None:
                self.on_failed(job)
            return False
        finally:
            with self._deliver_lock:
                self._delivering.discard(job.story_id)

        self.log(f"[stages] {job.story_id} complete")
        return True

    def _deliver_ready(self) -> int:
        delivered = 0
        for job in self.jobs:
            if self._deliver(job):
                delivered += 1
        return delivered

    def _finish_failed(self, stage: str, job: StoryJob, result: StageResult) -> None:
        job.failed = True
        job.last_error = result.message or f"{stage} failed"
        self.log(f"[stages] {job.story_id} skipped after {stage}: {job.last_error}")
        try:
            if self.on_failed is not None:
                self.on_failed(job)
        except Exception as error:  # noqa: BLE001
            self.log(f"[stages] could not record failure for {job.story_id}: {error}")

    def _is_finished(self) -> bool:
        with self._lock:
            return not any(self._work_left(job) for job in self.jobs)

    # ------------------------------------------------------------------
    # WORKER
    # ------------------------------------------------------------------

    def _worker(self, stage: str) -> None:
        while not self.stop_event.is_set():
            if self._deliver_ready():
                self._publish_status()
                continue

            job = self._claim(stage)
            if job is None:
                if self._is_finished():
                    return
                self.stop_event.wait(self.poll_seconds)
                continue

            self._publish_status()
            try:
                try:
                    result = self.runners[stage](job)
                except Exception as error:  # noqa: BLE001 - retry, never kill the batch
                    result = StageResult.failed(f"{type(error).__name__}: {error}")

                if result.ok or self.stage_done(stage, job):
                    job.stage_attempts[stage] = 0
                    job.retry_at[stage] = 0.0
                    if stage == "final":
                        self._deliver(job)
                    continue

                if result.blocked:
                    # A shared resource is cooling down. Do not count this as
                    # a story failure; another stage can keep working.
                    job.retry_at[stage] = self.clock() + self.poll_seconds
                    continue

                count = job.stage_attempts.get(stage, 0) + 1
                job.stage_attempts[stage] = count
                job.last_error = result.message or f"{stage} failed"
                self.log(
                    f"[stages] {job.story_id} {stage} failed "
                    f"({count}/{self.max_stage_attempts}): {job.last_error}"
                )

                if result.fatal or count > self.max_stage_attempts:
                    self._finish_failed(stage, job, result)
                else:
                    job.retry_at[stage] = (
                        self.clock() + self.retry_seconds * max(1, count)
                    )
            finally:
                self._release(stage, job)
                self._publish_status()

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            stages = {}
            for stage in STAGE_ORDER:
                done = sum(1 for job in self.jobs if self.stage_done(stage, job))
                stages[stage] = {
                    "active": [
                        job.story_id
                        for job in self.jobs
                        if job.prompt in self._claims[stage]
                    ],
                    "done": done,
                    "total": len(self.jobs),
                }
            active_stage = next(
                (stage for stage in STAGE_ORDER if self._active.get(stage)),
                "",
            )
            return {
                "mode": "stage-parallel",
                "stage": active_stage,
                "current_story": self._active.get(active_stage, ""),
                "stages": stages,
                "pending": sum(
                    1 for job in self.jobs if not (job.completed or job.failed)
                ),
                "completed": sum(1 for job in self.jobs if job.completed),
                "failed": sum(1 for job in self.jobs if job.failed),
                "total": len(self.jobs),
            }

    def _publish_status(self) -> None:
        if self.on_status is None:
            return
        try:
            self.on_status(self.status())
        except Exception:  # status must never stop production
            pass

    # ------------------------------------------------------------------
    # COORDINATOR
    # ------------------------------------------------------------------

    def run(self) -> dict:
        with self._lock:
            self.jobs = [job for job in self.jobs if not job.completed]
            for job in self.jobs:
                self.clip_ids(job)

        self.log(
            f"[stages] starting stage-parallel factory for {len(self.jobs)} "
            "stories"
        )
        self._publish_status()

        counts = {
            "script": 1,
            "images": 1,
            "voices": 1,
            "clips": self.clip_workers,
            "final": 1,
        }
        threads: list[threading.Thread] = []
        for stage, count in counts.items():
            if stage not in self.runners:
                continue
            for index in range(count):
                thread = threading.Thread(
                    target=self._worker,
                    args=(stage,),
                    name=f"stage-{stage}-{index}",
                    daemon=True,
                )
                threads.append(thread)
                thread.start()

        try:
            while any(thread.is_alive() for thread in threads):
                self._deliver_ready()
                self._publish_status()
                self.stop_event.wait(self.poll_seconds)
        except KeyboardInterrupt:
            self.log("[stages] interrupted; stopping workers")
            self.stop_event.set()

        for thread in threads:
            thread.join(timeout=30)

        self._deliver_ready()
        summary = self.status()
        self.log(
            "[stages] done: "
            f"{summary['completed']} complete, {summary['failed']} failed, "
            f"{summary['pending']} still pending"
        )
        return summary


__all__ = [
    "DEFAULT_PREREQUISITES",
    "STAGE_ORDER",
    "StageLayout",
    "StageResult",
    "StageScheduler",
    "StoryJob",
    "read_clip_ids",
]
