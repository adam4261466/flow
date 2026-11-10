"""A small blackboard for critic-loop decisions.

``production.json`` is intentionally separate from ``checkpoint.json``:
checkpoints describe which assets exist, while this file describes *what the
agents decided and why*.  It is safe to delete; the pipeline will recreate it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

MAX_LOG_ENTRIES = 300


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Budget:
    """Per-story spend counters used to stop runaway critic loops."""

    critic_calls: int = 0
    image_generations: int = 0
    tts_generations: int = 0
    critic_limit: int = 60

    def can_spend_critic(self, amount: int = 1) -> bool:
        return self.critic_calls + amount <= self.critic_limit

    def spend_critic(self, amount: int = 1) -> None:
        self.critic_calls += amount

    def spend_image(self, amount: int = 1) -> None:
        self.image_generations += amount

    def spend_tts(self, amount: int = 1) -> None:
        self.tts_generations += amount

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Budget":
        data = data or {}
        return cls(
            critic_calls=int(data.get("critic_calls", 0) or 0),
            image_generations=int(data.get("image_generations", 0) or 0),
            tts_generations=int(data.get("tts_generations", 0) or 0),
            critic_limit=int(data.get("critic_limit", 60) or 60),
        )


@dataclass
class ProductionState:
    """Story-scoped JSON blackboard shared by all critic loops."""

    story_dir: Path
    story: str
    data: dict = field(default_factory=dict)
    # Guards concurrent writes from the parallel stage workers.  Excluded from
    # repr/equality and from the serialized JSON.
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.story_dir = Path(self.story_dir)
        self.path = self.story_dir / "production.json"
        self.data.setdefault("story", self.story)
        self.data.setdefault("stage", "starting")
        self.data.setdefault("created_at", _now())
        self.data.setdefault("updated_at", _now())
        self.data.setdefault("artifacts", {})
        self.data.setdefault("decisions", [])
        self.data.setdefault("issues", [])
        self.data.setdefault("attempts", [])
        self.data.setdefault("agent_config", {})
        self.budget = Budget.from_dict(self.data.get("budget"))

    @classmethod
    def load_or_create(cls, story_dir, story: str) -> "ProductionState":
        story_dir = Path(story_dir)
        path = story_dir / "production.json"
        data = {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (FileNotFoundError, OSError, ValueError):
            data = {}
        return cls(story_dir=story_dir, story=story, data=data)

    def save(self) -> None:
        with self._lock:
            self.story_dir.mkdir(parents=True, exist_ok=True)
            self.data["story"] = self.story
            self.data["updated_at"] = _now()
            self.data["budget"] = self.budget.to_dict()

            for key in ("decisions", "issues", "attempts"):
                value = self.data.get(key)
                if isinstance(value, list) and len(value) > MAX_LOG_ENTRIES:
                    self.data[key] = value[-MAX_LOG_ENTRIES:]

            temporary = self.path.with_suffix(
                f".{os.getpid()}.{threading.get_ident()}.tmp"
            )
            temporary.write_text(
                json.dumps(self.data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            for attempt in range(5):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    # Windows briefly locks the destination when another
                    # process (e.g. a parallel stage) is reading it.
                    if attempt == 4:
                        temporary.unlink(missing_ok=True)
                        raise
                    time.sleep(0.05 * (attempt + 1))

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self.data["stage"] = stage

    def set_artifact(self, name: str, value) -> None:
        with self._lock:
            self.data.setdefault("artifacts", {})[name] = value

    def record_decision(
        self,
        agent: str,
        action: str,
        *,
        clip=None,
        reason: str | None = None,
        meta: dict | None = None,
    ) -> None:
        with self._lock:
            self.data.setdefault("decisions", []).append(
                {
                    "at": _now(),
                    "agent": agent,
                    "action": action,
                    "clip": clip,
                    "reason": reason,
                    "meta": meta or {},
                }
            )

    def record_issue(
        self,
        agent: str,
        message: str,
        *,
        clip=None,
        severity: str = "warning",
        meta: dict | None = None,
    ) -> None:
        with self._lock:
            self.data.setdefault("issues", []).append(
                {
                    "at": _now(),
                    "agent": agent,
                    "message": str(message),
                    "clip": clip,
                    "severity": severity,
                    "meta": meta or {},
                }
            )

    def record_attempt(
        self,
        kind: str,
        attempt: int,
        *,
        score: float | None = None,
        accepted: bool | None = None,
        issues: list[str] | None = None,
        clip=None,
    ) -> None:
        with self._lock:
            self.data.setdefault("attempts", []).append(
                {
                    "at": _now(),
                    "kind": kind,
                    "attempt": attempt,
                    "score": score,
                    "accepted": accepted,
                    "issues": list(issues or []),
                    "clip": clip,
                }
            )

    def summary(self) -> str:
        return (
            f"critic_calls={self.budget.critic_calls}/"
            f"{self.budget.critic_limit}, "
            f"images={self.budget.image_generations}, "
            f"tts={self.budget.tts_generations}, "
            f"issues={len(self.data.get('issues', []))}"
        )
