"""Persistent usage-limit ledger for the parallel stage workers.

Gemini and the local TTS server are shared, account/process-wide resources.
When one stage hits a usage limit we must not let the other stages hammer the
same resource, and a later run must remember the cooldown instead of
immediately retrying.

The ledger is a tiny JSON file (``factory_state.json`` by default) with one
entry per resource::

    {
      "version": 1,
      "updated_at": "2026-11-07 12:00:00",
      "resources": {
        "gemini": {
          "blocked_until": 1762524000.0,
          "blocked_until_text": "2026-11-07 13:00:00",
          "last_hit_at": "2026-11-07 12:00:00",
          "cooldown_minutes": 60.0,
          "message": "You have reached your limit..."
        }
      }
    }

Epoch seconds make the readiness check trivial and immune to clock formats.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path


# ----------------------------------------------------------------------
# Limit-message parsing
# ----------------------------------------------------------------------
#
# Gemini and Claude both phrase their cooldowns loosely, for example:
#     "Your limit resets at 5:00 PM."
#     "You've reached your limit. Try again in 2 hours."
# When we can read the reset time we wait exactly that long; otherwise the
# caller's default is used.

_RESET_RE = re.compile(
    r"reset[s]?(?:\s+at)?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
)
_IN_RE = re.compile(
    r"\bin\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?)\b"
)


def parse_reset_minutes(message: str, *, default: float = 60.0) -> float:
    """Best-effort cooldown (minutes) parsed from a usage-limit message."""

    text = (message or "").lower()

    match = _RESET_RE.search(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3)
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if 0 <= hour < 24 and 0 <= minute < 60:
            now = datetime.now()
            target = now.replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )
            if target <= now:
                target += timedelta(days=1)
            return max(1.0, (target - now).total_seconds() / 60.0)

    match = _IN_RE.search(text)
    if match:
        value = float(match.group(1))
        return value * 60.0 if match.group(2).startswith("h") else value

    return default


class LimitLedger:
    """Read/write cooldowns for shared resources (gemini, tts, ...)."""

    def __init__(self, path, *, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self.data = self._load()

    # ------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------

    def _load(self) -> dict:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                loaded.setdefault("resources", {})
                return loaded
        except (FileNotFoundError, OSError, ValueError):
            pass
        return {"version": 1, "resources": {}}

    def reload(self) -> None:
        """Re-read the file from disk.

        ``main.py`` runs as a subprocess and writes the same ledger, so the
        long-lived batch process must refresh before trusting what it has in
        memory.
        """

        self.data = self._load()

    def _save(self) -> None:
        self.data["version"] = 1
        self.data["updated_at"] = self._format(self.clock())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    @staticmethod
    def _format(epoch: float) -> str:
        return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------

    def block(
        self,
        resource: str,
        cooldown_minutes: float,
        message: str = "",
    ) -> dict:
        """Record that ``resource`` is limited for ``cooldown_minutes``."""

        now = self.clock()
        cooldown_minutes = max(0.0, float(cooldown_minutes))
        entry = {
            "blocked_until": now + cooldown_minutes * 60.0,
            "blocked_until_text": self._format(
                now + cooldown_minutes * 60.0
            ),
            "last_hit_at": self._format(now),
            "cooldown_minutes": cooldown_minutes,
            "message": str(message or ""),
        }
        self.data.setdefault("resources", {})[resource] = entry
        self._save()
        return entry

    def clear(self, resource: str) -> None:
        resources = self.data.setdefault("resources", {})
        if resource in resources:
            resources.pop(resource, None)
            self._save()

    def remaining(self, resource: str) -> float:
        """Seconds until ``resource`` may be used again (0 when ready)."""

        entry = self.data.get("resources", {}).get(resource)
        if not entry:
            return 0.0
        try:
            blocked_until = float(entry.get("blocked_until", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, blocked_until - self.clock())

    def is_blocked(self, resource: str) -> bool:
        return self.remaining(resource) > 0.0

    def info(self, resource: str) -> dict:
        return dict(self.data.get("resources", {}).get(resource) or {})

    def describe(self, resource: str) -> str:
        entry = self.info(resource)
        remaining = self.remaining(resource)
        if remaining <= 0.0:
            return f"{resource}: ready"
        minutes = remaining / 60.0
        until = entry.get("blocked_until_text", "?")
        return f"{resource}: blocked for {minutes:.1f} min (until {until})"
