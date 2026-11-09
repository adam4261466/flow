"""Persistent batch queue with stable per-prompt story folders.

The queue lives in ``batch_state.json`` next to the project.  Besides the
``pending``/``completed`` lists this file now remembers, for every prompt:

* the story folder (``story_<timestamp>``) it was assigned to, so restarting
  the machine resumes the *same* half-finished folder instead of starting a
  brand new one;
* how many render attempts that prompt has used, so a permanently broken
  story cannot loop forever;
* the script file Claude produced, so a restart does not rewrite the story.

The on-disk format stays backward compatible: older files that only contain
``completed``/``pending`` load without changes.
"""

from __future__ import annotations

import json
import os


class BatchStateManager:
    def __init__(self, state_file="batch_state.json"):
        self.state_file = state_file
        self.state = self._load()

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------

    def _load(self):
        data = {}
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}

        if not isinstance(data, dict):
            data = {}

        data.setdefault("completed", [])
        data.setdefault("pending", [])
        data.setdefault("stories", {})
        data.setdefault("settings", {})
        return data

    def save(self):
        tmp = f"{self.state_file}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=4)
        os.replace(tmp, self.state_file)

    # ------------------------------------------------------------------
    # Prompt queue
    # ------------------------------------------------------------------

    def add_prompts(self, prompts):
        for p in prompts:
            if p not in self.state["completed"] and p not in self.state["pending"]:
                self.state["pending"].append(p)
        self.save()

    def get_next(self):
        if not self.state["pending"]:
            return None
        return self.state["pending"][0]

    def mark_complete(self, prompt):
        if prompt in self.state["pending"]:
            self.state["pending"].remove(prompt)
        if prompt not in self.state["completed"]:
            self.state["completed"].append(prompt)
        self.save()

    def requeue(self, prompt):
        """Move a failed prompt to the back so other stories can progress."""
        if prompt in self.state["pending"]:
            self.state["pending"].remove(prompt)
            self.state["pending"].append(prompt)
        self.save()

    # ------------------------------------------------------------------
    # Per-prompt story folders
    # ------------------------------------------------------------------

    def get_story(self, prompt):
        """Return the saved story info for a prompt (empty dict if unknown)."""
        return dict(self.state["stories"].get(prompt) or {})

    def set_story(self, prompt, story_id, script_path=None):
        info = self.state["stories"].setdefault(prompt, {})
        info["story_id"] = story_id
        if script_path is not None:
            info["script"] = str(script_path)
        info.setdefault("attempts", 0)
        self.save()
        return info

    def bump_attempt(self, prompt):
        info = self.state["stories"].setdefault(prompt, {})
        info["attempts"] = int(info.get("attempts", 0)) + 1
        self.save()
        return info["attempts"]

    # ------------------------------------------------------------------
    # Batch settings (survive restarts)
    # ------------------------------------------------------------------

    def get_setting(self, name, default=None):
        return self.state["settings"].get(name, default)

    def set_setting(self, name, value):
        self.state["settings"][name] = value
        self.save()
