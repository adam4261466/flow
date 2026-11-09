"""Status tracking for the batch factory.

The GUI polls this file to show progress without having to parse logs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

STATUS_FILE = Path(__file__).resolve().parent / "factory_status.json"


def update_status(data: dict[str, Any]) -> None:
    """Write a status snapshot to disk."""
    try:
        data["updated_at"] = datetime.now().isoformat(sep=" ", timespec="seconds")
        STATUS_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), 
            encoding="utf-8"
        )
    except Exception:
        pass


def read_status() -> dict[str, Any]:
    """Read the latest status snapshot."""
    try:
        if STATUS_FILE.exists():
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}
