"""Console encoding helpers for Windows.

Windows consoles and redirected log files often still use a legacy code page
(cp1252/cp437).  Page text scraped from YouTube Studio, Gemini and Claude can
contain characters that code page cannot encode, for example:

* ``U+202F`` narrow no-break space (used in Studio's localized date/time text)
* ``U+E056`` a private-use icon glyph returned in Gemini responses

A plain ``print()`` of that text raised ``UnicodeEncodeError`` and, because the
prints sit inside upload/QC flows, it aborted a real YouTube upload and marked
good generated images as failed.  ``setup_console`` makes stdout/stderr
UTF-8 with ``errors="replace"`` so logging can never break the pipeline.
"""

from __future__ import annotations

import os
import sys
from typing import IO


def _reconfigure(stream: IO[str] | None) -> None:
    if stream is None:
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        # Detached/closed or exotic stream; printing will use its own rules.
        pass


def setup_console() -> None:
    """Make console/log output UTF-8 and tolerant of unencodable characters."""
    _reconfigure(sys.stdout)
    _reconfigure(sys.stderr)
    # Inherited by every Python child process started later (main.py renders,
    # uploader runs, ...).  ``setdefault`` keeps an explicit user setting.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
