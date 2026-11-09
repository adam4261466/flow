"""Text normalisation and JSON extraction helpers for critics."""

from __future__ import annotations

import difflib
import json
import re
import unicodedata

_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
    "\u2026": "...",
}


def normalize_for_match(text: str | None) -> str:
    """Lowercase alphanumeric form used for exact title comparisons.

    Punctuation, curly quotes, line breaks and capitalisation are ignored so a
    correct title never fails because of typography.  Letter-level differences
    are preserved, which is exactly what catches a misspelled thumbnail.
    """

    if not text:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    for source, target in _REPLACEMENTS.items():
        value = value.replace(source, target)
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def title_matches(expected: str, actual: str) -> bool:
    """True only when the rendered title matches word for word."""

    left = normalize_for_match(expected)
    right = normalize_for_match(actual)
    return bool(left) and left == right


def similarity(expected: str, actual: str) -> float:
    """0..1 similarity, useful for logging near-misses."""

    left = normalize_for_match(expected)
    right = normalize_for_match(actual)
    if not left and not right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def clamp_score(value, default: float = 5.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(10.0, number))


def extract_json_object(text: str | None) -> dict | None:
    """Return the first JSON object in a model response, tolerating fences."""

    if not text:
        return None

    value = str(text).strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()

    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        pass

    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        parsed = json.loads(value[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None
