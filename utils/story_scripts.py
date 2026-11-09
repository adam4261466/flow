"""Allocate the next continuous script number and story folder name.

The project's existing naming convention is the source of truth:

    input/script1.txt, input/script2.txt, ...
    output/story_001, output/story_002, ...

A generated story therefore becomes ``input/script{N}.txt`` and
``output/story_{N:03d}`` where ``N`` is one greater than the highest number
found in either location.
"""

from __future__ import annotations

import re
from pathlib import Path


SCRIPT_PATTERN = re.compile(r"^script(\d+)\.txt$", re.IGNORECASE)
STORY_PATTERN = re.compile(r"^story_(\d+)$", re.IGNORECASE)


def script_numbers(input_dir: str | Path) -> set[int]:
    """Return the numbers already used by ``input/scriptN.txt`` files."""
    directory = Path(input_dir)
    if not directory.is_dir():
        return set()

    numbers: set[int] = set()
    for path in directory.iterdir():
        match = SCRIPT_PATTERN.match(path.name)
        if match:
            numbers.add(int(match.group(1)))
    return numbers


def story_numbers(output_dir: str | Path) -> set[int]:
    """Return the numbers already used by ``output/story_NNN`` folders."""
    directory = Path(output_dir)
    if not directory.is_dir():
        return set()

    numbers: set[int] = set()
    for path in directory.iterdir():
        if not path.is_dir():
            continue
        match = STORY_PATTERN.match(path.name)
        if match:
            numbers.add(int(match.group(1)))
    return numbers


def existing_numbers(
    input_dir: str | Path,
    output_dir: str | Path,
) -> set[int]:
    return script_numbers(input_dir) | story_numbers(output_dir)


def next_number(
    input_dir: str | Path,
    output_dir: str | Path,
) -> int:
    """One greater than the highest script/story number in use (minimum 1)."""
    used = existing_numbers(input_dir, output_dir)
    return max(used, default=0) + 1


def story_name(number: int) -> str:
    return f"story_{int(number):03d}"


def script_name(number: int) -> str:
    return f"script{int(number)}.txt"


def next_script_path(
    input_dir: str | Path,
    output_dir: str | Path,
) -> tuple[int, Path, str]:
    """Return ``(number, script_path, story_folder_name)`` for a new story.

    Existing files/folders are never overwritten.  If a partially-used number
    is somehow present, the next free number is chosen.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    number = next_number(input_dir, output_dir)

    while True:
        candidate_script = input_dir / script_name(number)
        candidate_story = output_dir / story_name(number)
        if not candidate_script.exists() and not candidate_story.exists():
            return number, candidate_script, story_name(number)
        number += 1


def save_generated_script(
    text: str,
    input_dir: str | Path,
    output_dir: str | Path,
) -> tuple[Path, str]:
    """Write a newly generated script and return ``(path, story_name)``."""
    if not text or not text.strip():
        raise ValueError("Cannot save an empty generated script.")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    number, script_path, folder_name = next_script_path(input_dir, output_dir)
    if script_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing script: {script_path}")

    script_path.write_text(text.strip() + "\n", encoding="utf-8")
    return script_path, folder_name
