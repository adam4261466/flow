"""Tests for the TITLE/DESCRIPTION Claude writes with each story script."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.claude_browser import (  # noqa: E402
    KIDS_STORY_PROMPT,
    STORY_PROMPT,
    clean_claude_script,
)
from parser.script_parser import (  # noqa: E402
    extract_description,
    extract_title,
    parse_script,
)


SCRIPT = """GLOBAL

TITLE:
Pip and the Moon Lantern

DESCRIPTION:
A tiny firefly is afraid of the dark.
He finds the courage to help a lost bunny get home.

Characters:
Pip the firefly

Style:
soft watercolor storybook

$$

CLIP
Characters:
Pip the firefly
Moment:
Pip sits on a leaf, afraid of the dark forest.
Vibe:
gentle and quiet
Style:
soft watercolor
Script:
Hello little one. Once upon a time a tiny firefly named Pip was afraid of the dark.

$$

CLIP
Characters:
Pip, the Moon
Moment:
The Moon smiles and asks Pip for help.
Vibe:
warm
Style:
soft watercolor
Script:
The Moon whispered, Pip, will you help me light the path?

$$

CLIP
Characters:
Pip, the bunny
Moment:
Pip lights the path and the bunny gets home.
Vibe:
happy and sleepy
Style:
soft watercolor
Script:
Pip glowed bright and warm, and the bunny hopped safely home to bed.
"""


class ScriptMetadataTests(unittest.TestCase):
    def test_extracts_title_and_description(self):
        self.assertEqual(extract_title(SCRIPT), "Pip and the Moon Lantern")
        self.assertEqual(
            extract_description(SCRIPT),
            "A tiny firefly is afraid of the dark.\n"
            "He finds the courage to help a lost bunny get home.",
        )

    def test_metadata_does_not_break_clip_parsing(self):
        clips = parse_script(SCRIPT)

        self.assertEqual(len(clips), 3)
        self.assertEqual(clips[0].id, 1)

    def test_clean_claude_script_keeps_metadata(self):
        cleaned = clean_claude_script("Here is the script:\n```\n" + SCRIPT + "\n```")

        self.assertTrue(cleaned.startswith("GLOBAL"))
        self.assertIn("TITLE:", cleaned)
        self.assertIn("DESCRIPTION:", cleaned)
        self.assertEqual(extract_title(cleaned), "Pip and the Moon Lantern")

    def test_story_prompts_ask_for_title_and_description(self):
        for prompt in (STORY_PROMPT, KIDS_STORY_PROMPT):
            self.assertIn("TITLE:", prompt)
            self.assertIn("DESCRIPTION:", prompt)
            self.assertNotIn("DO NOT create a TITLE field", prompt)
            self.assertNotIn("Do not add a title.", prompt)

    def test_generation_banner_includes_the_story_label(self):
        import ai.claude_browser as module

        class FakeBrowser:
            def __init__(self, timeout=0):
                pass

            def generate_story(self, prompt):
                return SCRIPT

            def close(self):
                pass

        buffer = io.StringIO()
        with mock.patch.object(
            module, "ClaudeBrowser", FakeBrowser
        ), contextlib.redirect_stdout(buffer):
            module.generate_story_with_claude(label="story_1794133267")

        output = buffer.getvalue()
        self.assertIn(
            "CLAUDE STORY GENERATION: story_1794133267", output
        )

    def test_generation_banner_without_label_is_unchanged(self):
        import ai.claude_browser as module

        class FakeBrowser:
            def __init__(self, timeout=0):
                pass

            def generate_story(self, prompt):
                return SCRIPT

            def close(self):
                pass

        buffer = io.StringIO()
        with mock.patch.object(
            module, "ClaudeBrowser", FakeBrowser
        ), contextlib.redirect_stdout(buffer):
            module.generate_story_with_claude()

        self.assertIn("CLAUDE STORY GENERATION (attempt 1/2)", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
