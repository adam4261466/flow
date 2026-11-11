"""Tests for the offline (no Ollama / no LLM) planner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.local_video import AutoVideoPlanner, LocalVideoPlanner  # noqa: E402
from parser.script_parser import parse_script  # noqa: E402
from tests.fixtures import VALID_SCRIPT  # noqa: E402


NO_TITLE_SCRIPT = """$$
GLOBAL
Characters:
Olive: a small round baby owl with a yellow scarf.
Style:
warm storybook
$$
CLIP
Characters:
Olive: a small round baby owl with a yellow scarf.
Moment:
Establishing shot of the giant oak tree at dusk.
Vibe:
cozy
Style:
warm storybook
Script:
Olive the owl knew a very big word, and it was only five letters long.
$$
CLIP
Characters:
Olive
Moment:
Olive whispers the word to her friends.
Vibe:
happy
Style:
warm storybook
Script:
She whispered it to the whole wide wood.
$$
"""


class LocalPlannerTests(unittest.TestCase):
    def test_uses_title_header_when_present(self):
        clips = parse_script(VALID_SCRIPT)
        planner = LocalVideoPlanner(VALID_SCRIPT)

        self.assertEqual(planner.generate_title(clips), "Pip and the Moon Lantern")

    def test_derives_title_without_title_header(self):
        clips = parse_script(NO_TITLE_SCRIPT)
        title = LocalVideoPlanner(NO_TITLE_SCRIPT).generate_title(clips)

        self.assertTrue(title)
        self.assertNotIn("untitled", title.lower())
        self.assertGreaterEqual(len(title.split()), 2)

    def test_description_keeps_sentence_punctuation(self):
        clips = parse_script(VALID_SCRIPT)
        description = LocalVideoPlanner(VALID_SCRIPT).generate_description(
            clips, "Pip and the Moon Lantern"
        )

        self.assertIn("afraid of the dark.", description)
        self.assertIn("\n\n", description)

    def test_visual_plan_is_complete_and_shake_is_disabled(self):
        clips = parse_script(VALID_SCRIPT)
        plans = LocalVideoPlanner(VALID_SCRIPT).plan_clips(clips)

        self.assertEqual(set(plans), {clip.id for clip in clips})
        for plan in plans.values():
            self.assertEqual(plan["shake"], "none")
            self.assertIn(
                plan["motion"],
                {"static", "zoom_in", "zoom_out", "push_in", "pull_out",
                 "pan_left", "pan_right", "pan_up", "pan_down"},
            )
            self.assertIn(
                plan["transition_after"], {"cut", "fade_black", "flash_white"}
            )
            self.assertIn(
                plan["grade"], {"neutral", "cinematic", "moody", "cool", "warm"}
            )

    def test_thumbnail_plan_has_every_prompt_field(self):
        clips = parse_script(VALID_SCRIPT)
        plan = LocalVideoPlanner(VALID_SCRIPT).generate_thumbnail_plan(
            clips, "Pip and the Moon Lantern"
        )

        for key in (
            "subject", "scene", "emotion", "camera", "lighting",
            "color_palette", "composition", "title_position",
            "reference_clip",
        ):
            self.assertIn(key, plan)

        self.assertIn(plan["title_position"], {
            "top_left", "top_right", "bottom_left", "bottom_right",
        })
        self.assertIn(plan["reference_clip"], {clip.id for clip in clips})

    def test_auto_planner_falls_back_when_primary_fails(self):
        clips = parse_script(VALID_SCRIPT)

        class BrokenPlanner:
            def generate_title(self, clips):
                raise RuntimeError("402 Payment Required")

        fallback = LocalVideoPlanner(VALID_SCRIPT)
        planner = AutoVideoPlanner(BrokenPlanner(), fallback, log=lambda _: None)

        self.assertEqual(
            planner.generate_title(clips),
            "Pip and the Moon Lantern",
        )


if __name__ == "__main__":
    unittest.main()
