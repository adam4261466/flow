"""Tests for detecting Gemini's usage-limit messages in chat text."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.gemini_browser import _looks_like_usage_limit  # noqa: E402


class GeminiLimitTextTests(unittest.TestCase):
    def test_quota_text_response_is_a_limit(self):
        self.assertTrue(
            _looks_like_usage_limit(
                "I can create more images as soon as your limit resets. "
                "Check your usage in Settings."
            )
        )

    def test_banner_wording_is_a_limit(self):
        self.assertTrue(_looks_like_usage_limit("Limit reached. Falling back."))
        self.assertTrue(_looks_like_usage_limit("You have reached your limit."))

    def test_normal_answer_is_not_a_limit(self):
        self.assertFalse(_looks_like_usage_limit("Here is the image you asked for."))
        self.assertFalse(_looks_like_usage_limit(""))


if __name__ == "__main__":
    unittest.main()
