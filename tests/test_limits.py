"""Tests for the shared usage-limit handling.

Run with::

    py -3.13 -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import batch_generator  # noqa: E402
from agents.limits import LimitLedger  # noqa: E402
from batch_generator import (  # noqa: E402
    block_on_limit,
    parse_limit_minutes,
    wait_for_resource,
)


class FakeClock:
    """Deterministic clock: sleep() advances time instead of waiting."""

    def __init__(self, start: float = 1_700_000_000.0):
        self.now = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ParseLimitMinutesTests(unittest.TestCase):
    def test_clock_reset_message(self):
        # 5:00 PM is always 1..1440 minutes away, never negative or zero.
        minutes = parse_limit_minutes("You've reached your limit. Resets at 5:00 PM.")
        self.assertGreaterEqual(minutes, 1.0)
        self.assertLessEqual(minutes, 24 * 60.0)

    def test_relative_hours(self):
        self.assertEqual(parse_limit_minutes("Try again in 2 hours."), 120.0)

    def test_relative_minutes(self):
        self.assertEqual(parse_limit_minutes("Please wait, in 45 minutes."), 45.0)

    def test_falls_back_when_unparseable(self):
        self.assertEqual(parse_limit_minutes("Something went wrong"), 60.0)
        self.assertEqual(
            parse_limit_minutes("", default=17.0),
            17.0,
        )

    def test_block_on_limit_records_cooldown(self):
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as tmp:
            ledger = LimitLedger(Path(tmp) / "factory_state.json", clock=clock.time)
            minutes = block_on_limit(ledger, "claude", "in 2 hours")

            self.assertEqual(minutes, 120.0)
            self.assertGreater(ledger.remaining("claude"), 0.0)
            self.assertIn("blocked", ledger.describe("claude"))


class WaitForResourceTests(unittest.TestCase):
    def _ledger(self, tmp, clock):
        return LimitLedger(Path(tmp) / "factory_state.json", clock=clock.time)

    def test_limit_is_announced_once_then_silent(self):
        clock = FakeClock()
        announced: dict = {}
        with tempfile.TemporaryDirectory() as tmp:
            ledger = self._ledger(tmp, clock)
            ledger.block("gemini", 3.0, "You reached your limit")

            buffer = io.StringIO()
            with mock.patch.object(batch_generator.time, "monotonic", clock.monotonic), \
                    mock.patch.object(batch_generator.time, "sleep", clock.sleep), \
                    redirect_stdout(buffer):
                ok = wait_for_resource(
                    ledger,
                    "gemini",
                    label="Gemini",
                    max_wait_minutes=240.0,
                    announced=announced,
                )

            output = buffer.getvalue()
            self.assertTrue(ok)
            self.assertEqual(output.count("limit reached"), 1)
            self.assertEqual(output.count("This message will not repeat."), 1)
            self.assertIn("limit cleared", output)
            # The wait slept in small chunks, never one giant blocking call.
            self.assertTrue(clock.sleeps)
            self.assertTrue(all(sleep <= 30.0 for sleep in clock.sleeps))

    def test_already_announced_limit_stays_silent(self):
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as tmp:
            ledger = self._ledger(tmp, clock)
            ledger.block("claude", 1.0, "resets at 5:00 PM")
            info = ledger.info("claude")
            token = (info.get("last_hit_at"), info.get("blocked_until_text"))
            announced = {"claude": token}

            buffer = io.StringIO()
            with mock.patch.object(batch_generator.time, "monotonic", clock.monotonic), \
                    mock.patch.object(batch_generator.time, "sleep", clock.sleep), \
                    redirect_stdout(buffer):
                ok = wait_for_resource(
                    ledger,
                    "claude",
                    label="Claude",
                    max_wait_minutes=240.0,
                    announced=announced,
                )

            output = buffer.getvalue()
            self.assertTrue(ok)
            self.assertNotIn("limit reached", output)
            self.assertIn("limit cleared", output)

    def test_gives_up_after_max_wait(self):
        clock = FakeClock()
        announced: dict = {}
        with tempfile.TemporaryDirectory() as tmp:
            ledger = self._ledger(tmp, clock)
            ledger.block("gemini", 60.0, "cooldown")

            buffer = io.StringIO()
            with mock.patch.object(batch_generator.time, "monotonic", clock.monotonic), \
                    mock.patch.object(batch_generator.time, "sleep", clock.sleep), \
                    redirect_stdout(buffer):
                ok = wait_for_resource(
                    ledger,
                    "gemini",
                    label="Gemini",
                    max_wait_minutes=1.0,
                    announced=announced,
                )

            self.assertFalse(ok)
            self.assertIn("Stopping", buffer.getvalue())
            # It stopped long before burning the full hour of fake time.
            self.assertLess(clock.now - 1_700_000_000.0, 120.0)

    def test_ready_resource_returns_immediately(self):
        clock = FakeClock()
        announced: dict = {}
        with tempfile.TemporaryDirectory() as tmp:
            ledger = self._ledger(tmp, clock)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                ok = wait_for_resource(
                    ledger,
                    "gemini",
                    label="Gemini",
                    max_wait_minutes=240.0,
                    announced=announced,
                )
            self.assertTrue(ok)
            self.assertEqual(clock.sleeps, [])
            self.assertEqual(buffer.getvalue(), "")

    def test_sees_a_clear_written_by_another_process(self):
        clock = FakeClock()
        announced: dict = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "factory_state.json"
            ledger = LimitLedger(path, clock=clock.time)
            ledger.block("gemini", 10.0, "cooldown")
            writer = LimitLedger(path, clock=clock.time)

            start = clock.now

            def sleep(seconds):
                clock.sleep(seconds)
                # Simulate main.py clearing the ledger mid-wait.
                if clock.now - start >= 60.0:
                    writer.clear("gemini")

            buffer = io.StringIO()
            with mock.patch.object(batch_generator.time, "monotonic", clock.monotonic), \
                    mock.patch.object(batch_generator.time, "sleep", sleep), \
                    redirect_stdout(buffer):
                ok = wait_for_resource(
                    ledger,
                    "gemini",
                    label="Gemini",
                    max_wait_minutes=240.0,
                    announced=announced,
                )

            self.assertTrue(ok)
            self.assertLess(clock.now - start, 600.0)
            self.assertIn("limit cleared", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
