"""Tests for the Chatterbox TTS client's startup and progress reporting."""

from __future__ import annotations

import contextlib
import io
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio import chatterbox_tts  # noqa: E402
from audio.chatterbox_tts import ChatterboxTTS  # noqa: E402


class FakeResponse:
    status_code = 200
    content = b"x" * 512

    def json(self):
        return {}


def make_client(**kwargs):
    options = {
        "auto_start": True,
        "startup_timeout": 0.3,
        "poll_interval": 0.02,
    }
    options.update(kwargs)
    return ChatterboxTTS(**options)


class ServerStartupTests(unittest.TestCase):
    def test_busy_port_does_not_start_a_second_server(self):
        tts = make_client()

        with mock.patch.object(
            tts, "is_server_running", return_value=False
        ), mock.patch.object(
            tts, "_port_is_open", return_value=True
        ), mock.patch.object(
            tts, "start_server"
        ) as start, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                tts.ensure_server_running()

        start.assert_not_called()

    def test_closed_port_starts_the_server(self):
        tts = make_client()

        with mock.patch.object(
            tts, "is_server_running", side_effect=[False, True]
        ), mock.patch.object(
            tts, "_port_is_open", return_value=False
        ), mock.patch.object(
            tts, "start_server"
        ) as start, contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(tts.ensure_server_running())

        start.assert_called_once()

    def test_auto_start_disabled_raises_without_launching(self):
        tts = make_client(auto_start=False)

        with mock.patch.object(
            tts, "is_server_running", return_value=False
        ), mock.patch.object(
            tts, "start_server"
        ) as start, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                tts.ensure_server_running()

        start.assert_not_called()


class ProgressReportingTests(unittest.TestCase):
    def test_heartbeat_prints_while_request_runs(self):
        tts = make_client(auto_start=False)
        tts.progress_interval = 0.05
        tts.is_server_running = lambda: True

        def slow_post(*args, **kwargs):
            time.sleep(0.2)
            return FakeResponse()

        buffer = io.StringIO()
        with mock.patch.object(
            chatterbox_tts.requests, "post", slow_post
        ), contextlib.redirect_stdout(buffer):
            tts.generate_bytes("hello there")

        self.assertIn("still synthesizing", buffer.getvalue())

    def test_heartbeat_stops_after_the_request(self):
        tts = make_client(auto_start=False)
        tts.progress_interval = 0.05
        tts.is_server_running = lambda: True

        buffer = io.StringIO()
        with mock.patch.object(
            chatterbox_tts.requests,
            "post",
            lambda *a, **k: FakeResponse(),
        ), contextlib.redirect_stdout(buffer):
            tts.generate_bytes("hi")
            time.sleep(0.15)

        self.assertNotIn("still synthesizing", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
