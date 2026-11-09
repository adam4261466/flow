"""Tests for the single-instance YouTube upload lock."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_uploader import UploadLock  # noqa: E402


class UploadLockTests(unittest.TestCase):
    def test_second_process_cannot_acquire(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.lock"
            first = UploadLock(path)
            second = UploadLock(path)

            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())

            first.release()
            self.assertTrue(second.acquire())
            second.release()

    def test_release_removes_the_lock_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.lock"
            lock = UploadLock(path)
            self.assertTrue(lock.acquire())
            self.assertTrue(path.exists())
            lock.release()
            self.assertFalse(path.exists())

    def test_stale_lock_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.lock"
            path.write_text(
                json.dumps(
                    {
                        "pid": 999_999_999,
                        "started": time.time() - 60,
                    }
                ),
                encoding="utf-8",
            )

            lock = UploadLock(path, stale_seconds=60.0)
            self.assertTrue(lock.acquire())
            info = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotEqual(info["pid"], 999_999_999)
            lock.release()

    def test_dead_uploader_lock_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.lock"
            path.write_text(
                json.dumps(
                    {
                        "pid": 999_999_999,
                        "started": time.time(),
                    }
                ),
                encoding="utf-8",
            )

            lock = UploadLock(path)
            self.assertTrue(lock.acquire())
            lock.release()

    def test_release_does_not_remove_someone_elses_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.lock"
            # A live process (the test runner's parent) owns this lock.
            path.write_text(
                json.dumps({"pid": os.getppid(), "started": time.time()}),
                encoding="utf-8",
            )

            lock = UploadLock(path)
            self.assertFalse(lock.acquire())
            lock.release()
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
