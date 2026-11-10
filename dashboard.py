#!/usr/bin/env python3
"""FLOW control center - one local dashboard to run the whole story factory.

    py -3.13 dashboard.py            # opens the app window
    py -3.13 dashboard.py --no-open  # just start the server

Standard library only. It never imports the heavy pipeline modules: every
action starts the existing CLIs (batch_generator.py, main.py,
youtube_uploader.py) as one managed subprocess, so a crash in a render can
never take the dashboard down.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
UI_FILE = ROOT / "ui" / "index.html"
PROMPTS = ROOT / "prompts.txt"
BATCH_STATE = ROOT / "batch_state.json"
LEDGER = ROOT / "factory_state.json"
STATUS = ROOT / "utils" / "factory_status.json"
OUTPUT = ROOT / "output"
OUTBOX = ROOT / "outbox"
BATCH_INPUT = ROOT / "input" / "batch"
MANIFEST = OUTPUT / "youtube_uploads.json"
SETTINGS = ROOT / "dashboard_settings.json"
LOG_DIR = ROOT / "state" / "logs"

STORY_RE = re.compile(r"^story_[A-Za-z0-9_\-]+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")
CDP_PORT, TTS_PORT, OLLAMA_PORT = 9222, 8004, 11434

DEFAULT_SETTINGS = {
    "count": 30, "theme": "", "kids": True, "voices_first": True,
    "max_wait_minutes": 0, "max_story_attempts": 5, "no_agents": False,
    "upload_each": True, "upload": True, "per_day": 3, "start_per_day": "",
    "ramp_days": 0, "start_date": "", "publish_time": "18:00",
    "stage_parallel": True, "clip_workers": 1, "stage_retries": 3,
    "stage_poll_seconds": 5.0, "stage_retry_seconds": 30.0,
}


# ---------------------------------------------------------------- helpers
def read_json(path: Path, default):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, type(default)) else default
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_text(path: Path, limit: int = 60000) -> str:
    try:
        return path.read_text(encoding="utf-8")[:limit]
    except (OSError, UnicodeDecodeError):
        return ""


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port")


def num(value, default, low=0, high=10**6, cast=int):
    try:
        return max(low, min(high, cast(value)))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------- event hub
class Hub:
    """Tiny pub/sub that feeds every open browser tab (Server-Sent Events)."""

    def __init__(self):
        self.subs: set[queue.Queue] = set()
        self.lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self.lock:
            self.subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            self.subs.discard(q)

    def publish(self, event: str, data) -> None:
        with self.lock:
            targets = list(self.subs)
        for q in targets:
            try:
                q.put_nowait((event, data))
            except queue.Full:
                pass  # slow tab: drop, it resyncs on the next state event


HUB = Hub()


# ----------------------------------------------------------- job manager
class Job:
    def __init__(self, kind: str, label: str, cmd: list[str]):
        self.id = f"{int(time.time())}"
        self.kind, self.label, self.cmd = kind, label, cmd
        self.started = time.time()
        self.ended: float | None = None
        self.code: int | None = None
        self.stopping = False
        self.lines: deque[str] = deque(maxlen=3000)
        self.proc: subprocess.Popen | None = None
        self.logfile: Path | None = None

    def public(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "label": self.label,
            "started": self.started, "ended": self.ended, "code": self.code,
            "running": self.ended is None, "stopping": self.stopping,
            "cmd": " ".join(self.cmd[1:]),
        }


class Jobs:
    """Runs one pipeline process at a time (they all share one Chrome)."""

    def __init__(self):
        self.current: Job | None = None
        self.history: deque[Job] = deque(maxlen=12)
        self.lock = threading.Lock()

    def running(self) -> Job | None:
        j = self.current
        return j if j and j.ended is None else None

    def start(self, kind: str, label: str, cmd: list[str]) -> Job:
        with self.lock:
            if self.running():
                raise RuntimeError(f"'{self.current.label}' is still running. Stop it first.")
            job = Job(kind, label, cmd)
            env = os.environ.copy()
            env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            kw: dict = {}
            if os.name == "nt":
                kw["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                )
            else:
                kw["start_new_session"] = True
            job.proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw,
            )
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            job.logfile = LOG_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{kind}.log"
            self.current = job
            self.history.appendleft(job)
        threading.Thread(target=self._pump, args=(job,), daemon=True).start()
        self._prune_logs()
        HUB.publish("job", job.public())
        self.emit(job, f"▶ {label}", "sys")
        self.emit(job, "$ " + " ".join(cmd), "sys")
        return job

    def emit(self, job: Job, text: str, tag: str = "out") -> None:
        job.lines.append(text)
        if job.logfile:
            try:
                with job.logfile.open("a", encoding="utf-8") as f:
                    f.write(text + "\n")
            except OSError:
                pass
        HUB.publish("log", {"job": job.id, "tag": tag, "text": text})

    def _pump(self, job: Job) -> None:
        import io
        stream = io.TextIOWrapper(job.proc.stdout, encoding="utf-8", errors="replace", newline=None)
        for line in stream:
            line = line.rstrip("\n")
            if line.strip():
                self.emit(job, line)
        job.code = job.proc.wait()
        job.ended = time.time()
        verdict = "stopped" if job.stopping else ("finished" if job.code == 0 else f"exited with code {job.code}")
        self.emit(job, f"■ {job.label} {verdict}", "sys")
        HUB.publish("job", job.public())

    def stop(self) -> bool:
        job = self.running()
        if not job or not job.proc:
            return False
        job.stopping = True
        pid = job.proc.pid
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                threading.Timer(6, lambda: job.proc.poll() is None and os.killpg(os.getpgid(pid), signal.SIGKILL)).start()
        except (OSError, ProcessLookupError):
            pass
        HUB.publish("job", job.public())
        return True

    @staticmethod
    def _prune_logs(keep: int = 30) -> None:
        try:
            files = sorted(LOG_DIR.glob("*.log"))
            for old in files[:-keep]:
                old.unlink(missing_ok=True)
        except OSError:
            pass


JOBS = Jobs()


# ------------------------------------------------------- command builders
def py(*args: str) -> list[str]:
    return [sys.executable, *args]


def flag(cmd: list[str], on, name: str) -> None:
    if on:
        cmd.append(name)


def opt(cmd: list[str], value, name: str) -> None:
    if value not in (None, "", False):
        cmd.extend([name, str(value)])


def build_factory(b: dict) -> tuple[str, list[str]]:
    c = py("batch_generator.py", "--prompts-file", "prompts.txt")
    opt(c, str(b.get("theme", "")).strip(), "--theme")
    c += ["--count", str(num(b.get("count"), 30, 1, 500))]
    flag(c, b.get("kids"), "--kids")
    c += ["--max-story-attempts", str(num(b.get("max_story_attempts"), 5, 1, 50))]
    c.append("--voices-first" if b.get("voices_first", True) else "--no-voices-first")
    c += ["--max-wait-minutes", str(num(b.get("max_wait_minutes"), 0, 0, 100000, float))]
    flag(c, b.get("no_agents"), "--no-agents")
    c.append("--stage-parallel" if b.get("stage_parallel", True) else "--no-stage-parallel")
    c += ["--clip-workers", str(num(b.get("clip_workers"), 1, 1, 8))]
    c += ["--stage-retries", str(num(b.get("stage_retries"), 3, 0, 20))]
    c += ["--stage-poll-seconds", str(num(b.get("stage_poll_seconds"), 5.0, 1.0, 300.0, float))]
    c += ["--stage-retry-seconds", str(num(b.get("stage_retry_seconds"), 30.0, 1.0, 3600.0, float))]
    flag(c, b.get("upload_each"), "--upload-each")
    flag(c, b.get("upload"), "--upload")
    flag(c, b.get("dry_run_upload"), "--dry-run-upload")
    c += ["--per-day", str(num(b.get("per_day"), 3, 1, 50))]
    if str(b.get("start_per_day", "")).strip():
        c += ["--start-per-day", str(num(b.get("start_per_day"), 1, 1, 50))]
    if num(b.get("ramp_days"), 0, 0, 365):
        c += ["--ramp-days", str(num(b.get("ramp_days"), 0, 0, 365))]
    if DATE_RE.match(str(b.get("start_date", ""))):
        c += ["--start-date", b["start_date"]]
    if TIME_RE.match(str(b.get("publish_time", ""))):
        c += ["--publish-time", b["publish_time"]]
    return "Factory", c


def build_story(b: dict) -> tuple[str, list[str]]:
    story = str(b.get("story", "")).strip()
    if not STORY_RE.match(story):
        raise ValueError("Pick a valid story id.")
    script = BATCH_INPUT / f"{story}.txt"
    if not script.exists():
        script = ROOT / "input" / "script.txt"
    c = py("main.py", "--story", story, "--script", str(script), "--after-finish", "none")
    for key, name in (("kids", "--kids"), ("no_agents", "--no-agents"), ("recap", "--recap"),
                      ("voices_only", "--voices-only"), ("images_only", "--images-only"),
                      ("clips_only", "--clips-only"), ("final_only", "--final-only"),
                      ("thumbnail_only", "--thumbnail-only"),
                      ("regenerate_thumbnail", "--regenerate-thumbnail"),
                      ("no_parallel", "--no-parallel")):
        flag(c, b.get(key), name)
    return f"Story {story}", c


def build_upload(b: dict) -> tuple[str, list[str]]:
    c = py("youtube_uploader.py")
    c += ["--per-day", str(num(b.get("per_day"), 3, 1, 50))]
    if TIME_RE.match(str(b.get("publish_time", ""))):
        c += ["--time", b["publish_time"]]
    if DATE_RE.match(str(b.get("start_date", ""))):
        c += ["--start-date", b["start_date"]]
    if str(b.get("start_per_day", "")).strip():
        c += ["--start-per-day", str(num(b.get("start_per_day"), 1, 1, 50))]
    if num(b.get("ramp_days"), 0, 0, 365):
        c += ["--ramp-days", str(num(b.get("ramp_days"), 0, 0, 365))]
    flag(c, b.get("dry_run"), "--dry-run")
    return ("Upload plan (dry run)" if b.get("dry_run") else "YouTube upload"), c


def build_login(_: dict) -> tuple[str, list[str]]:
    code = "from ai.chrome_cdp import open_login_tabs; open_login_tabs()"
    return "Open login tabs", py("-c", code)


def build_ideas(b: dict) -> tuple[str, list[str]]:
    code = (
        "import sys; from pathlib import Path; from agents.prompt_architect import PromptArchitect\n"
        "theme, count, kids = sys.argv[1], int(sys.argv[2]), sys.argv[3] == '1'\n"
        "new = PromptArchitect().generate_batch(theme=theme, count=count, kids=kids)\n"
        "p = Path('prompts.txt'); old = p.read_text(encoding='utf-8').splitlines() if p.exists() else []\n"
        "have = {x.strip() for x in old}; add = [x for x in new if x.strip() not in have]\n"
        "p.write_text('\\n'.join([x for x in old if x.strip()] + add), encoding='utf-8')\n"
        "print(f'Added {len(add)} new story ideas to prompts.txt')\n"
    )
    theme = str(b.get("theme", "")).strip() or (
        "gentle bedtime stories for young children: animal friends, gentle magic, kindness"
        if b.get("kids", True) else "dark cinematic mysteries")
    return "Generate story ideas", py("-c", code, theme, str(num(b.get("count"), 10, 1, 100)),
                                       "1" if b.get("kids", True) else "0")


def build_tests(_: dict) -> tuple[str, list[str]]:
    try:
        import pytest  # noqa: F401
        return "Self-test", py("-m", "pytest", "tests", "-q")
    except ImportError:
        return "Self-test", py("-m", "unittest", "discover", "-s", "tests")


BUILDERS = {"factory": build_factory, "story": build_story, "upload": build_upload,
            "login": build_login, "ideas": build_ideas, "tests": build_tests}


# ------------------------------------------------------------ data views
_health = {"chrome": False, "tts": False, "ollama": False, "checked": 0.0}


def health_loop() -> None:
    while True:
        _health.update(chrome=port_open(CDP_PORT), tts=port_open(TTS_PORT),
                       ollama=port_open(OLLAMA_PORT), checked=time.time())
        time.sleep(4)


def story_stage(sid: str, manifest: dict) -> dict:
    out = OUTPUT / sid
    steps = {
        "script": (BATCH_INPUT / f"{sid}.txt").exists(),
        "voice": any((out / "audio").glob("audio_*.wav")) if (out / "audio").is_dir() else False,
        "images": any((out / "images").glob("*.*")) if (out / "images").is_dir() else False,
        "video": (out / "video" / "final_video.mp4").exists(),
        "thumb": (out / "thumbnail" / "thumbnail.png").exists(),
        "outbox": (OUTBOX / sid / "video.mp4").exists(),
        "uploaded": (manifest.get(sid) or {}).get("status") in ("scheduled", "uploaded", "published"),
    }
    return steps


def queue_view() -> dict:
    prompts = [l.strip() for l in read_text(PROMPTS).splitlines() if l.strip()]
    state = read_json(BATCH_STATE, {})
    stories = state.get("stories", {}) if isinstance(state.get("stories"), dict) else {}
    pending = state.get("pending", []) or []
    completed = set(state.get("completed", []) or [])
    manifest = read_json(MANIFEST, {})
    status = read_json(STATUS, {})
    active = status.get("current_story") if JOBS.running() else None
    order = list(dict.fromkeys(list(pending) + [p for p in prompts if p not in completed and p not in pending]
                               + [p for p in prompts if p in completed]))
    items = []
    for p in order:
        info = stories.get(p, {}) if isinstance(stories.get(p), dict) else {}
        sid = info.get("story_id", "")
        steps = story_stage(sid, manifest) if sid else {}
        if p in completed:
            st = "done" if steps.get("outbox") or steps.get("video") else "skipped"
        elif sid and sid == active:
            st = "active"
        else:
            st = "pending"
        items.append({"prompt": p, "story": sid, "attempts": info.get("attempts", 0),
                      "status": st, "steps": steps})
    return {"items": items, "kids": bool(state.get("settings", {}).get("kids")),
            "counts": {k: sum(1 for i in items if i["status"] == k)
                       for k in ("pending", "active", "done", "skipped")}}


def limits_view() -> dict:
    res = read_json(LEDGER, {}).get("resources", {})
    now = time.time()
    out = {}
    for name, e in res.items():
        if not isinstance(e, dict):
            continue
        left = max(0.0, float(e.get("blocked_until", 0) or 0) - now)
        out[name] = {"remaining": left, "until": e.get("blocked_until_text", ""),
                     "message": str(e.get("message", ""))[:200], "blocked": left > 0}
    return out


def library_view() -> list[dict]:
    manifest = read_json(MANIFEST, {})
    seen: dict[str, dict] = {}
    for base in (OUTBOX, OUTPUT):
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if not (d.is_dir() and STORY_RE.match(d.name)) or d.name in seen:
                continue
            video = d / "video.mp4" if base == OUTBOX else d / "video" / "final_video.mp4"
            if not video.exists():
                continue
            thumb = d / "thumbnail.png" if base == OUTBOX else d / "thumbnail" / "thumbnail.png"
            title = read_text(d / "title.txt", 300).strip() or "Untitled"
            rec = manifest.get(d.name) or {}
            seen[d.name] = {"id": d.name, "title": title, "thumb": thumb.exists(),
                            "size": video.stat().st_size, "mtime": video.stat().st_mtime,
                            "source": base.name, "status": rec.get("status", "ready"),
                            "scheduled_at": rec.get("scheduled_at", ""),
                            "video_url": rec.get("video_url", ""), "error": rec.get("error", "")}
    return sorted(seen.values(), key=lambda s: s["mtime"], reverse=True)


_snap = {"at": 0.0, "data": None, "lib_at": 0.0, "lib": []}
_snap_lock = threading.Lock()


def snapshot() -> dict:
    """Cached for 1s so many open tabs cost the same as one."""
    with _snap_lock:
        if _snap["data"] is not None and time.time() - _snap["at"] < 1.0:
            return _snap["data"]
        if time.time() - _snap["lib_at"] > 5.0:
            _snap["lib"], _snap["lib_at"] = library_view(), time.time()
        _snap["data"], _snap["at"] = _build_snapshot(_snap["lib"]), time.time()
        return _snap["data"]


def _build_snapshot(library: list) -> dict:
    status = read_json(STATUS, {})
    return {
        "time": time.time(),
        "job": JOBS.current.public() if JOBS.current else None,
        "history": [j.public() for j in JOBS.history],
        "status": status if JOBS.running() else {},
        "health": {k: _health[k] for k in ("chrome", "tts", "ollama")},
        "limits": limits_view(),
        "queue": queue_view(),
        "library": library,
        "settings": {**DEFAULT_SETTINGS, **read_json(SETTINGS, {})},
    }


# ----------------------------------------------------------- queue edits
def queue_edit(b: dict) -> dict:
    action = b.get("action")
    prompt = str(b.get("prompt", "")).strip()
    if action == "add":
        new = [l.strip() for l in str(b.get("text", "")).splitlines() if l.strip()]
        old = [l.strip() for l in read_text(PROMPTS).splitlines() if l.strip()]
        add = [n for n in dict.fromkeys(new) if n not in old]
        PROMPTS.write_text("\n".join(old + add), encoding="utf-8")
        return {"added": len(add)}
    if JOBS.running():
        raise RuntimeError("Stop the running job before reordering or removing queue items.")
    state = read_json(BATCH_STATE, {})
    state.setdefault("pending", []); state.setdefault("completed", []); state.setdefault("stories", {})
    state.setdefault("settings", {})
    if action == "remove":
        old = [l.strip() for l in read_text(PROMPTS).splitlines() if l.strip()]
        PROMPTS.write_text("\n".join(x for x in old if x != prompt), encoding="utf-8")
        state["pending"] = [x for x in state["pending"] if x != prompt]
    elif action in ("top", "bottom") and prompt in state["pending"]:
        state["pending"].remove(prompt)
        state["pending"].insert(0, prompt) if action == "top" else state["pending"].append(prompt)
    elif action == "retry":
        state["completed"] = [x for x in state["completed"] if x != prompt]
        if prompt not in state["pending"]:
            state["pending"].insert(0, prompt)
        if isinstance(state["stories"].get(prompt), dict):
            state["stories"][prompt]["attempts"] = 0
    elif action == "retry_skipped":
        for it in queue_view()["items"]:
            if it["status"] == "skipped":
                state["completed"] = [x for x in state["completed"] if x != it["prompt"]]
                if it["prompt"] not in state["pending"]:
                    state["pending"].append(it["prompt"])
                if isinstance(state["stories"].get(it["prompt"]), dict):
                    state["stories"][it["prompt"]]["attempts"] = 0
    else:
        raise ValueError("Unknown queue action.")
    write_json(BATCH_STATE, state)
    return {"ok": True}


def clear_limit(name: str) -> None:
    data = read_json(LEDGER, {"version": 1, "resources": {}})
    data.setdefault("resources", {}).pop(name, None)
    if name == "all":
        data["resources"] = {}
    write_json(LEDGER, data)


def story_detail(sid: str) -> dict:
    out, box = OUTPUT / sid, OUTBOX / sid
    manifest = read_json(MANIFEST, {})
    return {
        "id": sid,
        "title": read_text(box / "title.txt", 300) or read_text(out / "title.txt", 300),
        "description": read_text(box / "description.txt") or read_text(out / "description.txt"),
        "script": read_text(BATCH_INPUT / f"{sid}.txt", 20000),
        "steps": story_stage(sid, manifest),
        "upload": manifest.get(sid, {}),
        "thumb": (box / "thumbnail.png").exists() or (out / "thumbnail" / "thumbnail.png").exists(),
    }


def media_path(sid: str, kind: str) -> Path | None:
    if not STORY_RE.match(sid):
        return None
    cands = {"video": [OUTBOX / sid / "video.mp4", OUTPUT / sid / "video" / "final_video.mp4"],
             "thumb": [OUTBOX / sid / "thumbnail.png", OUTPUT / sid / "thumbnail" / "thumbnail.png"]}.get(kind, [])
    return next((p for p in cands if p.exists()), None)


def open_folder(which: str, sid: str = "") -> None:
    target = {"output": OUTPUT, "outbox": OUTBOX, "root": ROOT, "logs": LOG_DIR}.get(which, ROOT)
    if sid and STORY_RE.match(sid):
        target = (OUTBOX if (OUTBOX / sid).is_dir() else OUTPUT) / sid
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(target))  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


# ------------------------------------------------------------ HTTP layer
class Handler(BaseHTTPRequestHandler):
    server_version = "Flow/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # keep the console quiet
        pass

    # -- plumbing
    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}

    def _same_origin(self) -> bool:
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        return host.startswith(("127.0.0.1", "localhost")) and (
            origin is None or urlparse(origin).netloc == host)

    # -- GET
    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            body = UI_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/state":
            self._json(snapshot())
        elif path == "/api/events":
            self._sse()
        elif path.startswith("/api/story/"):
            sid = path.rsplit("/", 1)[-1]
            self._json(story_detail(sid) if STORY_RE.match(sid) else {}, 200 if STORY_RE.match(sid) else 404)
        elif path == "/api/log":
            job = JOBS.current
            self._json({"job": job.id if job else "", "lines": list(job.lines) if job else []})
        elif path.startswith("/media/"):
            parts = path.split("/")
            p = media_path(parts[2], parts[3]) if len(parts) == 4 else None
            self._file(p) if p else self._json({"error": "not found"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def _file(self, p: Path):
        size = p.stat().st_size
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else 0
            end = int(b) if b else size - 1
            end = min(end, size - 1)
            status = 206
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with p.open("rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    chunk = f.read(min(262144, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = HUB.subscribe()
        last_state = 0.0
        try:
            while True:
                try:
                    event, data = q.get(timeout=1.0)
                    self._emit(event, data)
                except queue.Empty:
                    pass
                if time.time() - last_state >= 1.5:
                    last_state = time.time()
                    self._emit("state", snapshot())
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            HUB.unsubscribe(q)

    def _emit(self, event: str, data):
        payload = json.dumps(data, ensure_ascii=False)
        self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    # -- POST
    def do_POST(self):
        if not self._same_origin():
            return self._json({"error": "forbidden"}, 403)
        path, b = urlparse(self.path).path, self._body()
        try:
            if path == "/api/run":
                kind = b.get("kind")
                if kind not in BUILDERS:
                    raise ValueError("Unknown action.")
                label, cmd = BUILDERS[kind](b)
                self._json(JOBS.start(kind, label, cmd).public())
            elif path == "/api/stop":
                self._json({"stopped": JOBS.stop()})
            elif path == "/api/queue":
                self._json(queue_edit(b))
            elif path == "/api/limits/clear":
                clear_limit(str(b.get("name", "all")))
                self._json({"ok": True})
            elif path == "/api/settings":
                clean = {k: b[k] for k in DEFAULT_SETTINGS if k in b}
                write_json(SETTINGS, {**read_json(SETTINGS, {}), **clean})
                self._json({"ok": True})
            elif path == "/api/open":
                open_folder(str(b.get("what", "root")), str(b.get("story", "")))
                self._json({"ok": True})
            elif path == "/api/shutdown":
                self._json({"ok": True})
                JOBS.stop()
                threading.Timer(0.3, lambda: os._exit(0)).start()
            else:
                self._json({"error": "not found"}, 404)
        except (RuntimeError, ValueError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


# ------------------------------------------------------------- launcher
def open_app_window(url: str) -> None:
    """Chrome/Edge in --app mode = a clean native-looking window, no tabs."""
    candidates = [
        shutil.which("msedge"), shutil.which("chrome"), shutil.which("google-chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    profile = ROOT / "state" / "app-profile"  # separate from the automation Chrome
    for exe in candidates:
        if exe and Path(exe).exists():
            subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={profile}",
                              "--window-size=1440,900", "--no-first-run"])
            return
    webbrowser.open(url)


def main() -> None:
    ap = argparse.ArgumentParser(description="FLOW control center")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    port = free_port(args.port)
    threading.Thread(target=health_loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{port}"
    print(f"FLOW control center running at {url}  (Ctrl+C to quit)")
    if not args.no_open:
        threading.Timer(0.6, open_app_window, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        JOBS.stop()


if __name__ == "__main__":
    main()
