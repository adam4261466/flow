#! /usr/bin/env python3
"""
AI Video Creator — Full Factory GUI.
"""
import argparse
import contextlib
import io
import json
import os
import queue
import re
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
import tkinter as tk

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_ROOT = PROJECT_ROOT / "output"
OUTBOX_ROOT = PROJECT_ROOT / "outbox"

from main import (
    main as pipeline_main,
    parse_args as pipeline_parse_args,
    SCRIPT_PATH,
    STORY_NAME,
    STORY_OUTPUT,
    IMAGE_OUTPUT,
    AUDIO_OUTPUT,
    VIDEO_OUTPUT,
    THUMBNAIL_OUTPUT,
    CHECKPOINT_FILE,
    GEMINI_CONVERSATION_FILE,
    RESUME_EXISTING,
    AUTO_RESUME_AFTER_LIMIT,
)
from utils.status import read_status, update_status
from utils.console import setup_console

setup_console()

# ----------------------------------------------------------------------
# Log Capture
# ----------------------------------------------------------------------

class QueueWriter(io.TextIOBase):
    def __init__(self, q, tag="stdout"):
        self.q = q
        self.tag = tag

    def write(self, s):
        if s:
            self.q.put((self.tag, s))

    def flush(self):
        pass

def capture_logs(log_queue):
    """Redirects stdout/stderr to the GUI log queue."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = QueueWriter(log_queue, "stdout")
    sys.stderr = QueueWriter(log_queue, "stderr")
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# ----------------------------------------------------------------------
# Background Runners
# ----------------------------------------------------------------------

class PipelineRunner(threading.Thread):
    def __init__(self, args, log_queue, generate_story=False):
        super().__init__(daemon=True)
        self.args = args
        self.log_queue = log_queue
        self.generate_story = generate_story
        self._stop = threading.Event()

    def run(self):
        with capture_logs(self.log_queue):
            try:
                if self.generate_story:
                    # This is a simplified call; in a real build, we'd use a helper
                    pass
                pipeline_main(self.args)
            except Exception as e:
                self.log_queue.put(("error", f"Pipeline crashed: {e}\n"))

    def stop(self):
        self._stop.set()

class UploadRunner(threading.Thread):
    def __init__(self, log_queue, start_date, publish_time, videos_per_day=3):
        super().__init__(daemon=True)
        self.log_queue = log_queue
        self.start_date = start_date
        self.publish_time = publish_time
        self.videos_per_day = videos_per_day

    def run(self):
        with capture_logs(self.log_queue):
            try:
                from youtube_uploader import upload_all_finished_videos
                upload_all_finished_videos(
                    videos_per_day=self.videos_per_day,
                    start_date=self.start_date,
                    publish_time=self.publish_time,
                )
            except Exception as e:
                self.log_queue.put(("error", f"Upload crashed: {e}\n"))

class LoginRunner(threading.Thread):
    def __init__(self, log_queue):
        super().__init__(daemon=True)
        self.log_queue = log_queue

    def run(self):
        with capture_logs(self.log_queue):
            try:
                from ai.chrome_cdp import LOGIN_URLS
                from ai.chrome_cdp import launch_chrome
                launch_chrome()
                # Simplified: logically opens tabs
                self.log_queue.put(("info", "Chrome launched. Please sign in to Gemini, Claude, and YouTube.\n"))
            except Exception as e:
                self.log_queue.put(("error", f"Login launch crashed: {e}\n"))

# ----------------------------------------------------------------------
# Main Application
# ----------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Video Factory Dashboard")
        self.geometry("1280x850")
        self.minsize(1100, 700)

        self.log_queue = queue.Queue()
        self.runner = None
        self.upload_runner = None
        self.login_runner = None

        self._setup_styles()
        self._build_ui()
        self._poll_log_queue()
        self._update_dashboard_loop()

    def _setup_styles(self):
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="blue", font=("Segoe UI", 9, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Stat.TLabel", font=("Consolas", 10))

    def _build_ui(self):
        # Main layout: Notebook (Tabs)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        # 1. Dashboard Tab
        self.tab_dash = ttk.Frame(self.nb)
        self.nb.add(self.tab_dash, text="Dashboard")
        self._build_dashboard()

        # 2. Batch Queue Tab
        self.tab_queue = ttk.Frame(self.nb)
        self.nb.add(self.tab_queue, text="Queue")
        self._build_queue()

        # 3. Publishing Tab
        self.tab_pub = ttk.Frame(self.nb)
        self.nb.add(self.tab_pub, text="Publishing")
        self._build_publishing()

        # 4. Inspector Tab
        self.tab_insp = ttk.Frame(self.nb)
        self.nb.add(self.tab_insp, text="Inspector")
        self._build_inspector()

        # 5. Limits & Accounts Tab
        self.tab_limits = ttk.Frame(self.nb)
        self.nb.add(self.tab_limits, text="Limits & Accounts")
        self._build_limits()

        # 6. Settings Tab
        self.tab_settings = ttk.Frame(self.nb)
        self.nb.add(self.tab_settings, text="Settings")
        self._build_settings()

        # 7. Logs Tab
        self.tab_logs = ttk.Frame(self.nb)
        self.nb.add(self.tab_logs, text="Logs")
        self._build_logs()

    # ------------------------------------------------------------------
    # DASHBOARD
    # ------------------------------------------------------------------
    def _build_dashboard(self):
        # Top Row: Health/Status Badges
        health_frame = ttk.LabelFrame(self.tab_dash, text="System Health")
        health_frame.pack(fill="x", padx=10, pady=10)

        self.health_vars = {}
        for i, service in enumerate(["Chrome CDP", "TTS Server", "Gemini", "Claude", "YouTube"]):
            var = tk.StringVar(value="Checking...")
            self.health_vars[service] = var
            lbl = ttk.Label(health_frame, textvariable=var, style="Stat.TLabel")
            lbl.grid(row=0, column=i, padx=20, pady=10)

        # Middle Row: Live Progress
        prog_frame = ttk.LabelFrame(self.tab_dash, text="Live Production Status")
        prog_frame.pack(fill="x", padx=10, pady=10)

        self.cur_story_var = tk.StringVar(value="Idle")
        ttk.Label(prog_frame, text="Current Story:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ttk.Label(prog_frame, textvariable=self.cur_story_var, font=("Segoe UI", 11, "bold")).grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.cur_stage_var = tk.StringVar(value="N/A")
        ttk.Label(prog_frame, text="Stage:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ttk.Label(prog_frame, textvariable=self.cur_stage_var, font=("Segoe UI", 11, "bold")).grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.prog_bar = ttk.Progressbar(prog_frame, length=400, mode="determinate")
        self.prog_bar.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        self.batch_stat_var = tk.StringVar(value="Queue: 0/0")
        ttk.Label(prog_frame, textvariable=self.batch_stat_var).grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Bottom Row: Quick Controls
        ctrl_frame = ttk.Frame(self.tab_dash)
        ctrl_frame.pack(fill="x", padx=10, pady=10)

        self.start_factory_btn = ttk.Button(ctrl_frame, text="🚀 Start Factory", command=self._start_factory)
        self.start_factory_btn.pack(side="left", padx=10)
        
        self.stop_factory_btn = ttk.Button(ctrl_frame, text="🛑 Stop", command=self._stop_factory, state="disabled")
        self.stop_factory_btn.pack(side="left", padx=10)

    # ------------------------------------------------------------------
    # QUEUE
    # ------------------------------------------------------------------
    def _build_queue(self):
        frame = ttk.Frame(self.tab_queue)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Queue Table
        columns = ("story", "premise", "status", "progress")
        self.queue_tree = ttk.Treeview(frame, columns=columns, show="headings")
        self.queue_tree.heading("story", text="Story ID")
        self.queue_tree.heading("premise", text="Premise")
        self.queue_tree.heading("status", text="Status")
        self.queue_tree.heading("progress", text="Progress")
        self.queue_tree.pack(fill="both", expand=True)

        # Queue Controls
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=10)
        ttk.Button(ctrl, text="Refresh Queue", command=self._refresh_queue).pack(side="left", padx=5)
        ttk.Button(ctrl, text="Regenerate Prompts", command=self._regent_prompts).pack(side="left", padx=5)

    # ------------------------------------------------------------------
    # PUBLISHING
    # ------------------------------------------------------------------
    def _build_publishing(self):
        frame = ttk.Frame(self.tab_pub)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Calendar/List
        self.pub_tree = ttk.Treeview(frame, columns=("story", "date", "time", "status"), show="headings")
        self.pub_tree.heading("story", text="Story")
        self.pub_tree.heading("date", text="Date")
        self.pub_tree.heading("time", text="Time")
        self.pub_tree.heading("status", text="Status")
        self.pub_tree.pack(fill="both", expand=True)

        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=10)
        ttk.Button(ctrl, text="⬆ Upload All", command=self._upload_all).pack(side="left", padx=5)
        ttk.Button(ctrl, text="📅 Re-schedule All", command=self._reschedule).pack(side="left", padx=5)

    # ------------------------------------------------------------------
    # INSPECTOR
    # ------------------------------------------------------------------
    def _build_inspector(self):
        frame = ttk.Frame(self.tab_insp)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Label(frame, text="Select a story from the Queue to inspect details...").pack(pady=100)

    # ------------------------------------------------------------------
    # LIMITS & ACCOUNTS
    # ------------------------------------------------------------------
    def _build_limits(self):
        frame = ttk.Frame(self.tab_limits)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.limit_text = tk.Text(frame, height=10, width=80, font=("Consolas", 10), state="disabled")
        self.limit_text.pack(pady=10)
        
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=10)
        ttk.Button(ctrl, text="🔐 Open Login Tabs", command=self._open_login).pack(side="left", padx=5)
        ttk.Button(ctrl, text="🔄 Clear Cooldowns", command=self._clear_limits).pack(side="left", padx=5)

    # ------------------------------------------------------------------
    # SETTINGS
    # ------------------------------------------------------------------
    def _build_settings(self):
        frame = ttk.Frame(self.tab_settings)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Label(frame, text="Pipeline and Upload settings will be implemented here.").pack(pady=20)

    # ------------------------------------------------------------------
    # LOGS
    # ------------------------------------------------------------------
    def _build_logs(self):
        frame = ttk.Frame(self.tab_logs)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.log_text = scrolledtext.ScrolledText(frame, height=30, state="disabled", wrap="word", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_config("stdout", foreground="#e0e0e0")
        self.log_text.tag_config("stderr", foreground="#ff9999")
        self.log_text.tag_config("error", foreground="#ff6666", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("info", foreground="#66ccff", font=("Consolas", 9, "bold"))

        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=5)
        ttk.Button(ctrl, text="🗑 Clear Log", command=self._clear_logs).pack(side="left", padx=5)

    # ------------------------------------------------------------------
    # LOGIC & LOOP
    # ------------------------------------------------------------------
    def _poll_log_queue(self):
        try:
            while True:
                tag, msg = self.log_queue.get_nowait()
                self.log_text.config(state="normal")
                self.log_text.insert(tk.END, msg, tag)
                self.log_text.see(tk.END)
                self.log_text.config(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _update_dashboard_loop(self):
        status = read_status()
        if status:
            self.cur_story_var.set(status.get("current_story", "Idle"))
            self.cur_stage_var.set(status.get("stage", "N/A"))
            p = status.get("pending", 0)
            c = status.get("completed", 0)
            self.batch_stat_var.set(f"Batch Progress: {c} / {p + c}")
            self.prog_bar["value"] = (c / (p + c) * 100) if (p + c) > 0 else 0
        
        self.after(1000, self._update_dashboard_loop)

    def _start_factory(self):
        # Launch batch_generator as a background process
        self.start_factory_btn.config(state="disabled")
        self.stop_factory_btn.config(state="normal")
        
        # Note: in a production GUI, we'd use a more robust process manager
        import subprocess
        subprocess.Popen([
            sys.executable, "batch_generator.py", 
            "--count", "30", "--per-day", "3", "--kids", 
            "--voices-first", "--max-wait-minutes", "0", 
            "--upload-each", "--upload"
        ], cwd=PROJECT_ROOT, creationflags=subprocess.CREATE_NO_WINDOW)
        
        self.log_queue.put(("info", "Factory started in background...\n"))

    def _stop_factory(self):
        # Simplified stop (usually involves killing process group)
        self.log_queue.put(("error", "Stop signal sent (requires process group management).\n"))
        self.start_factory_btn.config(state="normal")
        self.stop_factory_btn.config(state="disabled")

    def _upload_all(self):
        self.login_runner = UploadRunner(self.log_queue, date.today().isoformat(), "18:00")
        self.login_runner.start()

    def _open_login(self):
        self.login_runner = LoginRunner(self.log_queue)
        self.login_runner.start()

    def _clear_logs(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")

    def _refresh_queue(self):
        pass

    def _regent_prompts(self):
        pass

    def _reschedule(self):
        pass

    def _clear_limits(self):
        pass

if __name__ == "__main__":
    app = App()
    app.mainloop()
