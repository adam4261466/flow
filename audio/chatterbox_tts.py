"""Client for the local Chatterbox TTS Server.

This replaces the old Voicebox backend. Chatterbox exposes a single
synchronous endpoint (POST /tts) which returns the rendered audio bytes,
so a generation is just one request instead of start/poll/copy.

The client can also launch the server itself when it is not already
running, then wait until the model has finished loading.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import requests


DEFAULT_SERVER_URL = "http://127.0.0.1:8004"

DEFAULT_SERVER_DIR = Path(
    r"C:\Users\kabli\OneDrive\Desktop\projects\tts"
    r"\Chatterbox-TTS-Server-windows-easyInstallation"
)

DEFAULT_VOICE = (
    "ElevenLabs_2026-11-03T20_43_25_Amara_-_Calm__Elegant_and_"
    "Refined_pvc_sp95_s20_sb9_v4.mp3"
)

# Chatterbox's "Standard Narration" preset.
STANDARD_NARRATION = {
    "temperature": 0.8,
    "exaggeration": 0.4,
    "cfg_weight": 0.5,
    "seed": 42,
    "speed_factor": 1.0,
}


class ChatterboxTTS:
    """Generate speech with the local Chatterbox TTS Server."""

    def __init__(
        self,
        server=DEFAULT_SERVER_URL,
        voice=DEFAULT_VOICE,
        voice_mode="predefined",
        reference_audio_filename=None,
        language="en",
        server_dir=DEFAULT_SERVER_DIR,
        auto_start=True,
        startup_timeout=900,
        request_timeout=1800,
        poll_interval=2,
        split_text=True,
        chunk_size=120,
        temperature=STANDARD_NARRATION["temperature"],
        exaggeration=STANDARD_NARRATION["exaggeration"],
        cfg_weight=STANDARD_NARRATION["cfg_weight"],
        seed=STANDARD_NARRATION["seed"],
        speed_factor=STANDARD_NARRATION["speed_factor"],
    ):
        self.server = os.environ.get(
            "CHATTERBOX_SERVER_URL",
            server,
        ).rstrip("/")

        self.server_dir = Path(
            os.environ.get(
                "CHATTERBOX_SERVER_DIR",
                server_dir,
            )
        )

        self.voice = os.environ.get(
            "CHATTERBOX_VOICE",
            voice,
        )
        self.voice_mode = voice_mode
        self.reference_audio_filename = reference_audio_filename
        self.language = language

        self.auto_start = auto_start
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval

        self.split_text = split_text
        self.chunk_size = chunk_size

        self.temperature = temperature
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.seed = seed
        self.speed_factor = speed_factor

        if self.voice_mode not in ("predefined", "clone"):
            raise ValueError(
                "voice_mode must be 'predefined' or 'clone'."
            )

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _url(self, path):
        return f"{self.server}{path}"

    def is_server_running(self):
        """Return True when the Chatterbox API is reachable and ready."""

        try:
            response = requests.get(
                self._url("/openapi.json"),
                timeout=3,
            )
        except requests.RequestException:
            return False

        if response.status_code != 200:
            return False

        try:
            paths = response.json().get("paths", {})
        except ValueError:
            return False

        return "/tts" in paths

    def _server_python(self):
        if os.name == "nt":
            candidate = (
                self.server_dir
                / "venv"
                / "Scripts"
                / "python.exe"
            )
        else:
            candidate = (
                self.server_dir
                / "venv"
                / "bin"
                / "python"
            )

        if candidate.is_file():
            return candidate

        print(
            "WARNING: Chatterbox virtual environment not found. "
            f"Falling back to: {sys.executable}"
        )
        return Path(sys.executable)

    def start_server(self):
        """Launch server.py in its own console window."""

        server_script = self.server_dir / "server.py"

        if not server_script.is_file():
            raise RuntimeError(
                "Chatterbox server script was not found:\n"
                f"{server_script}"
            )

        python = self._server_python()

        environment = os.environ.copy()
        environment.setdefault("PYTHONUNBUFFERED", "1")

        # Do not pop a browser window while the video pipeline is running.
        environment.setdefault("CHATTERBOX_NO_BROWSER", "1")

        kwargs = {
            "cwd": str(self.server_dir),
            "env": environment,
        }

        if os.name == "nt":
            kwargs["creationflags"] = getattr(
                subprocess,
                "CREATE_NEW_CONSOLE",
                0,
            )
        else:
            kwargs["start_new_session"] = True

        print("Starting Chatterbox TTS server...")
        print(f"  Python: {python}")
        print(f"  Folder: {self.server_dir}")

        subprocess.Popen(
            [str(python), str(server_script)],
            **kwargs,
        )

    def ensure_server_running(self):
        """Start the server if needed and wait until it is ready."""

        if self.is_server_running():
            return True

        if not self.auto_start:
            raise RuntimeError(
                "Chatterbox TTS server is not reachable at "
                f"{self.server}. Start it with:\n"
                f"{self.server_dir / 'win-run.bat'}"
            )

        self.start_server()

        print(
            f"Waiting for Chatterbox TTS server at {self.server} "
            "(the model can take a few minutes to load)..."
        )

        deadline = time.monotonic() + self.startup_timeout
        next_notice = time.monotonic() + 15

        while time.monotonic() < deadline:
            if self.is_server_running():
                print("Chatterbox TTS server is ready.")
                return True

            now = time.monotonic()

            if now >= next_notice:
                remaining = max(0, int(deadline - now))
                print(
                    "  still loading the model... "
                    f"({remaining}s left)"
                )
                next_notice = now + 15

            time.sleep(self.poll_interval)

        raise RuntimeError(
            "Chatterbox TTS server did not become ready within "
            f"{self.startup_timeout} seconds. Check its console window "
            "or the server log:\n"
            f"{self.server_dir / 'logs' / 'tts_server.log'}"
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _build_payload(self, text, overrides):
        payload = {
            "text": text,
            "voice_mode": self.voice_mode,
            "output_format": "wav",
            "split_text": self.split_text,
            "chunk_size": self.chunk_size,
            "language": self.language,
        }

        if self.voice_mode == "predefined":
            if not self.voice:
                raise ValueError(
                    "A predefined voice filename is required."
                )
            payload["predefined_voice_id"] = self.voice
        else:
            if not self.reference_audio_filename:
                raise ValueError(
                    "reference_audio_filename is required in clone mode."
                )
            payload["reference_audio_filename"] = (
                self.reference_audio_filename
            )

        for key in (
            "temperature",
            "exaggeration",
            "cfg_weight",
            "seed",
            "speed_factor",
        ):
            value = getattr(self, key)

            if value is not None:
                payload[key] = value

        payload.update(
            {
                key: value
                for key, value in overrides.items()
                if value is not None
            }
        )

        return payload

    def generate_bytes(self, text, **overrides):
        """Return generated WAV audio as bytes."""

        if not text or not text.strip():
            raise ValueError(
                "Cannot generate speech from empty text."
            )

        self.ensure_server_running()

        payload = self._build_payload(text, overrides)

        try:
            response = requests.post(
                self._url("/tts"),
                json=payload,
                timeout=(10, self.request_timeout),
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not reach Chatterbox TTS server at "
                f"{self.server}: {exc}"
            ) from exc

        if response.status_code != 200:
            try:
                detail = response.json().get(
                    "detail",
                    response.text,
                )
            except ValueError:
                detail = response.text

            raise RuntimeError(
                "Chatterbox TTS request failed "
                f"({response.status_code}): {detail}"
            )

        audio = response.content

        if not audio or len(audio) < 100:
            raise RuntimeError(
                "Chatterbox TTS returned empty or invalid audio."
            )

        return audio

    def generate_to_file(self, text, output_file, **overrides):
        """Generate speech and save it to output_file."""

        output_file = Path(output_file)
        audio = self.generate_bytes(text, **overrides)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_file = output_file.with_name(
            output_file.name + ".part"
        )

        try:
            temp_file.write_bytes(audio)
            os.replace(temp_file, output_file)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

        print("Chatterbox audio saved to:")
        print(output_file.resolve())

        return output_file
