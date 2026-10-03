import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import requests


OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "gemma4:31b-cloud",
)


class OllamaSFX:

    def __init__(
        self,
        url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        timeout=120,
        startup_timeout=30,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.startup_timeout = startup_timeout

        self.ollama_process = None

        self.ensure_server()

    # =========================================================
    # OLLAMA SERVER
    # =========================================================

    def is_server_running(self):
        try:
            response = requests.get(
                f"{self.url}/api/tags",
                timeout=2,
            )

            return response.status_code == 200

        except requests.RequestException:
            return False

    def find_ollama(self):
        ollama = shutil.which("ollama")

        if ollama:
            return Path(ollama)

        possible_paths = [
            Path(
                os.environ.get(
                    "LOCALAPPDATA",
                    "",
                )
            )
            / "Programs"
            / "Ollama"
            / "ollama.exe",

            Path(
                r"C:\Program Files\Ollama\ollama.exe"
            ),

            Path(
                r"C:\Program Files (x86)\Ollama\ollama.exe"
            ),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            "Could not find Ollama.\n\n"
            "Make sure Ollama is installed."
        )

    def start_server(self):
        ollama_path = self.find_ollama()

        print("Starting Ollama server...")

        creation_flags = 0

        if os.name == "nt":
            creation_flags = subprocess.CREATE_NO_WINDOW

        self.ollama_process = subprocess.Popen(
            [
                str(ollama_path),
                "serve",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )

        print("Ollama process started.")

    def wait_for_server(self):
        print("Waiting for Ollama...")

        deadline = (
            time.time()
            + self.startup_timeout
        )

        while time.time() < deadline:

            if self.is_server_running():
                print("Ollama server is ready.")
                return

            time.sleep(0.5)

        raise RuntimeError(
            "Ollama was started, but its API "
            "did not become available.\n\n"
            f"Expected:\n{self.url}"
        )

    def ensure_server(self):
        if self.is_server_running():
            print("Ollama server is already running.")
            return

        print("Ollama server is not running.")

        self.start_server()
        self.wait_for_server()

    # =========================================================
    # JSON PARSER
    # =========================================================

    def _parse_json_response(self, content):
        """
        Parse JSON even when the model wraps it in Markdown.

        Handles:

            {"keywords": [...]}

        and:

            ```json
            {"keywords": [...]}
            ```
        """

        if not content:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        text = content.strip()

        # -----------------------------------------------------
        # First try the raw response.
        # -----------------------------------------------------

        try:
            return json.loads(text)

        except json.JSONDecodeError:
            pass

        # -----------------------------------------------------
        # Remove Markdown code fences.
        # -----------------------------------------------------

        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if fenced_match:
            json_text = fenced_match.group(1).strip()

            try:
                return json.loads(json_text)

            except json.JSONDecodeError:
                pass

        # -----------------------------------------------------
        # Try extracting the first JSON object.
        # -----------------------------------------------------

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:

            json_text = text[start:end + 1]

            try:
                return json.loads(json_text)

            except json.JSONDecodeError:
                pass

        raise RuntimeError(
            "Ollama returned a response that could not "
            "be parsed as JSON.\n\n"
            f"Response:\n{content}"
        )

    # =========================================================
    # CHAT
    # =========================================================

    def _chat(
        self,
        system_prompt,
        user_prompt,
    ):
        if not self.is_server_running():
            self.ensure_server()

        try:
            response = requests.post(
                f"{self.url}/api/chat",

                json={
                    "model": self.model,
                    "stream": False,

                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                },

                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.RequestException as error:

            raise RuntimeError(
                "Could not communicate with Ollama.\n\n"
                f"URL:\n{self.url}\n\n"
                f"Model:\n{self.model}\n\n"
                f"Error:\n{error}"
            ) from error

        try:
            data = response.json()

        except ValueError as error:

            raise RuntimeError(
                "Ollama returned invalid HTTP JSON."
            ) from error

        message = data.get(
            "message",
            {},
        )

        content = message.get(
            "content",
            "",
        )

        if not content:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return content.strip()

    # =========================================================
    # GENERATE SFX KEYWORDS
    # =========================================================

    def generate_keywords(self, clip):

        system_prompt = """
You select keywords for searching a sound-effect library.

Your task is NOT to describe the whole scene.

Your task is to produce short, concrete sound concepts
that are likely to appear in sound-effect filenames.

Good:
["bell", "church bell", "chime"]

Bad:
["mysterious", "cinematic", "dark atmosphere"]

Rules:
- Focus only on sounds that could actually be heard.
- Prefer concrete nouns.
- Prefer common words.
- Avoid abstract emotions.
- Maximum 8 keywords.

Return ONLY a JSON object.
Do not use Markdown.
Do not wrap the JSON in ```.

Required format:

{
    "keywords": [
        "keyword1",
        "keyword2"
    ]
}
"""

        user_prompt = f"""
Analyze this video clip.

Characters:
{clip.characters}

Moment:
{clip.moment}

Vibe:
{clip.vibe}

Style:
{clip.style}

Script:
{clip.script}

Generate sound-effect search keywords.
"""

        content = self._chat(
            system_prompt,
            user_prompt,
        )

        data = self._parse_json_response(
            content
        )

        keywords = data.get(
            "keywords",
            [],
        )

        if not isinstance(
            keywords,
            list,
        ):
            raise RuntimeError(
                "Ollama SFX keywords are not a list."
            )

        return [
            str(keyword).strip()
            for keyword in keywords
            if str(keyword).strip()
        ]

    # =========================================================
    # CHOOSE SFX
    # =========================================================

    def choose_sound(
        self,
        clip,
        candidate_files,
    ):

        if not candidate_files:
            return None

        candidates_text = "\n".join(
            f"- {path}"
            for path in candidate_files
        )

        system_prompt = """
You choose the best sound effect from a provided list.

CRITICAL RULE:

You may ONLY select a filename that appears EXACTLY
in the candidate list.

Do not invent filenames.
Do not modify filenames.
Do not change capitalization.
Do not create new filenames.

Return ONLY a JSON object.
Do not use Markdown.
Do not wrap the JSON in ```.

Required format:

{
    "selected": "EXACT_CANDIDATE_PATH"
}

If none is appropriate:

{
    "selected": null
}
"""

        user_prompt = f"""
Video clip:

Characters:
{clip.characters}

Moment:
{clip.moment}

Vibe:
{clip.vibe}

Style:
{clip.style}

Script:
{clip.script}

Candidate sound effects:

{candidates_text}

Choose the single best candidate.
"""

        content = self._chat(
            system_prompt,
            user_prompt,
        )

        data = self._parse_json_response(
            content
        )

        selected = data.get(
            "selected"
        )

        if selected is None:
            return None

        selected = str(
            selected
        ).strip()

        if selected not in candidate_files:

            raise RuntimeError(
                "Ollama selected a file that was "
                "not in the candidate list.\n\n"
                f"Selected:\n{selected}\n\n"
                "Candidates:\n"
                + "\n".join(
                    candidate_files
                )
            )

        return selected