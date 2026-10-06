import json
import shutil
import subprocess
import time
from pathlib import Path

import requests


class OllamaVideoPlanner:
    """Generate story title and visual plans with a local Ollama server.

    Camera shake is intentionally disabled at the source: the planner will
    always return shake='none'. The renderer also enforces this independently.
    """

    def __init__(self, base_url="http://127.0.0.1:11434", model="gemma4:31b-cloud"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.chat_url = f"{self.base_url}/api/chat"
        self.tags_url = f"{self.base_url}/api/tags"
        self._ensure_server()

    # ------------------------------------------------------------
    # OLLAMA SERVER
    # ------------------------------------------------------------

    def _ensure_server(self):
        try:
            response = requests.get(self.tags_url, timeout=5)
            response.raise_for_status()
            print("Ollama server is already running.")
            return
        except Exception:
            pass

        executable = shutil.which("ollama")
        if executable is None:
            common = [
                Path(r"C:\Program Files\Ollama\ollama.exe"),
                Path(r"C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe"),
            ]
            for candidate in common:
                candidate = Path(str(candidate).replace("%USERNAME%", Path.home().name))
                if candidate.exists():
                    executable = str(candidate)
                    break

        if executable is None:
            raise RuntimeError(
                "Ollama is not running and ollama.exe could not be found."
            )

        subprocess.Popen(
            [executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                response = requests.get(self.tags_url, timeout=3)
                response.raise_for_status()
                print("Ollama server started.")
                return
            except Exception:
                time.sleep(1)

        raise RuntimeError("Ollama server did not become available in time.")

    # ------------------------------------------------------------
    # REQUESTS
    # ------------------------------------------------------------

    @staticmethod
    def _extract_json(text):
        text = text.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                return json.loads(text[start : end + 1])

            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end > start:
                return json.loads(text[start : end + 1])

            raise

    def _chat(self, messages, temperature=0.2, timeout=300, retries=4):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        last_error = None

        for attempt in range(1, retries + 1):
            try:
                response = requests.post(
                    self.chat_url,
                    json=payload,
                    timeout=timeout,
                )

                if response.status_code in {502, 503, 504}:
                    raise requests.HTTPError(
                        f"Temporary Ollama gateway error: {response.status_code}",
                        response=response,
                    )

                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]

            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    break
                wait = min(2 ** (attempt - 1), 8)
                print(
                    f"Ollama request failed ({attempt}/{retries}): {exc}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        raise RuntimeError(f"Ollama request failed: {last_error}")

    # ------------------------------------------------------------
    # TITLE
    # ------------------------------------------------------------

    def generate_title(self, clips):
        story_text = "\n\n".join(
            f"Clip {clip.id}: {clip.moment}\nNarration: {clip.script}"
            for clip in clips
        )

        prompt = f"""
Generate one short cinematic title for this story.

Requirements:
- 2 to 8 words.
- Memorable.
- Specific to the story.
- No quotation marks.
- No subtitle.
- No emojis.
- Return JSON only.

JSON format:
{{
  "title": "..."
}}

Story:
{story_text}
""".strip()

        raw = self._chat(
            [
                {
                    "role": "system",
                    "content": "You are a professional film title writer. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )

        data = self._extract_json(raw)
        title = str(data.get("title", "")).strip()
        return title or "Untitled"

    # ------------------------------------------------------------
    # VISUAL PLAN
    # ------------------------------------------------------------

    def plan_clips(self, clips):
        clip_payload = []
        for clip in clips:
            clip_payload.append(
                {
                    "id": clip.id,
                    "characters": clip.characters,
                    "moment": clip.moment,
                    "vibe": clip.vibe,
                    "style": clip.style,
                    "script": clip.script,
                }
            )

        prompt = f"""
Create a visual editing plan for every clip below.

Return JSON ONLY as a list.

Allowed values:
- camera: wide cinematic shot, medium cinematic shot, close-up, extreme close-up, over-the-shoulder shot, low angle, high angle
- composition: centered, rule of thirds, foreground framing, symmetrical, leading lines, deep composition
- motion: static, zoom_in, zoom_out, push_in, pull_out, pan_left, pan_right, pan_up, pan_down
- transition_after: cut, fade_black, flash_white
- grade: neutral, cinematic, moody, cool, warm

IMPORTANT:
- Camera shake is completely disabled.
- Set shake to exactly "none" for EVERY clip.
- Do not invent any other shake value.
- Use motion only for smooth cinematic movement, never vibration.
- Keep character and environment continuity.

JSON item format:
{{
  "id": 1,
  "camera": "medium cinematic shot",
  "composition": "rule of thirds",
  "motion": "static",
  "shake": "none",
  "transition_after": "cut",
  "grade": "cinematic"
}}

Clips:
{json.dumps(clip_payload, ensure_ascii=False, indent=2)}
""".strip()

        raw = self._chat(
            [
                {
                    "role": "system",
                    "content": "You are a cinematic video editor. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        data = self._extract_json(raw)
        if not isinstance(data, list):
            raise RuntimeError("Ollama visual plan must be a JSON list.")

        allowed_motion = {
            "static",
            "zoom_in",
            "zoom_out",
            "push_in",
            "pull_out",
            "pan_left",
            "pan_right",
            "pan_up",
            "pan_down",
        }
        allowed_transition = {"cut", "fade_black", "flash_white"}
        allowed_grade = {"neutral", "cinematic", "moody", "cool", "warm"}

        plans = {}
        for item in data:
            clip_id = int(item["id"])
            motion = str(item.get("motion", "static")).strip().lower()
            transition = str(item.get("transition_after", "cut")).strip().lower()
            grade = str(item.get("grade", "cinematic")).strip().lower()

            if motion not in allowed_motion:
                motion = "static"
            if transition not in allowed_transition:
                transition = "cut"
            if grade not in allowed_grade:
                grade = "cinematic"

            plans[clip_id] = {
                "camera": str(item.get("camera", "medium cinematic shot")),
                "composition": str(item.get("composition", "rule of thirds")),
                "motion": motion,
                # HARD DISABLED.
                "shake": "none",
                "transition_after": transition,
                "grade": grade,
            }

        missing = [clip.id for clip in clips if clip.id not in plans]
        if missing:
            raise RuntimeError(
                f"Ollama visual plan is missing clip IDs: {missing}"
            )

        return plans

    # ------------------------------------------------------------
    # THUMBNAIL PLAN
    # ------------------------------------------------------------

    def generate_thumbnail_plan(self, clips, title):
        story_text = "\n\n".join(
            f"Clip {clip.id}: {clip.moment}\n{clip.script}"
            for clip in clips
        )

        prompt = f"""
Create a cinematic thumbnail plan for the story below.

Title: {title}

Return JSON ONLY with exactly these fields:
- subject
- scene
- emotion
- camera
- lighting
- color_palette
- composition
- title_position

Allowed title_position values:
- top_left
- top_right
- bottom_left
- bottom_right

Story:
{story_text}
""".strip()

        raw = self._chat(
            [
                {
                    "role": "system",
                    "content": "You are a professional cinematic thumbnail art director. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        data = self._extract_json(raw)
        title_position = str(data.get("title_position", "top_left")).lower()
        if title_position not in {
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        }:
            title_position = "top_left"

        return {
            "subject": str(data.get("subject", "Main character")),
            "scene": str(data.get("scene", "A dramatic cinematic scene")),
            "emotion": str(data.get("emotion", "mystery")),
            "camera": str(data.get("camera", "Wide cinematic shot")),
            "lighting": str(data.get("lighting", "dramatic cinematic lighting")),
            "color_palette": str(data.get("color_palette", "deep cinematic tones")),
            "composition": str(data.get("composition", "strong foreground/background separation")),
            "title_position": title_position,
        }
