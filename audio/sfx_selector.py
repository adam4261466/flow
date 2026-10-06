from pathlib import Path

import json
import re
import requests

from audio.sfx_library import SFXLibrary


class SFXSelector:
    """Select a clean SFX from the whole filename-based library."""

    BAD_SOUND_WORDS = {
        "static",
        "interference",
        "glitch",
        "distortion",
        "distorted",
        "crackle",
        "crackling",
        "clipping",
        "clip",
        "noise",
        "digitalnoise",
        "packetloss",
        "radioerror",
        "corrupt",
        "corrupted",
    }

    def __init__(
        self,
        sfx_directory,
        base_url="http://127.0.0.1:11434",
        model="gemma4:31b-cloud",
    ):
        self.library = SFXLibrary(sfx_directory)
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _chat(self, prompt, timeout=300):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You select realistic cinematic sound effects. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }

        last = None
        for attempt in range(1, 5):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=timeout,
                )
                if response.status_code in {502, 503, 504}:
                    raise requests.HTTPError(
                        f"Temporary Ollama error {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                return response.json()["message"]["content"]
            except Exception as exc:
                last = exc
                if attempt == 4:
                    break
                import time
                time.sleep(min(2 ** (attempt - 1), 8))

        raise RuntimeError(f"Ollama SFX request failed: {last}")

    @staticmethod
    def _parse_json(text):
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
            raise

    @classmethod
    def _looks_dirty(cls, path):
        normalized = re.sub(r"[_\-.]+", " ", str(path).lower())
        tokens = set(normalized.split())
        return bool(tokens.intersection(cls.BAD_SOUND_WORDS))

    def _clean_candidates(self, candidates):
        return [p for p in candidates if not self._looks_dirty(p)]

    def select_for_clip(self, clip):
        prompt = f"""
Analyze this scene and suggest 5 to 8 concrete sound-effect keywords.

Clip moment:
{clip.moment}

Narration:
{clip.script}

Rules:
- Select sounds that would naturally exist in the scene.
- Prefer physical Foley and environmental ambience.
- Avoid music.
- Avoid voice/message/radio processing sounds.
- Do NOT request or prefer static, interference, glitch, distortion,
  crackle, clipping, corrupted audio, digital noise, or scratch effects
  unless the script explicitly requires a broken electronic signal.
- Use concise searchable phrases.

Return JSON only:
{{
  "keywords": ["keyword 1", "keyword 2", "keyword 3"]
}}
""".strip()

        data = self._parse_json(self._chat(prompt))
        keywords = data.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(k).strip() for k in keywords if str(k).strip()]

        candidates = self.library.search_by_tags(keywords, limit=30)
        candidates = self._clean_candidates(candidates)

        if not candidates:
            return {
                "keywords": keywords,
                "candidates": [],
                "selected": None,
            }

        candidate_text = "\n".join(
            f"{index + 1}. {str(path)}"
            for index, path in enumerate(candidates)
        )

        selection_prompt = f"""
Choose the SINGLE best sound-effect file for this scene.

Scene:
{clip.moment}

Narration:
{clip.script}

Candidates:
{candidate_text}

Rules:
- Choose exactly one candidate from the list.
- Prefer a natural realistic effect or ambience.
- Avoid processed/glitchy/static/crackly/noisy effects.
- Return JSON only.

JSON format:
{{
  "selected": "exact candidate path"
}}
""".strip()

        selected_data = self._parse_json(self._chat(selection_prompt))
        selected = str(selected_data.get("selected", "")).strip()

        candidate_map = {str(path): path for path in candidates}
        selected_path = candidate_map.get(selected)

        if selected_path is None:
            # Case-insensitive exact-path recovery.
            lowered = {key.lower(): value for key, value in candidate_map.items()}
            selected_path = lowered.get(selected.lower())

        return {
            "keywords": keywords,
            "candidates": [str(path) for path in candidates],
            "selected": selected_path,
        }
