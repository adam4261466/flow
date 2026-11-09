"""Pluggable vision backends used by the image and thumbnail critics.

Order of preference in ``auto`` mode:

1. An explicitly configured Ollama vision model (``--vision-model``).
2. An OpenAI-compatible API when ``OPENAI_API_KEY`` is set.
3. The already-authenticated Gemini web session (zero extra install).
4. An automatically detected local Ollama vision model.
5. ``NullVision`` (technical checks only).

A critic failure is never fatal.  ``main.py`` catches it and continues with
the image/voice it already has.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import requests

from agents.text import clamp_score, extract_json_object

CRITIQUE_PROMPT = """You are a strict visual continuity supervisor for a cinematic story video.

Evaluate {subject}.

Creative brief:
{brief}

Check, in this order:
1. Character identity: same face, hair, age, body type and costume as the brief.
2. Environment and props: same location, key objects, time of day and weather.
3. Composition: clear subject, well framed, matching the requested camera and composition.
4. Lighting and colour: consistent with the established palette.
5. Artifacts: extra limbs or fingers, warped faces, unreadable text, watermarks, borders, collage, blur.

Return JSON only, no markdown:
{{"score": 0-10, "issues": ["short issue", "..."], "prompt_fix": "concise additions to the image prompt that would fix the issues"}}

Scoring: 10 = flawless, 8-9 = good enough to publish, 7 = acceptable, 5-6 = noticeable problems, 0-4 = unusable.
Do not generate an image. Do not write an essay.
"""

TEXT_PROMPT = """Transcribe the exact title text rendered inside {subject}.

Rules:
- Return every visible title word exactly as spelled, including punctuation.
- Do not translate, correct, paraphrase or add words.
- Ignore credits, watermarks, logos and UI text.
- If there is no title text, return an empty string.

Return JSON only: {{"text": "..."}}
Do not generate an image.
"""


@dataclass
class Critique:
    score: float
    issues: list[str] = field(default_factory=list)
    prompt_fix: str = ""
    raw: str = ""
    backend: str = ""


class VisionBackend:
    name = "none"
    available = False

    def critique_image(
        self,
        image_path,
        brief: str,
        *,
        same_conversation: bool = True,
    ) -> Critique:
        raise NotImplementedError

    def read_text(
        self,
        image_path,
        *,
        same_conversation: bool = True,
    ) -> str:
        raise NotImplementedError


class NullVision(VisionBackend):
    name = "none"
    available = False

    def critique_image(self, image_path, brief, *, same_conversation=True):
        return Critique(10.0, [], "", "", self.name)

    def read_text(self, image_path, *, same_conversation=True):
        return ""


def _parse_critique(raw: str, backend: str) -> Critique:
    data = extract_json_object(raw)
    if not data:
        return Critique(
            score=7.0,
            issues=["critic returned an unparseable response"],
            prompt_fix="",
            raw=str(raw)[:2000],
            backend=backend,
        )

    issues = data.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    issues = [str(item).strip() for item in issues if str(item).strip()]
    fix = str(data.get("prompt_fix", "") or "").strip()

    return Critique(
        score=clamp_score(data.get("score"), default=5.0),
        issues=issues[:8],
        prompt_fix=fix,
        raw=str(raw)[:2000],
        backend=backend,
    )


def _parse_text(raw: str) -> str:
    data = extract_json_object(raw)
    if data is not None:
        value = data.get("text", "")
        return str(value or "").strip()
    # Some models answer with the title itself instead of JSON.
    return str(raw or "").strip().strip('"').strip()


class OllamaVision(VisionBackend):
    name = "ollama"
    available = False

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5vl:7b",
        timeout: int = 180,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.available = self.is_available()

    def _model_names(self) -> list[str]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        return [
            str(item.get("name", ""))
            for item in response.json().get("models", [])
        ]

    def is_available(self) -> bool:
        try:
            names = self._model_names()
        except Exception:  # noqa: BLE001
            return False

        base = self.model.split(":")[0]
        return any(name == self.model or name.split(":")[0] == base for name in names)

    def _chat(self, prompt: str, image_path=None) -> str:
        message = {"role": "user", "content": prompt}
        if image_path is not None:
            image_path = Path(image_path)
            message["images"] = [
                base64.b64encode(image_path.read_bytes()).decode("ascii")
            ]

        payload = {
            "model": self.model,
            "messages": [message],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return str(response.json().get("message", {}).get("content", ""))

    def critique_image(self, image_path, brief, *, same_conversation=True):
        prompt = CRITIQUE_PROMPT.format(
            subject="the attached image", brief=brief
        )
        raw = self._chat(prompt, image_path)
        return _parse_critique(raw, self.name)

    def read_text(self, image_path, *, same_conversation=True):
        prompt = TEXT_PROMPT.format(subject="the attached image")
        return _parse_text(self._chat(prompt, image_path))


class OpenAIVision(VisionBackend):
    name = "openai"
    available = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: int = 180,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.available = bool(self.api_key)

    def _chat(self, prompt: str, image_path) -> str:
        image_path = Path(image_path)
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded}"
                            },
                        },
                    ],
                }
            ],
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return str(
            response.json()["choices"][0]["message"].get("content", "")
        )

    def critique_image(self, image_path, brief, *, same_conversation=True):
        prompt = CRITIQUE_PROMPT.format(
            subject="the attached image", brief=brief
        )
        return _parse_critique(self._chat(prompt, image_path), self.name)

    def read_text(self, image_path, *, same_conversation=True):
        prompt = TEXT_PROMPT.format(subject="the attached image")
        return _parse_text(self._chat(prompt, image_path))


class GeminiWebVision(VisionBackend):
    """Use the existing Chrome/Gemini tab as a critic.

    When ``same_conversation`` is true, the follow-up prompt runs in the same
    Gemini conversation that just generated the image, so the image does not
    need to be uploaded again.  Otherwise the image is attached to a fresh
    conversation.
    """

    name = "gemini_web"
    available = False

    def __init__(self, browser, timeout: int = 180) -> None:
        self.browser = browser
        self.timeout = timeout
        self.available = browser is not None

    def _ask(self, prompt: str, image_path, same_conversation: bool) -> str:
        if self.browser is None:
            raise RuntimeError("Gemini browser is not connected")
        return self.browser.ask_text(
            prompt,
            new_conversation=not same_conversation,
            reference_image=None if same_conversation else image_path,
            timeout_seconds=self.timeout,
        )

    def critique_image(self, image_path, brief, *, same_conversation=True):
        subject = (
            "the image you just generated in this conversation"
            if same_conversation
            else "the attached image"
        )
        prompt = CRITIQUE_PROMPT.format(subject=subject, brief=brief)
        raw = self._ask(prompt, image_path, same_conversation)
        return _parse_critique(raw, self.name)

    def read_text(self, image_path, *, same_conversation=True):
        subject = (
            "the image you just generated in this conversation"
            if same_conversation
            else "the attached image"
        )
        prompt = TEXT_PROMPT.format(subject=subject)
        raw = self._ask(prompt, image_path, same_conversation)
        return _parse_text(raw)


def detect_ollama_vision_model(base_url: str) -> str | None:
    """Return the first locally installed model likely to accept images."""

    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        names = [
            str(item.get("name", ""))
            for item in response.json().get("models", [])
        ]
    except Exception:  # noqa: BLE001
        return None

    markers = (
        "llava",
        "vision",
        "qwen2.5vl",
        "qwen2-vl",
        "minicpm-v",
        "moondream",
        "gemma3",
    )
    for name in names:
        lowered = name.lower()
        if any(marker in lowered for marker in markers):
            return name
    return None


def build_vision_backend(config, gemini=None, log=print) -> VisionBackend:
    """Choose a critic backend from config, environment and availability."""

    if not getattr(config, "enabled", True):
        return NullVision()

    backend = str(getattr(config, "vision_backend", "auto")).lower()
    if backend == "none":
        return NullVision()

    if backend == "gemini":
        if gemini is None:
            log("[agents] Gemini critic requested but no browser is attached.")
            return NullVision()
        log("[agents] vision backend: Gemini web (existing Chrome session)")
        return GeminiWebVision(gemini)

    if backend == "ollama":
        model = config.vision_model or detect_ollama_vision_model(
            config.ollama_url
        )
        if not model:
            log(
                "[agents] Ollama critic requested but no vision model was found. "
                "Pull one with: ollama pull qwen2.5vl:7b"
            )
            return NullVision()
        vision = OllamaVision(config.ollama_url, model)
        if not vision.available:
            log(f"[agents] Ollama model {model!r} is not installed.")
            return NullVision()
        log(f"[agents] vision backend: Ollama ({model})")
        return vision

    if backend == "openai":
        vision = OpenAIVision(
            base_url=config.openai_base_url,
            model=config.openai_model,
        )
        if not vision.available:
            log("[agents] OPENAI_API_KEY is not set; OpenAI critic disabled.")
            return NullVision()
        log(f"[agents] vision backend: OpenAI ({config.openai_model})")
        return vision

    # auto
    if config.vision_model:
        vision = OllamaVision(config.ollama_url, config.vision_model)
        if vision.available:
            log(
                f"[agents] vision backend: Ollama ({config.vision_model})"
            )
            return vision

    if os.environ.get("OPENAI_API_KEY"):
        vision = OpenAIVision(
            base_url=config.openai_base_url,
            model=config.openai_model,
        )
        if vision.available:
            log(f"[agents] vision backend: OpenAI ({config.openai_model})")
            return vision

    if gemini is not None:
        log("[agents] vision backend: Gemini web (existing Chrome session)")
        return GeminiWebVision(gemini)

    model = detect_ollama_vision_model(config.ollama_url)
    if model:
        vision = OllamaVision(config.ollama_url, model)
        if vision.available:
            log(f"[agents] vision backend: Ollama ({model})")
            return vision

    log(
        "[agents] no semantic vision backend available; "
        "technical checks only (set --vision-model or OPENAI_API_KEY)."
    )
    return NullVision()
