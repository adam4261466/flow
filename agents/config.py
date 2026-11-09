"""Configuration for the Level-2 critic layer.

Every setting can be overridden by a CLI flag (see ``main.py``) or an
environment variable.  Defaults are conservative: critics never block work
they cannot judge, and every loop has a small retry budget.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

VALID_BACKENDS = {"auto", "gemini", "ollama", "openai", "none"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


@dataclass
class AgentConfig:
    """Runtime settings for image/thumbnail/voice critic loops."""

    enabled: bool = True

    # auto: pick the best available backend (Gemini web, OpenAI, or local
    # Ollama vision).  gemini/ollama/openai force one.  none disables critics.
    vision_backend: str = "auto"
    vision_model: str | None = None

    # An image scoring at or above this is accepted.  Gemini's own critique is
    # a guide, not an oracle, so 7 is intentionally forgiving.
    image_threshold: float = 7.0

    max_image_retries: int = 2
    max_thumbnail_retries: int = 2
    max_voice_retries: int = 1

    # Hard ceiling on semantic critic calls per story.  Technical health
    # checks (file size, duration, silence) do not count against this.
    max_critic_calls: int = 60

    # Existing, already-accepted images are not re-critiqued by default.
    critique_existing: bool = False

    ollama_url: str = "http://127.0.0.1:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    @classmethod
    def from_args(cls, args) -> "AgentConfig":
        """Build from argparse namespaces without requiring every flag."""

        def arg(name, default, cast):
            value = getattr(args, name, None)
            if value is None:
                return default
            return cast(value)

        enabled = not bool(getattr(args, "no_agents", False))
        enabled = _env_flag("FLOW_AGENTS", enabled)

        backend = (
            arg("vision_backend", None, str)
            or os.environ.get("FLOW_VISION_BACKEND")
            or "auto"
        ).strip().lower()
        if backend not in VALID_BACKENDS:
            raise ValueError(
                "vision_backend must be one of: "
                + ", ".join(sorted(VALID_BACKENDS))
                + f" (got {backend!r})"
            )

        model = arg("vision_model", None, str) or os.environ.get(
            "FLOW_VISION_MODEL"
        )
        if model is not None:
            model = str(model).strip() or None

        threshold = arg(
            "image_threshold",
            _env_float("FLOW_IMAGE_THRESHOLD", 7.0),
            float,
        )
        threshold = max(0.0, min(10.0, threshold))

        return cls(
            enabled=enabled,
            vision_backend=backend,
            vision_model=model,
            image_threshold=threshold,
            max_image_retries=max(
                0,
                min(
                    10,
                    arg(
                        "max_image_retries",
                        _env_int("FLOW_MAX_IMAGE_RETRIES", 2),
                        int,
                    ),
                ),
            ),
            max_thumbnail_retries=max(
                0,
                min(
                    10,
                    arg(
                        "max_thumbnail_retries",
                        _env_int("FLOW_MAX_THUMBNAIL_RETRIES", 2),
                        int,
                    ),
                ),
            ),
            max_voice_retries=max(
                0,
                min(
                    10,
                    arg(
                        "max_voice_retries",
                        _env_int("FLOW_MAX_VOICE_RETRIES", 1),
                        int,
                    ),
                ),
            ),
            max_critic_calls=max(
                0,
                arg(
                    "max_critic_calls",
                    _env_int("FLOW_MAX_CRITIC_CALLS", 60),
                    int,
                ),
            ),
            critique_existing=_env_flag("FLOW_CRITIQUE_EXISTING", False),
            ollama_url=os.environ.get(
                "FLOW_OLLAMA_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            openai_base_url=os.environ.get(
                "FLOW_OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/"),
            openai_model=os.environ.get(
                "FLOW_OPENAI_MODEL", "gpt-4o-mini"
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        if not self.enabled:
            return "critic loops disabled"
        backend = self.vision_backend
        if self.vision_model:
            backend += f" ({self.vision_model})"
        return (
            f"critic loops on: vision={backend}, "
            f"image_threshold={self.image_threshold:g}, "
            f"retries=image:{self.max_image_retries}"
            f"/thumb:{self.max_thumbnail_retries}"
            f"/voice:{self.max_voice_retries}, "
            f"critic_budget={self.max_critic_calls}"
        )
