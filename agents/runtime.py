"""Bundle config + state + vision backend for the pipeline hooks."""

from __future__ import annotations

from agents.config import AgentConfig
from agents.critics import ImageCritic, ThumbnailCritic, VoiceCritic
from agents.state import ProductionState
from agents.vision import build_vision_backend


class AgentRuntime:
    def __init__(
        self,
        config: AgentConfig,
        state: ProductionState | None,
        *,
        gemini=None,
        log=print,
    ) -> None:
        self.config = config
        self.state = state
        self.gemini = gemini
        self.log = log
        if self.state is not None:
            # The CLI/env setting is authoritative for this run.
            self.state.budget.critic_limit = config.max_critic_calls
        self.vision = build_vision_backend(config, gemini=gemini, log=log)

    @classmethod
    def create(
        cls,
        config: AgentConfig,
        state: ProductionState | None,
        *,
        gemini=None,
        log=print,
    ) -> "AgentRuntime":
        return cls(config, state, gemini=gemini, log=log)

    def attach_gemini(self, gemini) -> None:
        """Rebuild the vision backend after a new browser session connects."""

        self.gemini = gemini
        if self.config.vision_backend in {"auto", "gemini"}:
            self.vision = build_vision_backend(
                self.config, gemini=gemini, log=self.log
            )

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def image_critic(self) -> ImageCritic:
        return ImageCritic(
            self.vision,
            state=self.state,
            threshold=self.config.image_threshold,
        )

    def thumbnail_critic(self, title: str) -> ThumbnailCritic:
        return ThumbnailCritic(
            self.vision,
            expected_title=title,
            state=self.state,
        )

    def voice_critic(self) -> VoiceCritic:
        return VoiceCritic()

    def describe(self) -> str:
        mode = "on" if self.enabled else "off"
        vision = self.vision.name if self.vision else "none"
        available = getattr(self.vision, "available", False)
        return f"critic loops {mode}; vision={vision} (available={available})"
