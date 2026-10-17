"""Token Plan Chat Completions with the documented Spark model catalog."""

from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningCapability
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    ThinkingObjectReasoning,
)

_PROFILE = OpenAIChatProfile(
    OpenAIChatRequestPolicy(
        provider_name="IFLYTEK_TOKEN_PLAN",
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    ),
    ThinkingObjectReasoning(
        enabled={"type": "enabled"},
        disabled={"type": "disabled"},
    ),
)


class IFlytekTokenPlanProvider(OpenAIChatProvider):
    """Reuse chat transport without querying an unavailable models endpoint."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        admission: ProviderAdmissionController,
        client: AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(config, profile=_PROFILE, admission=admission, client=client)

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """List the native Spark models documented for Token Plan."""
        return frozenset(
            ProviderModelInfo(
                model_id,
                supports_thinking=True,
                input_modalities=frozenset({ModelInputModality.TEXT}),
                reasoning_capability=ReasoningCapability.OPTIONAL,
            )
            for model_id in ("spark-x2.5", "xsparkx2flash")
        )
