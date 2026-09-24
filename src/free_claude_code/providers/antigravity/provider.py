"""Native Google Antigravity provider for FCC."""

from __future__ import annotations

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.reasoning import ReasoningEffort
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat.provider import OpenAIChatProvider
from free_claude_code.providers.openai_chat.profiles import OpenAIChatProfile
from free_claude_code.providers.openai_chat.reasoning import NamedEffortReasoning
from free_claude_code.providers.openai_chat.request_policy import OpenAIChatRequestPolicy

from .auth import AntigravityAuthManager
from .chat_adapter import AntigravityChatAdapter
from .client import AntigravityClient

_ANTIGRAVITY_EFFORTS = (
    (ReasoningEffort.MINIMAL, "low"),
    (ReasoningEffort.LOW, "low"),
    (ReasoningEffort.MEDIUM, "medium"),
    (ReasoningEffort.HIGH, "high"),
    (ReasoningEffort.XHIGH, "high"),
    (ReasoningEffort.MAX, "high"),
)

ANTIGRAVITY_PROFILE = OpenAIChatProfile(
    request_policy=OpenAIChatRequestPolicy(
        provider_name="ANTIGRAVITY",
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
    ),
    reasoning=NamedEffortReasoning(
        efforts=_ANTIGRAVITY_EFFORTS,
        disabled_value="off",
        field="reasoning_effort",
    ),
    reasoning_delta_field="reasoning_content",
    reasoning_delta_fallback_field="reasoning",
)


class AntigravityProvider(OpenAIChatProvider):
    """Reuse FCC's mature chat conversion around a Cloud Code-backed adapter."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        auth: AntigravityAuthManager,
        admission: ProviderAdmissionController,
    ) -> None:
        cloud_code = AntigravityClient(auth=auth, base_url=config.base_url)
        adapter = AntigravityChatAdapter(cloud_code, base_url=config.base_url)
        self._cloud_code = cloud_code
        super().__init__(
            config,
            profile=ANTIGRAVITY_PROFILE,
            admission=admission,
            client=adapter,  # type: ignore[arg-type]
        )

    async def cleanup(self) -> None:
        await self._cloud_code.close()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        payload = await self._cloud_code.fetch_available_models()
        models = payload.get("models")
        if not isinstance(models, dict):
            return frozenset()
        return frozenset(
            ProviderModelInfo(model_id=model_id)
            for model_id in models
            if isinstance(model_id, str) and model_id
        )
