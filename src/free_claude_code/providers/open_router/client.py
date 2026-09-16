"""OpenRouter provider implementation."""

import json

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.diagnostics import extract_upstream_error_detail
from free_claude_code.core.reasoning import ReasoningEffort
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.model_listing import extract_tool_capable_model_infos
from free_claude_code.providers.openai_chat import (
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    ReasoningObject,
    validate_extra_body_does_not_override_canonical_fields,
)
from free_claude_code.providers.reasoning_compatibility import (
    reasoning_control_rejected,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="OPENROUTER",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    include_extra_body=True,
    extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
)


class OpenRouterChatBehavior(OpenAIChatBehavior):
    """OpenRouter Chat adaptation without HTTP ownership."""

    def reasoning_disable_rejected(self, error: Exception) -> bool:
        detail = extract_upstream_error_detail(error)
        if detail.status_code not in {None, 200}:
            return False
        try:
            payload = json.loads(detail.body_text or "")
        except ValueError:
            return False
        if not isinstance(payload, dict):
            return False
        error_body = payload.get("error", payload)
        if not isinstance(error_body, dict):
            return False
        code = error_body.get("code")
        return (
            isinstance(code, int)
            and code == 400
            and reasoning_control_rejected(error_body, ("reasoning",))
        )


class OpenRouterProvider(OpenAIChatProvider):
    """OpenRouter provider using the OpenAI-compatible Chat Completions API."""

    def __init__(
        self, config: ProviderConfig, *, admission: ProviderAdmissionController
    ):
        super().__init__(
            config,
            behavior=OpenRouterChatBehavior(_PROFILE),
            admission=admission,
        )

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Advertise OpenRouter tool models with reasoning capability metadata."""
        payload = await self._list_models_payload()
        return extract_tool_capable_model_infos(
            payload, provider_name=self._provider_name
        )


_PROFILE = OpenAIChatProfile(
    _REQUEST_POLICY,
    ReasoningObject(tuple((effort, effort.value) for effort in ReasoningEffort)),
    structured_reasoning_details=True,
)
