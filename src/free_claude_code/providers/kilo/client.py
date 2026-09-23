"""Kilo.ai provider implementation."""

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.reasoning import ReasoningEffort
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    ReasoningObject,
    validate_extra_body_does_not_override_canonical_fields,
)

from .models import extract_kilo_model_infos

_PROFILE = OpenAIChatProfile(
    OpenAIChatRequestPolicy(
        provider_name="KILO",
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
        include_extra_body=True,
        extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
    ),
    ReasoningObject(tuple((effort, effort.value) for effort in ReasoningEffort)),
    reasoning_delta_field="reasoning",
    structured_reasoning_details=True,
)


class KiloProvider(OpenAIChatProvider):
    """Kilo gateway adapter with capability-aware discovery and reasoning."""

    def __init__(
        self, config: ProviderConfig, *, admission: ProviderAdmissionController
    ) -> None:
        super().__init__(config, profile=_PROFILE, admission=admission)

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Advertise Kilo chat models that can execute FCC agent requests."""
        payload = await self._list_models_payload()
        return extract_kilo_model_infos(
            payload,
            provider_name=self._provider_name,
        )
