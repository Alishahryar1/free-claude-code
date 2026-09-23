"""Mistral La Plateforme provider implementation (OpenAI-compatible chat completions)."""

from collections.abc import Mapping
from typing import Any

from loguru import logger

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    OpenAIModelListing,
)

from .reasoning import (
    apply_mistral_reasoning_request_shape,
    clone_body_without_mistral_reasoning,
    is_mistral_reasoning_rejection,
    normalize_mistral_stream,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="MISTRAL",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
)
_PROFILE = OpenAIChatProfile(
    _REQUEST_POLICY,
    NO_REASONING,
    model_listing=OpenAIModelListing(
        input_modality_boolean_paths=(
            (
                ModelInputModality.TEXT,
                ("capabilities", "completion_chat"),
            ),
            (ModelInputModality.IMAGE, ("capabilities", "vision")),
        ),
        context_window_tokens_path=("max_context_length",),
    ),
)


class MistralChatBehavior(OpenAIChatBehavior):
    """Mistral Chat adaptation without HTTP ownership."""

    @property
    def reasoning_off_fields(self) -> tuple[tuple[str, ...], ...]:
        return (("reasoning_effort",),)

    def finalize_chat_body(
        self,
        body: dict[str, Any],
        *,
        reasoning: ReasoningPolicy,
    ) -> dict:
        apply_mistral_reasoning_request_shape(body, reasoning=reasoning)
        return body

    def retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Retry once without Mistral reasoning fields when a model rejects them."""
        if not is_mistral_reasoning_rejection(error):
            return None
        retry_body = clone_body_without_mistral_reasoning(body)
        if retry_body is None:
            return None
        logger.warning(
            "MISTRAL_STREAM: retrying without reasoning after upstream rejection"
        )
        return retry_body

    def normalize_stream(self, stream: Any, _body: Mapping[str, Any]) -> Any:
        return normalize_mistral_stream(stream)


class MistralProvider(OpenAIChatProvider):
    """Mistral API using ``https://api.mistral.ai/v1/chat/completions``."""

    def __init__(
        self, config: ProviderConfig, *, admission: ProviderAdmissionController
    ):
        super().__init__(
            config,
            behavior=MistralChatBehavior(_PROFILE),
            admission=admission,
        )
