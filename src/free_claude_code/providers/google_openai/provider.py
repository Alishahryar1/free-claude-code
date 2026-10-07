"""Shared Google behavior for OpenAI-compatible Gemini endpoints."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import Any

from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    ChatStreamOutput,
    OpenAIAsyncCredentialProvider,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatProvider,
)

from .thought_signatures import apply_google_thought_signatures

_MAX_TOOL_CALL_EXTRA_CONTENT_CACHE = 4096


class GoogleChatBehavior(OpenAIChatBehavior):
    """Google Chat adaptation without HTTP ownership."""

    def __init__(self, profile: OpenAIChatProfile) -> None:
        super().__init__(profile)
        self._tool_call_extra_content_by_id: dict[str, dict[str, Any]] = {}

    def record_tool_call_extra_content(
        self, tool_call_id: str, extra_content: dict[str, Any]
    ) -> None:
        if (
            tool_call_id not in self._tool_call_extra_content_by_id
            and len(self._tool_call_extra_content_by_id)
            >= _MAX_TOOL_CALL_EXTRA_CONTENT_CACHE
        ):
            self._tool_call_extra_content_by_id.pop(
                next(iter(self._tool_call_extra_content_by_id))
            )
        self._tool_call_extra_content_by_id[tool_call_id] = deepcopy(extra_content)

    def finalize_chat_body(
        self,
        body: dict[str, Any],
        *,
        reasoning: ReasoningPolicy,
    ) -> dict[str, Any]:
        apply_google_thought_signatures(
            body,
            tool_call_extra_content_by_id=self._tool_call_extra_content_by_id,
        )
        return body

    def extra_reasoning_events(
        self, delta: Any, output: ChatStreamOutput, *, output_reasoning: bool
    ) -> Iterator[str]:
        extra = getattr(delta, "extra_content", None)
        google = extra.get("google") if isinstance(extra, Mapping) else None
        signature = (
            google.get("thought_signature") if isinstance(google, Mapping) else None
        )
        if isinstance(signature, str) and signature:
            output.defer_opaque_reasoning(signature)
        return iter(())


class GoogleOpenAIProvider(OpenAIChatProvider):
    """Shared thought-signature and request behavior for Google Gemini APIs."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        profile: OpenAIChatProfile,
        admission: ProviderAdmissionController,
        api_key_provider: OpenAIAsyncCredentialProvider | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            config,
            behavior=GoogleChatBehavior(profile),
            admission=admission,
            api_key_provider=api_key_provider,
            default_headers=default_headers,
        )
