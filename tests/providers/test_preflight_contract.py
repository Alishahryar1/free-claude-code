"""The shared OpenAI-chat provider owns explicit request preflight."""

from collections.abc import AsyncIterator, Mapping
from unittest.mock import MagicMock

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from tests.providers.support import immediate_admission, make_provider_config


class RecordingChatBehavior(OpenAIChatBehavior):
    def __init__(self) -> None:
        super().__init__(
            OpenAIChatProfile(
                OpenAIChatRequestPolicy("TEST", ReasoningReplayMode.DISABLED),
                NO_REASONING,
            )
        )
        self.build_calls: list[tuple[MessagesRequest, ReasoningPolicy]] = []

    def build_messages_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict:
        self.build_calls.append((request, reasoning))
        return {}


class ProviderWithoutPreflight(BaseProvider):
    async def cleanup(self) -> None:
        return None

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        return frozenset()

    async def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


def test_provider_base_requires_an_explicit_preflight_implementation() -> None:
    with pytest.raises(TypeError, match="preflight_messages"):
        ProviderWithoutPreflight(
            make_provider_config(api_key="test", base_url="https://test.invalid")
        )


def test_openai_provider_owns_preflight() -> None:
    assert OpenAIChatProvider.preflight_messages is not BaseProvider.preflight_messages


def test_provider_preflight_calls_builder_and_preserves_policy() -> None:
    behavior = RecordingChatBehavior()
    provider = OpenAIChatProvider(
        make_provider_config(api_key="test", base_url="https://test.invalid"),
        behavior=behavior,
        admission=immediate_admission(),
        client=MagicMock(),
    )
    request = MessagesRequest(
        model="test-model",
        messages=[Message(role="user", content="hello")],
    )

    provider.preflight_messages(request, reasoning=ReasoningPolicy.off())

    assert behavior.build_calls == [(request, ReasoningPolicy.off())]
