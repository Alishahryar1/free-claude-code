"""The shared OpenAI-chat provider owns request conversion during stream construction."""

from unittest.mock import MagicMock

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
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


def test_provider_stream_constructor_calls_builder_and_preserves_policy() -> None:
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

    provider.stream_messages(request, reasoning=ReasoningPolicy.off())

    assert behavior.build_calls == [(request, ReasoningPolicy.off())]
