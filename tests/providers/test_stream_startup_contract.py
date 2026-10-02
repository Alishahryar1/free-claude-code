"""The shared OpenAI-chat provider owns request conversion during stream construction."""

from unittest.mock import MagicMock

import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.core.recovery import RecoveryCheckpoint
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
)


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
        preserve_features: bool = False,
    ) -> dict:
        self.build_calls.append((request, reasoning))
        return {}


@pytest.mark.asyncio
async def test_provider_preparation_calls_builder_and_preserves_policy() -> None:
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

    async with provider.open_messages(
        request, reasoning=ReasoningPolicy.off()
    ) as candidate:
        await candidate.prepare(RecoveryCheckpoint("messages"))

    assert behavior.build_calls == [(request, ReasoningPolicy.off())]
