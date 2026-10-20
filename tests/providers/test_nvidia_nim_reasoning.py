"""NIM named reasoning across both supported ingress protocols."""

import pytest

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from tests.providers.test_history_transports import _harness, _saved_reply
from tests.providers.test_nvidia_nim import _alias_provider, _alias_request


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    "policy,expected",
    [
        *((ReasoningPolicy(effort=effort), effort.value) for effort in ReasoningEffort),
        (ReasoningPolicy.off(), "none"),
        (ReasoningPolicy.on(), "high"),
        (ReasoningPolicy.on(budget_tokens=2048), "high"),
        (ReasoningPolicy.provider_default(), None),
    ],
)
async def test_named_effort_reaches_nim_without_numeric_budget(wire, policy, expected):
    def responder(bodies):
        body = bodies[-1]
        assert body.get("reasoning_effort") == expected
        assert "reasoning_budget" not in body
        assert "chat_template_kwargs" not in body
        return 200, [
            {
                "id": "response",
                "object": "chat.completion.chunk",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ]

    async with _harness("chat", responder, chat_provider_factory=_alias_provider) as (
        _,
        bodies,
        provider,
    ):
        stream = (
            provider.stream_messages
            if wire == "messages"
            else provider.stream_responses
        )
        reply = await _saved_reply(stream(_alias_request(wire), reasoning=policy), wire)

    assert reply
    assert len(bodies) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    "policy", [ReasoningPolicy.off(), ReasoningPolicy(effort=ReasoningEffort.MAX)]
)
async def test_unsupported_named_effort_is_not_silently_remapped(wire, policy):
    def responder(bodies):
        return 400, {
            "message": "reasoning_effort must be one of low, medium, high",
            "param": "reasoning_effort",
        }

    async with _harness("chat", responder, chat_provider_factory=_alias_provider) as (
        _,
        bodies,
        provider,
    ):
        stream = (
            provider.stream_messages
            if wire == "messages"
            else provider.stream_responses
        )
        with pytest.raises(ExecutionFailure, match="reasoning_effort"):
            await _saved_reply(stream(_alias_request(wire), reasoning=policy), wire)

    assert len(bodies) == 1
