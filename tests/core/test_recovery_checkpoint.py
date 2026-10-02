"""Continuation preserves the request and drops only proven full-prefix replays."""

import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.passthrough import NativeMessagesRequest
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest
from free_claude_code.core.recovery import RecoveryCheckpoint
from free_claude_code.core.recovery_request import continue_request
from free_claude_code.core.stream_events import ExactPrefixFilter


@pytest.mark.parametrize(
    ("prefix", "chunks", "expected"),
    [
        ("abc", ["a", "b", "c", "def"], "def"),
        ("abc", ["abc"], ""),
        ("abc", ["ab", "d"], "abd"),
        ("abc", ["ab"], "ab"),
        ("hello ", [" world"], " world"),
        ("", ["hello"], "hello"),
    ],
)
def test_only_a_complete_exact_prefix_is_removed(
    prefix: str, chunks: list[str], expected: str
) -> None:
    matching = ExactPrefixFilter(prefix)
    result = "".join(matching.feed(chunk) for chunk in chunks) + matching.finish()
    assert result == expected


def test_messages_continuation_preserves_tools_and_original_request() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "selected",
            "max_tokens": 32000,
            "system": "Keep the instructions",
            "messages": [{"role": "user", "content": "Do the work"}],
            "tools": [{"name": "read", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
        }
    )
    before = request.model_dump()
    checkpoint = RecoveryCheckpoint("messages", ({"type": "text", "text": "First"},))

    continued = continue_request(request, checkpoint)

    assert request.model_dump() == before
    assert len(continued.messages) == 3
    for field in ("tools", "tool_choice", "system", "max_tokens"):
        assert getattr(continued, field) == getattr(request, field)
    assert continued.messages[1].model_dump()["content"][0]["text"] == "First"


def test_native_continuation_retains_unknown_request_fields() -> None:
    request = NativeMessagesRequest(
        {
            "model": "native",
            "messages": [{"role": "user", "content": "Question"}],
            "future_control": {"opaque": [1, 2, 3]},
            "tools": [{"type": "future_native_tool", "name": "work"}],
        }
    )
    checkpoint = RecoveryCheckpoint("messages", ({"type": "text", "text": "First"},))
    continued = continue_request(request, checkpoint)
    assert continued.body["future_control"] == request.body["future_control"]
    assert continued.body["tools"] == request.body["tools"]
    assert isinstance(request.body["messages"], list)
    assert len(request.body["messages"]) == 1
    assert isinstance(continued.body["messages"], list)
    assert len(continued.body["messages"]) == 3


def test_responses_continuation_preserves_controls_and_uses_materialized_output() -> (
    None
):
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "selected",
            "input": "Question",
            "instructions": "Use the tools",
            "tools": [{"type": "function", "name": "read", "parameters": {}}],
            "text": {"format": {"type": "json_object"}},
            "max_output_tokens": 32000,
        }
    )
    checkpoint = RecoveryCheckpoint(
        "responses",
        (
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "First"}],
            },
        ),
    )
    continued = continue_request(request, checkpoint)
    assert request.input == "Question"
    assert isinstance(continued.input, list)
    assert len(continued.input) == 3
    for field in ("tools", "instructions", "text", "max_output_tokens"):
        assert getattr(continued, field) == getattr(request, field)
