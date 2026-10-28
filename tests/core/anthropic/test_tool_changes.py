"""Claude Code deferred-tool announcements survive every Messages ingress path."""

from copy import deepcopy

import pytest

from free_claude_code.core.anthropic.conversion import (
    OpenAIConversionError,
    build_base_request_body,
)
from free_claude_code.core.anthropic.models import MessagesRequest, TokenCountRequest
from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    NativeMessagesOptions,
    build_native_messages_request,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses.errors import ResponsesConversionError
from free_claude_code.core.openai_responses.provider_input import (
    build_responses_provider_request,
)
from free_claude_code.core.reasoning import ReasoningPolicy

# The first request Claude Code sends when it announces a deferred tool.
ANNOUNCEMENT: JsonObject = {
    "role": "system",
    "content": [
        {"type": "text", "text": "Session context"},
        {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "old"}},
        {
            "type": "tool_addition",
            "tool": {"type": "tool_reference", "name": "lookup"},
            "cache_control": {"type": "ephemeral"},
        },
    ],
}
LOOKUP_TOOL: JsonObject = {
    "name": "lookup",
    "description": "Look something up",
    "input_schema": {"type": "object", "properties": {}},
    "defer_loading": True,
}
INLINE_DEFINITION: JsonObject = {
    "role": "system",
    "content": [
        {
            "type": "tool_addition",
            "tool": {"type": "tool_definition", "definition": LOOKUP_TOOL},
        }
    ],
}


def _request(*messages: JsonObject) -> MessagesRequest:
    return MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}, *messages],
            "tools": [LOOKUP_TOOL],
        }
    )


def test_messages_and_token_count_requests_accept_tool_changes() -> None:
    request = _request(ANNOUNCEMENT)
    count = TokenCountRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}, ANNOUNCEMENT]}
    )

    for parsed in (request, count):
        blocks = parsed.messages[1].content
        assert isinstance(blocks, list)
        assert [block.type for block in blocks] == [
            "text",
            "tool_removal",
            "tool_addition",
        ]


def test_chat_conversion_drops_reference_changes_and_keeps_declared_tool() -> None:
    body = build_base_request_body(_request(ANNOUNCEMENT))

    assert "tool_addition" not in str(body["messages"])
    assert "tool_removal" not in str(body["messages"])
    assert "Session context" in str(body["messages"])
    assert [tool["function"]["name"] for tool in body["tools"]] == ["lookup"]


def test_chat_conversion_rejects_inline_tool_definition() -> None:
    with pytest.raises(OpenAIConversionError, match="inline tool definition"):
        build_base_request_body(_request(INLINE_DEFINITION))


def test_responses_conversion_drops_reference_changes() -> None:
    body = build_responses_provider_request(
        _request(ANNOUNCEMENT), reasoning=ReasoningPolicy.provider_default()
    )

    assert body["instructions"] == "Session context"
    assert [tool["name"] for tool in body["tools"]] == ["lookup"]


def test_responses_conversion_rejects_inline_tool_definition() -> None:
    with pytest.raises(ResponsesConversionError, match="inline tool definition"):
        build_responses_provider_request(
            _request(INLINE_DEFINITION), reasoning=ReasoningPolicy.provider_default()
        )


@pytest.mark.parametrize("announcement", [ANNOUNCEMENT, INLINE_DEFINITION])
def test_native_body_keeps_tool_change_message_in_place(
    announcement: JsonObject,
) -> None:
    prepared = build_native_messages_request(
        _request(announcement), options=NativeMessagesOptions("m", 1024)
    )

    assert prepared.body["messages"] == [
        {"role": "user", "content": "hi"},
        deepcopy(announcement),
    ]
    assert "system" not in prepared.body


def test_native_body_still_requires_a_conversational_turn() -> None:
    request = MessagesRequest.model_validate({"model": "m", "messages": [ANNOUNCEMENT]})

    with pytest.raises(NativeMessagesError, match="conversational turn"):
        build_native_messages_request(request, options=NativeMessagesOptions("m", 1024))
