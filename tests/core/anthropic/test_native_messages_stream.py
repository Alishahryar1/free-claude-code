"""Native relay validation never rewrites identities or completes a truncated stream."""

from collections.abc import AsyncIterator
from typing import cast

import pytest

from free_claude_code.core.anthropic.native import NativeMessagesError
from free_claude_code.core.anthropic.native_stream import (
    NativeMessagesRelay,
    NativeMessagesStreamState,
)
from free_claude_code.core.anthropic.sse_aggregation import (
    aggregate_anthropic_sse_to_message,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_lines
from free_claude_code.core.json_types import JsonObject, JsonValue

_START: JsonObject = {
    "type": "message_start",
    "message": {
        "id": "upstream-id",
        "type": "message",
        "model": "upstream",
        "role": "assistant",
        "content": [],
        "usage": {"input_tokens": 3, "output_tokens": 0},
        "native_extension": "kept",
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", [None, []])
async def test_native_citations_survive_block_completion_and_fragmented_aggregation(
    initial: JsonValue,
) -> None:
    citation: JsonObject = {
        "type": "char_location",
        "cited_text": "source",
        "document_index": 0,
        "start_char_index": 0,
        "end_char_index": 6,
        "document_title": "doc",
    }
    events: list[JsonObject] = [
        _START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "answer", "citations": initial},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "citations_delta", "citation": citation},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]
    state = NativeMessagesStreamState()
    completed = None
    for event in events:
        block = state.accept(cast(str, event["type"]), event)
        if block is not None:
            completed = block
    assert completed is not None and completed["citations"] == [citation]
    relay = NativeMessagesRelay(public_model="public")

    async def stream() -> AsyncIterator[str]:
        for event in events:
            chunk = relay.feed(cast(str, event["type"]), event)
            for start in range(0, len(chunk), 3):
                yield chunk[start : start + 3]

    message, error, complete = await aggregate_anthropic_sse_to_message(stream())
    assert complete and error is None
    assert message["content"][0]["citations"] == [citation]


def test_relay_preserves_native_identity_and_extensions_without_mutating_input() -> (
    None
):
    relay = NativeMessagesRelay(public_model="public")
    result = parse_sse_lines(relay.feed("message_start", _START).splitlines())[0].data
    assert result["message"]["model"] == "public"
    assert result["message"]["id"] == "upstream-id"
    assert result["message"]["native_extension"] == "kept"
    assert cast(JsonObject, _START["message"])["model"] == "upstream"
    block: JsonObject = {
        "type": "content_block_start",
        "index": 5,
        "content_block": {"type": "text", "text": "hello", "citations": []},
    }
    assert (
        parse_sse_lines(relay.feed("content_block_start", block).splitlines())[0].data
        == block
    )
    assert not relay.completed


@pytest.mark.asyncio
async def test_fragmented_native_relay_preserves_nonstreaming_thinking_and_usage() -> (
    None
):
    relay = NativeMessagesRelay(public_model="public")
    events: list[JsonObject] = [
        _START,
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "signature_delta", "signature": "sig-"},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "signature_delta", "signature": "end"},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 8},
        },
        {"type": "message_stop"},
    ]

    async def stream() -> AsyncIterator[str]:
        for event in events:
            chunk = relay.feed(cast(str, event["type"]), event)
            for start in range(0, len(chunk), 5):
                yield chunk[start : start + 5]

    message, error, complete = await aggregate_anthropic_sse_to_message(stream())
    assert relay.completed and complete and error is None
    assert message["id"] == "upstream-id" and message["model"] == "public"
    assert message["content"] == [
        {"type": "thinking", "thinking": "", "signature": "sig-end"}
    ]
    assert message["usage"] == {"input_tokens": 3, "output_tokens": 8}


@pytest.mark.parametrize(
    "event",
    [
        _START,
        {"type": "message_stop"},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "orphan"},
        },
        {
            "type": "content_block_start",
            "index": True,
            "content_block": {"type": "text", "text": ""},
        },
        {"type": "message_delta", "delta": {"stop_reason": "unknown"}},
    ],
)
def test_invalid_lifecycles_are_not_reported_as_success(event: JsonObject) -> None:
    relay = NativeMessagesRelay(public_model="public")
    relay.feed("message_start", _START)
    with pytest.raises(NativeMessagesError):
        relay.feed(cast(str, event["type"]), event)
    assert not relay.completed


def test_reused_block_indexes_and_events_after_terminal_are_rejected() -> None:
    relay = NativeMessagesRelay(public_model="public")
    relay.feed("message_start", _START)
    start: JsonObject = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    relay.feed("content_block_start", start)
    relay.feed("content_block_stop", {"type": "content_block_stop", "index": 0})
    with pytest.raises(NativeMessagesError, match="Duplicate"):
        relay.feed("content_block_start", start)
    relay.feed(
        "message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
    )
    relay.feed("message_stop", {"type": "message_stop"})
    with pytest.raises(NativeMessagesError, match="after message_stop"):
        relay.feed("message_stop", {"type": "message_stop"})


@pytest.mark.parametrize("partial", ['{"x":', "[]", '{"x":NaN}'])
def test_invalid_json_tools_never_complete(partial: str) -> None:
    relay = NativeMessagesRelay(public_model="public")
    relay.feed("message_start", _START)
    relay.feed(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "c",
                "name": "lookup",
                "input": {},
            },
        },
    )
    relay.feed(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": partial},
        },
    )
    with pytest.raises(NativeMessagesError, match="arguments"):
        relay.feed("content_block_stop", {"type": "content_block_stop", "index": 0})
    assert not relay.completed
