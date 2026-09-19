"""Native Messages events preserve Responses identity, replay and completion."""

from collections.abc import Mapping
from typing import cast

import pytest

from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    NativeMessagesOptions,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_lines
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    decode_replay,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_responses import (
    AnthropicToResponsesStream,
    OpenAIResponsesRequest,
    build_responses_messages_request,
)

_SCOPE = "github_copilot/anthropic_messages"


def _stream(*, tools: list[JsonObject] | None = None) -> AnthropicToResponsesStream:
    request = OpenAIResponsesRequest(model="public", input="hi", tools=tools)
    prepared = build_responses_messages_request(
        request, options=NativeMessagesOptions("upstream", 4096)
    )
    return AnthropicToResponsesStream(
        request,
        public_model="public",
        tool_identities=prepared.tool_identities,
        replay_origin=ReplayOrigin(_SCOPE, "messages", "", "", "upstream"),
    )


def _feed(
    stream: AnthropicToResponsesStream, kind: str, **fields: JsonValue
) -> list[JsonObject]:
    chunks = stream.feed(kind, {"type": kind, **fields})
    return [
        cast(JsonObject, event.data)
        for event in parse_sse_lines("".join(chunks).splitlines())
    ]


def _start(stream: AnthropicToResponsesStream) -> list[JsonObject]:
    return _feed(
        stream,
        "message_start",
        message={
            "id": "native-id",
            "type": "message",
            "role": "assistant",
            "model": "upstream",
            "content": [],
            "usage": {
                "input_tokens": 11,
                "cache_read_input_tokens": 7,
                "cache_creation_input_tokens": 5,
                "output_tokens": 0,
            },
        },
    )


def _end(
    stream: AnthropicToResponsesStream,
    reason: str = "end_turn",
    *,
    usage: JsonValue = None,
) -> list[JsonObject]:
    events = _feed(
        stream,
        "message_delta",
        delta={"stop_reason": reason, "stop_sequence": None},
        usage=usage if usage is not None else {"output_tokens": 19},
    )
    assert not stream.completed
    events += _feed(stream, "message_stop")
    return events


def test_text_thinking_and_redacted_blocks_keep_order_replay_and_exact_usage() -> None:
    stream = _stream()
    events = _start(stream)
    events += _feed(
        stream,
        "content_block_start",
        index=3,
        content_block={"type": "thinking", "thinking": "", "extension": {"x": 1}},
    )
    events += _feed(
        stream,
        "content_block_delta",
        index=3,
        delta={"type": "thinking_delta", "thinking": "think"},
    )
    events += _feed(
        stream,
        "content_block_delta",
        index=3,
        delta={"type": "signature_delta", "signature": "sig-"},
    )
    events += _feed(
        stream,
        "content_block_delta",
        index=3,
        delta={"type": "signature_delta", "signature": "end"},
    )
    events += _feed(stream, "content_block_stop", index=3)
    events += _feed(
        stream,
        "content_block_start",
        index=5,
        content_block={"type": "redacted_thinking", "data": "opaque"},
    )
    events += _feed(stream, "content_block_stop", index=5)
    events += _feed(
        stream,
        "content_block_start",
        index=9,
        content_block={"type": "text", "text": "Hello "},
    )
    events += _feed(
        stream,
        "content_block_delta",
        index=9,
        delta={"type": "text_delta", "text": "world"},
    )
    events += _feed(stream, "content_block_stop", index=9)
    events += _end(
        stream,
        usage={
            "input_tokens": 11,
            "cache_read_input_tokens": 7,
            "cache_creation_input_tokens": 5,
            "cache_creation": {"ephemeral_5m_input_tokens": 5},
            "output_tokens": 19,
            "output_tokens_details": {"thinking_tokens": 4},
        },
    )
    assert stream.completed
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    response = events[-1]["response"]
    assert isinstance(response, Mapping)
    assert response["id"] == cast(Mapping[str, JsonValue], events[0]["response"])["id"]
    assert response["model"] == "public" and response["status"] == "completed"
    output = response["output"]
    assert isinstance(output, list) and len(output) == 3
    first = output[0]
    assert isinstance(first, Mapping)
    assert first["content"] == [{"type": "reasoning_text", "text": "think"}]
    assert decode_replay(cast(str, first["encrypted_content"])).native == {
        "type": "thinking",
        "thinking": "think",
        "signature": "sig-end",
        "extension": {"x": 1},
    }
    assert decode_replay(
        cast(str, cast(Mapping[str, JsonValue], output[1])["encrypted_content"])
    ).native == {"type": "redacted_thinking", "data": "opaque"}
    assert cast(Mapping[str, JsonValue], output[2])["content"] == [
        {"type": "output_text", "text": "Hello world", "annotations": []}
    ]
    assert response["usage"] == {
        "input_tokens": 23,
        "input_tokens_details": {
            "cached_tokens": 7,
            "cache_creation_tokens": 5,
            "cache_creation": {"ephemeral_5m_input_tokens": 5},
        },
        "output_tokens": 19,
        "output_tokens_details": {"reasoning_tokens": 4},
        "total_tokens": 42,
    }
    assert (
        "".join(
            cast(str, event["delta"])
            for event in events
            if event["type"] == "response.output_text.delta"
        )
        == "Hello world"
    )
    assert sum(event["type"] == "response.completed" for event in events) == 1


def test_parallel_tools_preserve_call_ids_namespace_and_emit_arguments_once() -> None:
    stream = _stream(
        tools=[
            {
                "type": "namespace",
                "name": "ops",
                "tools": [{"type": "function", "name": "lookup"}],
            },
            {"type": "custom", "name": "command"},
        ]
    )
    events = _start(stream)
    events += _feed(
        stream,
        "content_block_start",
        index=4,
        content_block={
            "type": "tool_use",
            "id": "call-a",
            "name": "ops__lookup",
            "input": {},
            "caller": {"type": "direct"},
        },
    )
    events += _feed(
        stream,
        "content_block_start",
        index=7,
        content_block={
            "type": "tool_use",
            "id": "call-b",
            "name": "command",
            "input": {},
        },
    )
    for fragment in ('{"q":', '"ok"}'):
        events += _feed(
            stream,
            "content_block_delta",
            index=4,
            delta={"type": "input_json_delta", "partial_json": fragment},
        )
    events += _feed(
        stream,
        "content_block_delta",
        index=7,
        delta={"type": "input_json_delta", "partial_json": '{"input":"pwd"}'},
    )
    assert not any(
        event["type"] == "response.function_call_arguments.delta" for event in events
    )
    events += _feed(stream, "content_block_stop", index=7)
    events += _feed(stream, "content_block_stop", index=4)
    events += _end(stream, "tool_use")
    response = cast(Mapping[str, JsonValue], events[-1]["response"])
    output = cast(list[JsonObject], response["output"])
    assert output[0]["call_id"] == "call-a" and output[0]["namespace"] == "ops"
    assert output[0]["name"] == "lookup" and output[0]["arguments"] == '{"q":"ok"}'
    assert output[1]["call_id"] == "call-b" and output[1]["input"] == "pwd"
    assert output[0]["id"] != output[0]["call_id"]
    for index, prefix in ((0, "fc_"), (1, "ctc_")):
        item_id = output[index]["id"]
        assert isinstance(item_id, str) and item_id.startswith(prefix)
        coordinates = [event for event in events if event.get("output_index") == index]
        assert [
            cast(JsonObject, event["item"])["id"]
            for event in coordinates
            if "item" in event
        ] == [item_id, item_id]
        assert [event["item_id"] for event in coordinates if "item_id" in event] == [
            item_id,
            item_id,
        ]
    assert [
        event["delta"]
        for event in events
        if event["type"] == "response.function_call_arguments.delta"
    ] == ['{"q":"ok"}']
    assert [
        event["delta"]
        for event in events
        if event["type"] == "response.custom_tool_call_input.delta"
    ] == ["pwd"]


def test_signature_only_thinking_has_replay_without_invented_visible_text() -> None:
    stream = _stream()
    _start(stream)
    _feed(
        stream,
        "content_block_start",
        index=0,
        content_block={"type": "thinking", "thinking": ""},
    )
    _feed(
        stream,
        "content_block_delta",
        index=0,
        delta={"type": "signature_delta", "signature": "opaque"},
    )
    _feed(stream, "content_block_stop", index=0)
    events = _end(stream)
    response = cast(Mapping[str, JsonValue], events[-1]["response"])
    item = cast(list[JsonObject], response["output"])[0]
    assert "content" not in item
    assert decode_replay(cast(str, item["encrypted_content"])).native["thinking"] == ""
    assert "output_tokens_details" not in cast(
        Mapping[str, JsonValue], response["usage"]
    )


def test_max_tokens_is_incomplete_and_success_waits_for_native_message_stop() -> None:
    stream = _stream()
    _start(stream)
    _feed(
        stream,
        "content_block_start",
        index=0,
        content_block={"type": "text", "text": "partial"},
    )
    _feed(stream, "content_block_stop", index=0)
    events = _end(stream, "max_tokens")
    assert events[-1]["type"] == "response.incomplete"
    assert cast(Mapping[str, JsonValue], events[-1]["response"])[
        "incomplete_details"
    ] == {"reason": "max_output_tokens"}


@pytest.mark.parametrize("usage", [{}, {"output_tokens": True}, {"output_tokens": -1}])
def test_invalid_final_usage_does_not_publish_stale_initial_counts(
    usage: JsonObject,
) -> None:
    stream = _stream()
    _start(stream)
    events = _end(stream, usage=usage)
    assert cast(Mapping[str, JsonValue], events[-1]["response"])["usage"] is None


def test_failure_retains_partial_text_without_completing_partial_tool_arguments() -> (
    None
):
    stream = _stream(tools=[{"type": "function", "name": "lookup"}])
    _start(stream)
    _feed(
        stream,
        "content_block_start",
        index=0,
        content_block={"type": "text", "text": "partial text"},
    )
    _feed(
        stream,
        "content_block_start",
        index=1,
        content_block={"type": "tool_use", "id": "c", "name": "lookup", "input": {}},
    )
    _feed(
        stream,
        "content_block_delta",
        index=1,
        delta={"type": "input_json_delta", "partial_json": '{"x":'},
    )
    failure = ExecutionFailure(FailureKind.TIMEOUT, 504, "timed out", False)
    events = [
        event.data
        for event in parse_sse_lines(
            "".join(stream.terminal_failure(failure)).splitlines()
        )
    ]
    assert [event["type"] for event in events] == ["response.failed"]
    output = events[0]["response"]["output"]
    assert [item["status"] for item in output] == ["incomplete", "incomplete"]
    assert output[0]["content"][0]["text"] == "partial text"
    assert output[1]["arguments"] == ""
    assert stream.terminal_failure(failure) == []
    with pytest.raises(NativeMessagesError, match="terminal"):
        _feed(stream, "message_stop")


@pytest.mark.parametrize(
    "arguments", ['{"input":1}', '{"extra":"x"}', '{"input":"ok","extra":1}']
)
def test_malformed_custom_tool_wrapper_cannot_become_completed_output(
    arguments: str,
) -> None:
    stream = _stream(tools=[{"type": "custom", "name": "command"}])
    _start(stream)
    _feed(
        stream,
        "content_block_start",
        index=0,
        content_block={"type": "tool_use", "id": "c", "name": "command", "input": {}},
    )
    _feed(
        stream,
        "content_block_delta",
        index=0,
        delta={"type": "input_json_delta", "partial_json": arguments},
    )
    with pytest.raises(NativeMessagesError, match="exactly one text"):
        _feed(stream, "content_block_stop", index=0)
    failure = ExecutionFailure(FailureKind.UPSTREAM, 502, "invalid tool", False)
    failed = parse_sse_lines("".join(stream.terminal_failure(failure)).splitlines())[
        -1
    ].data
    output = failed["response"]["output"]
    assert len(output) == 1
    assert output[0]["call_id"] == "c" and output[0]["status"] == "incomplete"


def test_terminal_usage_omits_missing_cache_and_provisional_thinking_details() -> None:
    stream = _stream()
    _feed(
        stream,
        "message_start",
        message={
            "id": "native",
            "type": "message",
            "role": "assistant",
            "model": "upstream",
            "content": [],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 0,
                "output_tokens_details": {"thinking_tokens": 0},
            },
        },
    )
    _feed(
        stream,
        "content_block_start",
        index=0,
        content_block={"type": "thinking", "thinking": "thought", "signature": "sig"},
    )
    _feed(stream, "content_block_stop", index=0)
    events = _end(stream)
    response = cast(Mapping[str, JsonValue], events[-1]["response"])
    assert response["usage"] == {
        "input_tokens": 3,
        "output_tokens": 19,
        "total_tokens": 22,
    }
