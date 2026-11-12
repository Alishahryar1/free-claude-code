"""Buffered native calls still satisfy the destination's executable wire contract."""

from copy import deepcopy

import httpx
import pytest
from openai.types.responses.response_output_item_added_event import (
    ResponseOutputItemAddedEvent,
)

from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    chat_sse_streaming_response,
)
from free_claude_code.application.response_collection import collect_response
from free_claude_code.core.anthropic.passthrough import NativeMessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.chat_completions.request import ChatRequest
from tests.api.test_tool_call_buffer import _response
from tests.providers.test_anthropic_messages_transport import Wire, _sse
from tests.providers.test_anthropic_provider import native_body, provider
from tests.providers.test_history_transports import _harness
from tests.providers.test_native_tool_arguments import tool_events


async def public_frames(body, wire):
    if wire == "chat":

        def fail(error):
            raise error

        response = await chat_sse_streaming_response(
            body, request_id="native-wire", pre_start_error_response=fail
        )
    else:
        response = await _response(wire, body)
    assert isinstance(response, ManagedStreamingResponse)
    try:
        return [str(frame) async for frame in response.body_iterator]
    finally:
        await response.aclose()


def truncated_messages():
    events = tool_events("messages", "{}")
    events[-2]["delta"]["stop_reason"] = "max_tokens"
    events[-2:-2] = [
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "later"},
        },
        {"type": "content_block_stop", "index": 1},
    ]
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize("passthrough", [False, True])
@pytest.mark.parametrize("public", [False, True])
async def test_filtered_messages_blocks_have_dense_consistent_client_indexes(
    passthrough, public
):
    events = truncated_messages()

    async def consume(body):
        return (
            await public_frames(body, "messages")
            if public
            else await collect_response(body, protocol="messages")
        )

    if passthrough:
        p = provider(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Wire([_sse(*events)]),
            )
        )
        try:
            frames = await consume(
                p.stream_native_messages(
                    NativeMessagesRequest({**native_body(), "stream": True})
                )
            )
        finally:
            await p.cleanup()
    else:
        async with _harness("messages", lambda _: (200, events), max_attempts=1) as (
            send,
            _,
            _,
        ):
            frames = await consume(
                send("messages", [{"role": "user", "content": "read"}])
            )
    output = [event.data for event in parse_sse_text("".join(frames))]
    indexed = [(value["type"], value["index"]) for value in output if "index" in value]
    assert indexed == [
        ("content_block_start", 0),
        ("content_block_delta", 0),
        ("content_block_stop", 0),
    ]
    assert "call_probe" not in "".join(frames)
    assert output[-2]["delta"]["stop_reason"] == "max_tokens"


def parallel_responses():
    events = tool_events("responses", "{}")
    first = events[1]["item"]
    second = {**first, "id": "fc_second", "call_id": "call_second"}
    done = {**second, "status": "completed", "arguments": "{}"}
    events[2:2] = [
        {"type": "response.output_item.added", "output_index": 1, "item": second}
    ]
    events[-1:-1] = [
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_second",
            "output_index": 1,
            "delta": "{}",
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_second",
            "output_index": 1,
            "arguments": "{}",
        },
        {"type": "response.output_item.done", "output_index": 1, "item": done},
    ]
    events[-1]["response"]["output"].append(done)
    for index, event in enumerate(events):
        event["sequence_number"] = index
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_only", [False, True])
@pytest.mark.parametrize("public", [False, True])
async def test_native_responses_number_every_emitted_event_in_output_order(
    snapshot_only, public
):
    events = parallel_responses()
    if snapshot_only:
        events = [events[0], events[-1]]
    async with _harness("responses", lambda _: (200, events), max_attempts=1) as (
        send,
        bodies,
        _,
    ):
        body = send("responses", [{"role": "user", "content": "read"}])
        frames = (
            await public_frames(body, "responses")
            if public
            else await collect_response(body, protocol="responses")
        )
    output = [event.data for event in parse_sse_text("".join(frames))]
    assert [value.get("sequence_number") for value in output] == list(
        range(len(output))
    )
    for value in output:
        if value["type"] == "response.output_item.added":
            ResponseOutputItemAddedEvent.model_validate(value)
    assert len(bodies) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [None, "", "   ", "duplicate"])
async def test_native_chat_never_releases_calls_without_unambiguous_ids(invalid):
    events = tool_events("chat", "{}")
    first = events[0]["choices"][0]["delta"]["tool_calls"][0]
    if invalid == "duplicate":
        first["function"]["arguments"] = "{}"
        other = deepcopy(first)
        other["index"] = 1
        events[0]["choices"][0]["delta"]["tool_calls"].append(other)
        events = [events[0], events[-1]]
    elif invalid is None:
        first.pop("id")
    else:
        first["id"] = invalid
    prefix = {
        **events[0],
        "choices": [
            {"index": 0, "delta": {"content": "prefix"}, "finish_reason": None}
        ],
    }
    events.insert(0, prefix)
    async with _harness("chat", lambda _: (200, events), max_attempts=2) as (
        _,
        bodies,
        p,
    ):
        body = p.stream_chat(
            ChatRequest.model_validate(
                {
                    "model": "native",
                    "messages": [{"role": "user", "content": "read"}],
                    "stream": True,
                }
            )
        )
        frames = await public_frames(body, "chat")
    output = [event.data for event in parse_sse_text("".join(frames))]
    assert any("error" in event for event in output)
    assert not any(
        (choice.get("delta") or {}).get("tool_calls")
        for event in output
        for choice in event.get("choices", [])
    )
    assert len(bodies) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol,field",
    [
        ("messages", "id"),
        ("messages", "name"),
        ("responses", "call_id"),
        ("responses", "name"),
    ],
)
async def test_native_complete_calls_require_execution_identity(protocol, field):
    events = tool_events(protocol, "{}")
    if protocol == "messages":
        events[1]["content_block"].pop(field)
        p = provider(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Wire([_sse(*events)]),
            )
        )
        try:
            frames = await public_frames(
                p.stream_native_messages(
                    NativeMessagesRequest({**native_body(), "stream": True})
                ),
                "messages",
            )
        finally:
            await p.cleanup()
    else:
        for event in events:
            if "item" in event:
                event["item"].pop(field, None)
            if "response" in event:
                for item in event["response"]["output"]:
                    item.pop(field, None)
        async with _harness("responses", lambda _: (200, events), max_attempts=1) as (
            send,
            _,
            _,
        ):
            frames = await public_frames(
                send("responses", [{"role": "user", "content": "read"}]), "responses"
            )
    output = [event.data for event in parse_sse_text("".join(frames))]
    assert output[-1]["type"] in {"error", "response.failed"}
    assert not any(
        event.get("content_block", {}).get("type") == "tool_use"
        or event.get("item", {}).get("type") == "function_call"
        for event in output
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["id", "name", "duplicate"])
async def test_native_json_messages_validate_the_same_complete_call_identity(invalid):
    block = {"type": "tool_use", "id": "call", "name": "read", "input": {}}
    content = [block]
    if invalid == "duplicate":
        content.append(deepcopy(block))
    else:
        block.pop(invalid)
    p = provider(
        lambda _: httpx.Response(
            200, json={"type": "message", "stop_reason": "tool_use", "content": content}
        )
    )
    try:
        from free_claude_code.core.failures import ExecutionFailure

        with pytest.raises(ExecutionFailure):
            await collect_response(
                p.stream_native_messages(NativeMessagesRequest(native_body())),
                protocol="messages",
                native_json=True,
            )
    finally:
        await p.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("complete", [False, True])
async def test_generated_alias_with_arguments_waits_for_remaining_name_and_source_completion(
    wire, complete
):
    original = "synthetic.invalid/name:" + "long_" * 20

    def reply(bodies):
        alias = bodies[-1]["tools"][0]["function"]["name"]
        events = tool_events("chat", "{}")
        first = events[0]["choices"][0]["delta"]["tool_calls"][0]
        first["function"] = {"name": alias[:15], "arguments": "{}"}
        if not complete:
            return 200, events[:1]
        second = deepcopy(events[0])
        second["choices"][0]["delta"]["tool_calls"] = [
            {"index": 0, "function": {"name": alias[15:]}}
        ]
        return 200, [events[0], second, events[-1]]

    tools = (
        [{"name": original, "input_schema": {"type": "object"}}]
        if wire == "messages"
        else [{"type": "function", "name": original, "parameters": {"type": "object"}}]
    )
    async with _harness("chat", reply, max_attempts=1) as (send, bodies, _):
        frames = await public_frames(
            send(wire, [{"role": "user", "content": "read"}], tools=tools), wire
        )
    output = [event.data for event in parse_sse_text("".join(frames))]
    names = [
        event.get("content_block", event.get("item", {})).get("name")
        for event in output
        if event["type"] in {"content_block_start", "response.output_item.added"}
    ]
    assert names == ([original] if complete else [])
    assert len(bodies) == 1
