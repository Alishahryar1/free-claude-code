"""Ordinary tool arguments and harness error feedback survive protocol translation."""

import json

import pytest

from free_claude_code.core.anthropic.sse_aggregation import (
    aggregate_anthropic_sse_to_message,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses.streaming.event_builders import (
    ResponseEventBuilder,
)
from tests.providers.test_history_transports import _events_for, _harness
from tests.wire_payload import (
    array_payload,
    object_array,
    object_payload,
    string_payload,
)

ARGUMENTS = [
    '{"path":"ok"}',
    '{"path":',
    "{}",
    '{"path":42}',
    "[]",
    "null",
    "",
    '{"x":1e400,"unicode":"\\u263a"}',
]


def tool_events(
    protocol: str, arguments: str, *, name: str = "read"
) -> list[JsonObject]:
    if protocol == "chat":
        template = _events_for("chat")[0]
        deltas = [
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_probe",
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }
                ]
            }
        ]
        deltas += [
            {"tool_calls": [{"index": 0, "function": {"arguments": part}}]}
            for part in arguments
        ]
        return [
            {
                **template,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            for delta in deltas
        ] + [
            {
                **template,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
        ]
    if protocol == "messages":
        return [
            _events_for("messages")[0],
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_probe",
                    "name": name,
                    "input": {},
                },
            },
            *[
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": part},
                }
                for part in arguments or [""]
            ],
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 10},
            },
            {"type": "message_stop"},
        ]
    builder = ResponseEventBuilder()
    item = {
        "type": "function_call",
        "id": "fc_probe",
        "call_id": "call_probe",
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }
    response = {
        "id": "resp_probe",
        "object": "response",
        "created_at": 0,
        "model": "upstream",
        "status": "completed",
        "output": [item],
        "usage": {"input_tokens": 1, "output_tokens": 10, "total_tokens": 11},
    }
    wire = builder.response_created({**response, "status": "in_progress", "output": []})
    wire += builder.output_item_added(
        0, {**item, "status": "in_progress", "arguments": ""}
    )
    wire += "".join(
        builder.function_call_arguments_delta("fc_probe", 0, part) for part in arguments
    )
    wire += builder.function_call_arguments_done("fc_probe", 0, arguments)
    wire += builder.output_item_done(0, item) + builder.response_completed(response)
    return [event.data for event in parse_sse_text(wire)]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("arguments", ARGUMENTS)
async def test_completed_native_arguments_are_transparent(
    protocol: str, wire: str, arguments: str
) -> None:
    async with _harness(
        protocol, lambda _: (200, tool_events(protocol, arguments))
    ) as (send, bodies, _):
        history = [{"role": "user", "content": "read"}]
        events = parse_sse_text(
            "".join(
                [
                    frame
                    async for chunk in send(
                        wire,
                        history,
                        tools=[
                            {
                                "type": "function",
                                "name": "read",
                                "parameters": {"type": "object"},
                            }
                        ]
                        if wire == "responses"
                        else [{"name": "read", "input_schema": {"type": "object"}}],
                    )
                    for frame in chunk.frames
                ]
            )
        )
        assert len(bodies) == 1
        if wire == "responses":
            # A Messages zero-argument call streams one empty delta after input {}.
            expected = "{}" if protocol == "messages" and not arguments else arguments
            assert events[-1].event == "response.completed"
            call = array_payload(object_payload(events[-1].data["response"])["output"])[
                0
            ]
            assert object_payload(call)["arguments"] == expected
            assert object_payload(call)["call_id"] == "call_probe"
            assert (
                "".join(
                    string_payload(e.data["delta"])
                    for e in events
                    if e.event == "response.function_call_arguments.delta"
                )
                == expected
            )
        else:
            assert events[-1].event == "message_stop"
            assert not any(e.event == "error" for e in events)
            assert (
                "".join(
                    string_payload(object_payload(e.data["delta"])["partial_json"])
                    for e in events
                    if object_payload(e.data.get("delta", {})).get("type")
                    == "input_json_delta"
                )
                == arguments
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("arguments", ARGUMENTS)
async def test_harness_tool_error_is_kept_in_next_request(
    protocol: str, arguments: str
) -> None:
    async with _harness(
        protocol, lambda _: (200, tool_events(protocol, arguments))
    ) as (send, bodies, _):
        history = [{"role": "user", "content": "read"}]
        first = parse_sse_text(
            "".join(
                [
                    frame
                    async for chunk in send("responses", history)
                    for frame in chunk.frames
                ]
            )
        )
        call = array_payload(object_payload(first[-1].data["response"])["output"])[0]
        history += [
            call,
            {
                "type": "function_call_output",
                "call_id": object_payload(call)["call_id"],
                "output": "Invalid arguments. Supply a path string.",
            },
        ]
        _ = [
            frame
            async for chunk in send("responses", history)
            for frame in chunk.frames
        ]
        if protocol == "chat":
            sent = bodies[-1]["messages"]
            assert (
                object_payload(
                    object_payload(
                        array_payload(
                            object_payload(array_payload(sent)[-2])["tool_calls"]
                        )[0]
                    )["function"]
                )["arguments"]
                == arguments
            )
            assert (
                object_payload(array_payload(sent)[-1])["tool_call_id"]
                == object_payload(call)["call_id"]
            )
            assert (
                object_payload(array_payload(sent)[-1])["content"]
                == object_payload(history[-1])["output"]
            )
        else:
            assert array_payload(bodies[-1]["input"])[-2:] == history[-2:]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments", ['{"path":', "[]", "null", '{"x":NaN}', '{"x":1e400}']
)
@pytest.mark.parametrize("upstream_error", [False, True])
async def test_nonstream_messages_requires_an_object_and_preserves_upstream_errors(
    arguments: str, upstream_error: bool
) -> None:
    error = {"type": "overloaded_error", "message": "Try again"}

    async def stream():
        for event in tool_events("messages", arguments):
            yield "data: " + json.dumps(event) + "\n\n"
        if upstream_error:
            yield "data: " + json.dumps({"type": "error", "error": error}) + "\n\n"

    if upstream_error:
        _, actual, _ = await aggregate_anthropic_sse_to_message(stream())
        assert actual == error
    else:
        with pytest.raises(ExecutionFailure) as raised:
            await aggregate_anthropic_sse_to_message(stream())
        assert raised.value.kind is FailureKind.UPSTREAM
        assert raised.value.status_code == 502
        assert not raised.value.retryable


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    "text",
    [
        "before <tool_call><function=read><parameter=path>file</parameter></function></tool_call> after",
        "before ● <function=read><parameter=path>file</parameter> after",
        'WebSearch {"query":"example"}',
        'WebFetch {"url":"https://example.test"}',
        "before <|im_end|> after",
    ],
)
async def test_tool_looking_text_stays_literal_across_chunk_boundaries(
    wire: str, text: str
) -> None:
    template = _events_for("chat")[0]
    for split in range(len(text) + 1):
        events = [
            {
                **template,
                "choices": [
                    {"index": 0, "delta": {"content": part}, "finish_reason": None}
                ],
            }
            for part in [text[:split], text[split:]]
        ]
        events.append(
            {
                **template,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
        )
        async with _harness("chat", lambda _, events=events: (200, events)) as (
            send,
            _bodies,
            _,
        ):
            output = parse_sse_text(
                "".join(
                    [
                        frame
                        async for chunk in send(
                            wire,
                            [{"role": "user", "content": "read"}],
                            tools=[
                                {
                                    "type": "function",
                                    "name": "read",
                                    "parameters": {"type": "object"},
                                }
                            ]
                            if wire == "responses"
                            else [{"name": "read", "input_schema": {"type": "object"}}],
                        )
                        for frame in chunk.frames
                    ]
                )
            )
        if wire == "responses":
            items = object_payload(output[-1].data["response"])["output"]
            assert all(
                object_payload(item)["type"] == "message"
                for item in array_payload(items)
            )
            assert (
                "".join(
                    string_payload(part["text"])
                    for item in array_payload(items)
                    for part in object_array(object_payload(item)["content"])
                )
                == text
            )
        else:
            assert not any(
                object_payload(e.data.get("content_block", {})).get("type")
                == "tool_use"
                for e in output
            )
            assert (
                "".join(
                    string_payload(
                        object_payload(e.data.get("delta", {})).get("text", "")
                    )
                    for e in output
                )
                == text
            )
            assert (
                next(
                    object_payload(e.data["delta"])["stop_reason"]
                    for e in output
                    if e.event == "message_delta"
                )
                == "end_turn"
            )
