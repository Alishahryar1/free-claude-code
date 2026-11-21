"""Chat clients retain upstream tools and protected history through real HTTP."""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from free_claude_code.core.anthropic.stream_contracts import SSEEvent, parse_sse_text
from free_claude_code.core.chat_completions.request import ChatRequest
from free_claude_code.core.chat_completions.response import ChatResponseAccumulator
from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.providers.anthropic_messages.transport import (
    AnthropicMessagesTransport,
)
from free_claude_code.providers.openai_chat.provider import OpenAIChatProvider
from free_claude_code.providers.openai_responses.transport import (
    OpenAIResponsesTransport,
)
from tests.providers.test_anthropic_messages_transport import Endpoint, _events
from tests.providers.test_history_transports import _events_for, _harness, _native
from tests.providers.test_native_tool_arguments import tool_events
from tests.wire_payload import array_payload, object_array, object_payload


async def collect(stream: AsyncIterator[StreamOutput]):
    accumulator = ChatResponseAccumulator()
    chunks = [frame async for chunk in stream for frame in chunk.frames]
    for event in parse_sse_text("".join(chunks)):
        if event.data:
            accumulator.feed(event.data)
    assert chunks[-1] == "data: [DONE]\n\n"
    accumulator.finish()
    return accumulator.result()


def _stream_chat(
    transport: AnthropicMessagesTransport
    | OpenAIChatProvider
    | OpenAIResponsesTransport,
    request: ChatRequest,
    reasoning: ReasoningPolicy,
):
    if isinstance(transport, AnthropicMessagesTransport):
        return transport.stream_chat(
            request,
            endpoint_context=Endpoint(),
            input_tokens=0,
            request_id=None,
            response_model="public",
            reasoning=reasoning,
        )
    return transport.stream_chat(
        request,
        input_tokens=0,
        request_id=None,
        response_model="public",
        reasoning=reasoning,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("include_usage", "input_count"),
    [(True, 4.0), (False, 4.0), (False, "unavailable")],
)
async def test_messages_to_chat_completes_with_provider_usage(
    include_usage: bool, input_count: JsonValue
):
    events = _events("Hello")
    object_payload(events[0]["message"])["usage"] = {
        "input_tokens": input_count,
        "cache_read_input_tokens": 5.0,
        "cache_creation_input_tokens": 6.0,
        "output_tokens": 0,
    }
    events[-2]["usage"] = {"output_tokens": 7.0}
    request = ChatRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": include_usage},
        }
    )
    async with _harness("messages", lambda _: (200, events)) as (_, _, transport):
        result = await collect(_stream_chat(transport, request, ReasoningPolicy()))

    message = object_payload(object_array(result["choices"])[0]["message"])
    assert message["content"] == "Hello"
    if include_usage:
        usage = object_payload(result["usage"])
        assert usage == {
            "prompt_tokens": 15,
            "completion_tokens": 7,
            "total_tokens": 22,
            "prompt_tokens_details": {"cached_tokens": 5},
        }
        assert type(usage["completion_tokens"]) is int
        assert (
            type(object_payload(usage["prompt_tokens_details"])["cached_tokens"]) is int
        )
    else:
        assert "usage" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["messages", "responses"])
async def test_chat_round_trip_preserves_native_reasoning_once(protocol: str):
    async with _harness(protocol) as (_, bodies, transport):
        request = ChatRequest.model_validate(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        first = await collect(_stream_chat(transport, request, ReasoningPolicy()))
        assert first["model"] == "public"
        message = first["choices"][0]["message"]
        assert "Find 17." in json.dumps(message)
        second = request.model_copy(
            update={
                "messages": [
                    *request.messages,
                    message,
                    {"role": "user", "content": "next"},
                ]
            }
        )
        await collect(_stream_chat(transport, second, ReasoningPolicy()))
    history = (
        object_payload(array_payload(bodies[-1]["messages"])[1])["content"]
        if protocol == "messages"
        else bodies[-1]["input"]
    )
    history = object_array(history)
    assert history.count(_native(protocol)) == 1
    reasoning = [
        item for item in history if item.get("type") in {"thinking", "reasoning"}
    ]
    assert reasoning == [_native(protocol)]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["messages", "responses"])
async def test_chat_explicit_off_keeps_protected_state_without_readable_echo(
    protocol: str,
):
    async with _harness(protocol) as (_, _, transport):
        request = ChatRequest(
            model="test", messages=[{"role": "user", "content": "hello"}]
        )
        result = await collect(_stream_chat(transport, request, ReasoningPolicy.off()))
    message = result["choices"][0]["message"]
    assert "Find 17." not in json.dumps(message)
    assert any(
        detail["type"] == "reasoning.encrypted"
        for detail in message["reasoning_details"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["messages", "responses", "chat"])
@pytest.mark.parametrize("arguments", ['{"path":', "[]", "null", "", '{"path":42}'])
async def test_completed_chat_tool_arguments_remain_opaque(
    protocol: str, arguments: str
):
    async with _harness(
        protocol, lambda _: (200, tool_events(protocol, arguments))
    ) as (_, _, transport):
        request = ChatRequest.model_validate(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "read"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        },
                    }
                ],
            }
        )
        result = await collect(_stream_chat(transport, request, ReasoningPolicy()))
    # A Messages zero-argument call streams one empty delta after input {}.
    expected = "{}" if protocol == "messages" and not arguments else arguments
    assert (
        result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        == expected
    )


@pytest.mark.asyncio
async def test_chat_round_trip_keeps_multiple_summary_parts():
    from tests.providers.test_history_transports import _events_for

    native = _native("responses")
    array_payload(native["summary"]).append(
        {"type": "summary_text", "text": "Then verify 19."}
    )

    def reply(_):
        events = _events_for("responses")
        events[-2]["item"] = native
        object_payload(events[-1]["response"])["output"] = [native]
        return 200, events

    async with _harness("responses", reply) as (_, bodies, transport):
        request = ChatRequest(
            model="test", messages=[{"role": "user", "content": "hello"}]
        )
        kwargs = {
            "input_tokens": 0,
            "request_id": None,
            "response_model": "public",
            "reasoning": ReasoningPolicy(),
        }
        first = await collect(transport.stream_chat(request, **kwargs))
        request = request.model_copy(
            update={
                "messages": [
                    *request.messages,
                    first["choices"][0]["message"],
                    {"role": "user", "content": "next"},
                ]
            }
        )
        await collect(transport.stream_chat(request, **kwargs))
    assert native in object_array(bodies[-1]["input"])


@pytest.mark.asyncio
async def test_chat_refusal_round_trip_preserves_responses_refusal_part():
    from tests.providers.test_openai_responses_transport import _completed_response

    item = {
        "type": "message",
        "id": "msg_refusal",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
    }
    completed = _completed_response(model="m")
    completed["output"] = [item]
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**completed, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": item,
        },
        {"type": "response.completed", "sequence_number": 2, "response": completed},
    ]
    async with _harness("responses", lambda _: (200, events)) as (_, bodies, transport):
        request = ChatRequest(
            model="m", messages=[{"role": "user", "content": "hello"}]
        )
        kwargs = {
            "input_tokens": 0,
            "request_id": None,
            "response_model": "public",
            "reasoning": ReasoningPolicy(),
        }
        first = await collect(transport.stream_chat(request, **kwargs))
        message = first["choices"][0]["message"]
        assert message["refusal"] == item["content"][0]["refusal"]
        await collect(
            transport.stream_chat(
                request.model_copy(
                    update={
                        "messages": [
                            *request.messages,
                            message,
                            {"role": "user", "content": "An allowed topic"},
                        ]
                    }
                ),
                **kwargs,
            )
        )
    assistant = next(
        value
        for value in object_array(bodies[-1]["input"])
        if value.get("role") == "assistant"
    )
    assert object_payload(assistant)["content"] == item["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "field"), [("reasoning.text", "text"), ("reasoning.summary", "summary")]
)
async def test_native_chat_json_reassembles_readable_reasoning_but_keeps_opaque_boundaries(
    kind: str, field: str
):
    opaque = [
        {"type": "reasoning.encrypted", "index": 0, "id": "a", "data": "first"},
        {"type": "reasoning.encrypted", "index": 0, "id": "b", "data": "second"},
    ]
    details = [
        {"type": kind, "index": 0, "id": "readable", field: "Check "},
        {"type": kind, "index": 0, "id": "readable", field: "the file."},
        *opaque,
    ]
    events: list[dict[str, Any]] = [
        {
            "id": "completion",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_details": [detail]},
                    "finish_reason": None,
                }
            ],
        }
        for detail in details
    ]
    events.append(
        {
            "id": "completion",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}
            ],
        }
    )
    async with _harness("chat", lambda _: (200, events)) as (_, bodies, transport):
        request = ChatRequest(
            model="m", messages=[{"role": "user", "content": "hello"}], stream=False
        )
        first = await collect(transport.stream_chat(request))
        message = first["choices"][0]["message"]
        expected = [
            {"type": kind, "index": 0, "id": "readable", field: "Check the file."},
            *opaque,
        ]
        assert message["reasoning_details"] == expected
        await collect(
            transport.stream_chat(
                request.model_copy(
                    update={
                        "messages": [
                            *request.messages,
                            message,
                            {"role": "user", "content": "next"},
                        ]
                    }
                )
            )
        )
    assert object_array(bodies[-1]["messages"])[1]["reasoning_details"] == expected


def _tool_names(event: SSEEvent):
    data = event.data
    block = data.get("content_block")
    if isinstance(block, dict) and block.get("type") == "tool_use":
        return [block["name"]]
    items = [data.get("item")]
    response = data.get("response")
    if isinstance(response, dict):
        items += array_payload(response.get("output") or [])
    return [
        item["name"]
        for item in items
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_fragmented_tool_name_starts_with_complete_name(wire: str):
    template = _events_for("chat")[0]
    deltas = [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_probe",
                    "type": "function",
                    "function": {"name": "Re", "arguments": ""},
                }
            ]
        },
        {
            "tool_calls": [
                {"index": 0, "function": {"name": "ad", "arguments": '{"path":"a"}'}}
            ]
        },
    ]
    events = [
        {**template, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        for delta in deltas
    ] + [
        {
            **template,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }
    ]
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    async with _harness("chat", lambda _: (200, events)) as (send, _, _):
        stream = send(
            wire,
            [{"role": "user", "content": "read"}],
            tools=[{"type": "function", "name": "Read", "parameters": schema}]
            if wire == "responses"
            else [{"name": "Read", "input_schema": schema}],
        )
        output = parse_sse_text(
            "".join([frame async for chunk in stream for frame in chunk.frames])
        )
    names = [name for event in output for name in _tool_names(event)]
    assert names and set(names) == {"Read"}
