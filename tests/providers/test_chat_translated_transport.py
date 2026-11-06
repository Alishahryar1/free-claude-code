"""Chat clients retain upstream tools and protected history through real HTTP."""

import json
from typing import Any

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.chat_completions.request import ChatRequest
from free_claude_code.core.chat_completions.response import ChatResponseAccumulator
from free_claude_code.core.reasoning import ReasoningPolicy
from tests.providers.test_anthropic_messages_transport import Endpoint
from tests.providers.test_history_transports import _harness, _native
from tests.providers.test_native_tool_arguments import tool_events


async def collect(stream):
    accumulator = ChatResponseAccumulator()
    chunks = [chunk async for chunk in stream]
    for event in parse_sse_text("".join(chunks)):
        if event.data:
            accumulator.feed(event.data)
    assert chunks[-1] == "data: [DONE]\n\n"
    accumulator.finish()
    return accumulator.result()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["messages", "responses"])
async def test_chat_round_trip_preserves_native_reasoning_once(protocol):
    async with _harness(protocol) as (_, bodies, transport):
        request = ChatRequest.model_validate(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        kwargs = {
            "input_tokens": 0,
            "request_id": None,
            "response_model": "public",
            "reasoning": ReasoningPolicy(),
        }
        if protocol == "messages":
            kwargs["endpoint_context"] = Endpoint()
        first = await collect(transport.stream_chat(request, **kwargs))
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
        await collect(transport.stream_chat(second, **kwargs))
    history = (
        bodies[-1]["messages"][1]["content"]
        if protocol == "messages"
        else bodies[-1]["input"]
    )
    assert history.count(_native(protocol)) == 1
    reasoning = [
        item for item in history if item.get("type") in {"thinking", "reasoning"}
    ]
    assert reasoning == [_native(protocol)]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["messages", "responses"])
async def test_chat_explicit_off_keeps_protected_state_without_readable_echo(protocol):
    async with _harness(protocol) as (_, _, transport):
        request = ChatRequest(
            model="test", messages=[{"role": "user", "content": "hello"}]
        )
        kwargs = {
            "input_tokens": 0,
            "request_id": None,
            "response_model": "public",
            "reasoning": ReasoningPolicy.off(),
        }
        if protocol == "messages":
            kwargs["endpoint_context"] = Endpoint()
        result = await collect(transport.stream_chat(request, **kwargs))
    message = result["choices"][0]["message"]
    assert "Find 17." not in json.dumps(message)
    assert any(
        detail["type"] == "reasoning.encrypted"
        for detail in message["reasoning_details"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["messages", "responses", "chat"])
@pytest.mark.parametrize("arguments", ['{"path":', "[]", "null", "", '{"path":42}'])
async def test_completed_chat_tool_arguments_remain_opaque(protocol, arguments):
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
        kwargs = {
            "input_tokens": 0,
            "request_id": None,
            "response_model": "public",
            "reasoning": ReasoningPolicy(),
        }
        if protocol == "messages":
            kwargs["endpoint_context"] = Endpoint()
        result = await collect(transport.stream_chat(request, **kwargs))
    assert (
        result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        == arguments
    )


@pytest.mark.asyncio
async def test_chat_round_trip_keeps_multiple_summary_parts():
    from tests.providers.test_history_transports import _events_for

    native = _native("responses")
    native["summary"].append({"type": "summary_text", "text": "Then verify 19."})

    def reply(_):
        events = _events_for("responses")
        events[-2]["item"] = native
        events[-1]["response"]["output"] = [native]
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
    assert native in bodies[-1]["input"]


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
        value for value in bodies[-1]["input"] if value.get("role") == "assistant"
    )
    assert assistant["content"] == item["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "field"), [("reasoning.text", "text"), ("reasoning.summary", "summary")]
)
async def test_native_chat_json_reassembles_readable_reasoning_but_keeps_opaque_boundaries(
    kind, field
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
    assert bodies[-1]["messages"][1]["reasoning_details"] == expected
