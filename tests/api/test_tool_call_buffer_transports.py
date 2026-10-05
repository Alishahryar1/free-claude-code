"""Existing transport and tool-conversion contracts survive HTTP buffering."""

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.history_replay import decode_replay
from free_claude_code.providers.open_router import OpenRouterProvider
from tests.api.test_tool_call_buffer import _response
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_history_transports import _harness
from tests.providers.test_native_tool_arguments import tool_events


def _tools(wire):
    return (
        [{"type": "function", "name": "read", "parameters": {"type": "object"}}]
        if wire == "responses"
        else [{"name": "read", "input_schema": {"type": "object"}}]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("arguments", ['{"path":"x"}', '{"path":', "null"])
async def test_real_transport_arguments_reach_http_unchanged(protocol, wire, arguments):
    async with _harness(
        protocol, lambda _: (200, tool_events(protocol, arguments))
    ) as (send, bodies, _):
        response = await _response(
            wire, send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire))
        )
        result = "".join([str(chunk) async for chunk in response.body_iterator])
        await response.aclose()
    events = parse_sse_text(result)
    assert len(bodies) == 1
    if wire == "responses":
        call = events[-1].data["response"]["output"][0]
        assert call["arguments"] == arguments
        assert call["call_id"] == "call_probe"
        assert events[-1].event == "response.completed"
    else:
        assert (
            "".join(
                event.data["delta"]["partial_json"]
                for event in events
                if event.data.get("delta", {}).get("type") == "input_json_delta"
            )
            == arguments
        )
        assert events[-1].event == "message_stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_real_transport_cutoff_cannot_complete_partial_tool(protocol, wire):
    events = tool_events(protocol, "x")
    arguments = '{"path":"' + "x" * 70_000
    if protocol == "chat":
        events[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = (
            arguments
        )
        events = events[:-1]
    elif protocol == "messages":
        events[2]["delta"]["partial_json"] = arguments
        events = events[:3]
    else:
        events[2]["delta"] = arguments
        events = events[:3]

    # Force the existing provider holdback to commit before the incomplete call,
    # including writers that already retain tool argument deltas internally.
    text = "visible before call " + "x" * 70_000
    if protocol == "chat":
        events.insert(
            0,
            {
                **events[0],
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ],
            },
        )
    elif protocol == "messages":
        for event in events[1:]:
            event["index"] = 1
        events[1:1] = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
            {"type": "content_block_stop", "index": 0},
        ]
    else:
        for event in events[1:]:
            event["output_index"] = 1
        events[1:1] = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "msg_prefix",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_prefix",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            },
        ]

    def chat_provider():
        return OpenRouterProvider(
            make_provider_config(api_key="key", base_url="https://provider.invalid/v1"),
            admission=immediate_admission(max_attempts=1),
        )

    async with _harness(
        protocol,
        lambda _: (200, events),
        chat_provider_factory=chat_provider,
        max_attempts=1,
    ) as (send, bodies, _):
        response = await _response(
            wire, send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire))
        )
        result = "".join([str(chunk) async for chunk in response.body_iterator])
        await response.aclose()
    assert len(bodies) == 1
    assert text in result
    assert "call_probe" not in result
    assert parse_sse_text(result)[-1].event == (
        "error" if wire == "messages" else "response.failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_native_signed_thinking_interleaves_with_parallel_tool_calls(wire):
    events = [
        tool_events("messages", "x")[0],
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "",
                "signature": "",
                "extension": {"keep": 17},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "before tools"},
        },
        *[
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": f"call_{index}",
                    "name": "read",
                    "input": {},
                },
            }
            for index in (1, 2)
        ],
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": " during tools"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "signed-native"},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"path":"first"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"path":"second"}'},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 10},
        },
        {"type": "message_stop"},
    ]
    async with _harness("messages", lambda _: (200, events)) as (send, bodies, _):
        response = await _response(
            wire, send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire))
        )
        result = "".join([str(chunk) async for chunk in response.body_iterator])
        await response.aclose()
    assert len(bodies) == 1
    delivered = parse_sse_text(result)
    if wire == "messages":
        calls = [
            event.data["content_block"]["id"]
            for event in delivered
            if event.data.get("content_block", {}).get("type") == "tool_use"
        ]
        signature = next(
            event.data["delta"]["signature"]
            for event in delivered
            if event.data.get("delta", {}).get("type") == "signature_delta"
        )
    else:
        output = delivered[-1].data["response"]["output"]
        calls = [item["call_id"] for item in output if item["type"] == "function_call"]
        signature = output[0]["encrypted_content"]
    assert calls == ["call_1", "call_2"]
    assert decode_replay(signature).native == {
        "type": "thinking",
        "thinking": "before tools during tools",
        "signature": "signed-native",
        "extension": {"keep": 17},
    }
