"""Source completion, rather than name guesses, owns public tool calls."""

from copy import deepcopy

import pytest

from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    chat_sse_streaming_response,
    gemini_sse_streaming_response,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.chat_completions.request import ChatRequest
from free_claude_code.core.gemini import GeminiRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from tests.api.test_tool_call_buffer import _response
from tests.providers.test_chat_translated_transport import _tool_names
from tests.providers.test_history_transports import _events_for, _harness
from tests.providers.test_native_tool_arguments import tool_events


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    "name,fragments", [("Read", ("Re", "ad")), ("foofoo", ("foo", "foo"))]
)
async def test_parallel_name_fragments_wait_for_source_completion(
    wire, name, fragments
):
    template = _events_for("chat")[0]
    deltas = [
        {
            "tool_calls": [
                {
                    "index": 8,
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": fragments[0], "arguments": '{"path":'},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "index": 2,
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "other", "arguments": "{}"},
                }
            ]
        },
        {
            "tool_calls": [
                {"index": 8, "function": {"name": fragments[1], "arguments": '"a"}'}}
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
    tools = [
        {"type": "function", "name": value, "parametersJsonSchema": {"type": "object"}}
        if wire == "responses"
        else {"name": value, "input_schema": {"type": "object"}}
        for value in (name, "other")
    ]
    async with _harness("chat", lambda _: (200, events), max_attempts=1) as (
        send,
        bodies,
        _,
    ):
        response = await _response(
            wire, send(wire, [{"role": "user", "content": "read"}], tools=tools)
        )
        try:
            output = parse_sse_text(
                "".join([str(chunk) async for chunk in response.body_iterator])
            )
        finally:
            await response.aclose()
    starts = {"content_block_start", "response.output_item.added"}
    names = [
        value
        for event in output
        if event.event in starts
        for value in _tool_names(event)
    ]
    assert names == [name, "other"]
    assert len(bodies) == 1


@pytest.mark.asyncio
async def test_incomplete_response_item_never_becomes_a_complete_message_tool():
    events = deepcopy(tool_events("responses", '{"path":'))
    for event in events:
        item = event.get("item")
        if isinstance(item, dict) and event["type"] == "response.output_item.done":
            item["status"] = "incomplete"
    terminal = events[-1]
    terminal["type"] = "response.incomplete"
    terminal["response"]["status"] = "incomplete"
    terminal["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
    terminal["response"]["output"][0]["status"] = "incomplete"
    async with _harness("responses", lambda _: (200, events), max_attempts=1) as (
        send,
        _,
        _,
    ):
        response = await _response(
            "messages", send("messages", [{"role": "user", "content": "read"}])
        )
        assert isinstance(response, ManagedStreamingResponse)
        try:
            output = "".join([str(chunk) async for chunk in response.body_iterator])
        finally:
            await response.aclose()
    assert "call_probe" not in output
    assert "input_json_delta" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "chat", "gemini"])
async def test_completed_response_call_survives_an_earlier_unfinished_call(wire):
    original = tool_events("responses", "{}")
    first = deepcopy(original[1]["item"])
    first.update(
        id="fc_abandoned", call_id="call_abandoned", status="in_progress", arguments=""
    )
    second = deepcopy(original[1]["item"])
    second.update(
        id="fc_ready", call_id="call_ready", status="completed", arguments="{}"
    )
    events = [
        original[0],
        {"type": "response.output_item.added", "output_index": 0, "item": first},
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {**second, "status": "in_progress", "arguments": ""},
        },
        {"type": "response.output_item.done", "output_index": 1, "item": second},
        {
            "type": "response.failed",
            "response": {
                **original[-1]["response"],
                "status": "failed",
                "output": [first, second],
                "error": {"code": "server_error", "message": "synthetic interruption"},
            },
        },
    ]
    async with _harness("responses", lambda _: (200, events), max_attempts=1) as (
        send,
        _,
        provider,
    ):
        if wire in {"messages", "responses"}:
            response = await _response(
                wire, send(wire, [{"role": "user", "content": "read"}])
            )
        else:
            options = {
                "input_tokens": 0,
                "request_id": "complete-call",
                "response_model": "public",
                "reasoning": ReasoningPolicy.provider_default(),
            }
            if wire == "chat":
                body = provider.stream_chat(
                    ChatRequest.model_validate(
                        {
                            "model": "upstream",
                            "messages": [{"role": "user", "content": "read"}],
                        }
                    ),
                    **options,
                )
                wrap = chat_sse_streaming_response
            else:
                body = provider.stream_gemini(
                    GeminiRequest.model_validate(
                        {
                            "model": "upstream",
                            "contents": [{"parts": [{"text": "read"}]}],
                            "tools": [
                                {
                                    "functionDeclarations": [
                                        {
                                            "name": "read",
                                            "parametersJsonSchema": {"type": "object"},
                                        }
                                    ]
                                }
                            ],
                        }
                    ),
                    **options,
                )
                wrap = gemini_sse_streaming_response

            def unexpected(error):
                raise error

            response = await wrap(
                body, request_id="complete-call", pre_start_error_response=unexpected
            )
        assert isinstance(response, ManagedStreamingResponse)
        try:
            output = "".join([str(chunk) async for chunk in response.body_iterator])
        finally:
            await response.aclose()
    assert "call_abandoned" not in output
    assert "call_ready" in output
