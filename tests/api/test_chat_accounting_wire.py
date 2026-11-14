"""OpenRouter accounting traverses the SDK and public/private destination paths."""

from copy import deepcopy

import pytest

from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    gemini_sse_streaming_response,
)
from free_claude_code.application.response_collection import collect_response
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.chat_completions.request import ChatRequest
from free_claude_code.core.chat_completions.response import ChatResponseAccumulator
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.gemini import GeminiRequest
from tests.api.test_native_tool_wire_contract import public_frames
from tests.core.test_chat_accounting_trailers import accounting, terminal
from tests.core.test_chat_tool_buffer import call, event
from tests.providers.test_history_transports import _harness


def source_events(tools):
    if tools:
        events = [
            event([call(0, "read", '{"path":', "a")]),
            event([call(1, "read", '{"path":', "b")]),
            event([call(1, arguments='"b"}')]),
            event([call(0, arguments='"a"}')]),
        ]
    else:
        events = [event(content="Hello")]
    reason = "tool_calls" if tools else "stop"
    events.extend([terminal(reason), accounting(reason)])
    events[-1]["usage"].update(
        prompt_tokens_details={"cached_tokens": 4, "cache_write_tokens": 2},
        completion_tokens_details={"reasoning_tokens": 3},
    )
    return [
        {
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "free",
            "provider": "fixture",
            **chunk,
        }
        for chunk in events
    ]


def destination_stream(wire, send, provider):
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    if wire == "chat":
        return provider.stream_chat(
            ChatRequest.model_validate(
                {
                    "model": "free",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream_options": {"include_usage": True},
                }
            )
        )
    if wire == "gemini":
        return provider.stream_gemini(
            GeminiRequest.model_validate(
                {
                    "model": "free",
                    "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                    "tools": [
                        {
                            "functionDeclarations": [
                                {
                                    "name": "read",
                                    "parameters": {
                                        "type": "OBJECT",
                                        "properties": {"path": {"type": "STRING"}},
                                    },
                                }
                            ]
                        }
                    ],
                }
            )
        )
    tools = (
        [{"type": "function", "name": "read", "parameters": schema}]
        if wire == "responses"
        else [{"name": "read", "input_schema": schema}]
    )
    return send(wire, [{"role": "user", "content": "hello"}], tools=tools)


async def consume(body, wire, public):
    if not public:
        return await collect_response(body, protocol=wire)
    if wire != "gemini":
        return await public_frames(body, wire)

    def fail(error):
        raise error

    response = await gemini_sse_streaming_response(
        body, pre_start_error_response=fail, request_id="accounting"
    )
    assert isinstance(response, ManagedStreamingResponse)
    try:
        return [str(frame) async for frame in response.body_iterator]
    finally:
        await response.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["chat", "messages", "responses", "gemini"])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("tools", [False, True])
async def test_accounting_retains_usage_and_exactly_once_completion(
    wire, public, tools
):
    events = source_events(tools)
    # Repeated accounting must not reseal calls or add usage totals.
    events.append(deepcopy(events[-1]))
    completed = []
    async with _harness("chat", lambda _: (200, events), max_attempts=2) as (
        send,
        bodies,
        provider,
    ):
        stream = destination_stream(wire, send, provider)

        async def observe():
            try:
                async for packet in stream:
                    completed.extend(packet.completed_calls)
                    yield packet
            finally:
                await stream.aclose()

        frames = await consume(observe(), wire, public)
        assert len(bodies) == 1
    assert len(completed) == (2 if tools else 0)
    output = [e.data for e in parse_sse_text("".join(frames))]
    if wire == "chat":
        accumulator = ChatResponseAccumulator()
        for value in output:
            accumulator.feed(value)
        accumulator.finish()
        message = accumulator.result()["choices"][0]["message"]
        assert accumulator.result()["usage"] == events[-1]["usage"]
        assert (
            sum(
                c.get("finish_reason") is not None
                for value in output
                for c in value.get("choices", [])
            )
            == 1
        )
        assert "".join(frames).count("data: [DONE]") == 1
        assert all(value["choices"] == [] for value in output if value.get("usage"))
        assert (
            [c["id"] for c in message["tool_calls"]] == ["a", "b"]
            if tools
            else message["content"] == "Hello"
        )
    elif wire == "messages":
        finals = [v for v in output if v.get("type") == "message_delta"]
        assert len(finals) == 1
        assert finals[0]["usage"] == {
            "input_tokens": 26,
            "output_tokens": 13,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 2,
        }
        assert sum(v.get("type") == "message_stop" for v in output) == 1
        if tools:
            assert [
                v["content_block"]["id"]
                for v in output
                if v.get("content_block", {}).get("type") == "tool_use"
            ] == ["a", "b"]
        else:
            assert (
                "".join(v.get("delta", {}).get("text", "") for v in output) == "Hello"
            )
    elif wire == "responses":
        finals = [v for v in output if v.get("type") == "response.completed"]
        assert len(finals) == 1
        assert finals[0]["response"]["usage"]["input_tokens"] == 32
        assert finals[0]["response"]["usage"]["output_tokens"] == 13
        items = finals[0]["response"]["output"]
        assert (
            [i["call_id"] for i in items if i["type"] == "function_call"] == ["a", "b"]
            if tools
            else items[0]["content"][0]["text"] == "Hello"
        )
    else:
        assert (
            sum(
                bool(c.get("finishReason"))
                for v in output
                for c in v.get("candidates", [])
            )
            == 1
        )
        assert output[-1]["usageMetadata"] == {
            "promptTokenCount": 32,
            "candidatesTokenCount": 10,
            "totalTokenCount": 45,
            "cachedContentTokenCount": 4,
            "thoughtsTokenCount": 3,
        }
        parts = [
            p
            for v in output
            for c in v.get("candidates", [])
            for p in c.get("content", {}).get("parts", [])
        ]
        assert (
            [p["functionCall"]["id"] for p in parts if "functionCall" in p]
            == ["a", "b"]
            if tools
            else "".join(p.get("text", "") for p in parts) == "Hello"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["chat", "messages", "responses", "gemini"])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("failure", ["content", "error"])
async def test_late_semantic_output_is_not_accounting_or_another_generation(
    wire, public, failure
):
    events = source_events(True)
    if failure == "content":
        events[-1]["choices"][0]["delta"]["content"] = "MUST_NOT_LEAK"
    else:
        events[-1]["error"] = {"message": "upstream failed", "code": 503}
    async with _harness("chat", lambda _: (200, events), max_attempts=2) as (
        send,
        bodies,
        provider,
    ):
        stream = destination_stream(wire, send, provider)
        if public:
            frames = await consume(stream, wire, True)
            assert "MUST_NOT_LEAK" not in "".join(frames)
            output = [e.data for e in parse_sse_text("".join(frames))]
            assert (
                output[-1].get("error") or output[-1].get("type") == "response.failed"
            )
        else:
            with pytest.raises(ExecutionFailure):
                await consume(stream, wire, False)
        assert len(bodies) == 1
