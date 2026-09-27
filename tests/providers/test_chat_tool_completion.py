import json

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers import stream_recovery
from free_claude_code.providers.openai_chat import transport as chat_transport
from tests.providers.request_factory import make_messages_request
from tests.providers.test_openai_chat_transport import _transport

pytestmark = pytest.mark.asyncio
SCHEMA = {
    "type": "object",
    "properties": {"flag": {"type": "boolean"}},
    "required": ["flag"],
}


def _reply(delta, finish_reason="tool_calls"):
    chunk = {
        "id": "chat_test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
    )


def _request():
    return make_messages_request(
        "model", tools=[{"name": "Task", "input_schema": SCHEMA}]
    )


@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_text_tool_boolean_is_decoded_before_emission(wire):
    def reply(request):
        return _reply(
            {"content": "● <function=Task><parameter=flag>true</parameter>"}, "stop"
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        transport = _transport(client)
        stream = (
            transport.stream_messages(_request())
            if wire == "messages"
            else transport.stream_responses(
                OpenAIResponsesRequest.model_validate(
                    {
                        "model": "model",
                        "input": "hello",
                        "tools": [
                            {"type": "function", "name": "Task", "parameters": SCHEMA}
                        ],
                    }
                )
            )
        )
        events = parse_sse_text("".join([part async for part in stream]))
    arguments = []
    for event in events:
        if event.event == "response.function_call_arguments.delta":
            arguments.append(event.data["delta"])
        elif event.data.get("delta", {}).get("type") == "input_json_delta":
            arguments.append(event.data["delta"]["partial_json"])
    assert json.loads("".join(arguments)) == {"flag": True}


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize(
    "finish_reason", ["stop", "tool_calls", "length", "content_filter"]
)
async def test_normal_malformed_call_fails_without_repair(
    monkeypatch, committed, finish_reason
):
    recovery = stream_recovery.RecoveryController()
    recovery._holdback = stream_recovery.RecoveryHoldbackBuffer(
        holdback_seconds=0 if committed else 100000, max_bytes=10**9
    )
    monkeypatch.setattr(chat_transport, "RecoveryController", lambda: recovery)
    calls = []

    def reply(request):
        calls.append(request)
        return _reply(
            {
                "content": "working",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "good",
                        "type": "function",
                        "function": {"name": "Task", "arguments": '{"flag":true}'},
                    },
                    {
                        "index": 1,
                        "id": "bad",
                        "type": "function",
                        "function": {"name": "Task", "arguments": '{"flag":'},
                    },
                ],
            },
            finish_reason,
        )

    frames = []
    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        with pytest.raises(ExecutionFailure) as caught:
            async for frame in _transport(client, max_attempts=3).stream_messages(
                _request()
            ):
                frames.append(frame)
                assert frame
    assert caught.value.kind is FailureKind.UPSTREAM
    assert caught.value.status_code == 502
    assert not caught.value.retryable
    assert len(calls) == 1
    events = parse_sse_text("".join(frames))
    assert not any(e.event in ("message_delta", "message_stop") for e in events)
    if committed:
        starts = [e.data["index"] for e in events if e.event == "content_block_start"]
        stops = [e.data["index"] for e in events if e.event == "content_block_stop"]
        assert sorted(starts) == sorted(stops)
        assert len(starts) == 3
    else:
        assert not frames


@pytest.mark.parametrize(
    "arguments",
    ["", " ", "[]", "null", "true", '"text"', '{"flag":NaN}', '{"flag":Infinity}'],
)
async def test_native_non_object_arguments_are_rejected(arguments):
    def reply(request):
        return _reply(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call",
                        "type": "function",
                        "function": {"name": "Task", "arguments": arguments},
                    }
                ]
            }
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        with pytest.raises(ExecutionFailure, match="valid JSON object"):
            _ = [part async for part in _transport(client).stream_messages(_request())]


@pytest.mark.parametrize(
    "arguments", ["{}", '{ "flag": "true", "amount": 1.2300e999 }']
)
async def test_native_objects_are_not_schema_checked_or_reserialized(arguments):
    def reply(request):
        return _reply(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call",
                        "type": "function",
                        "function": {"name": "Task", "arguments": arguments},
                    }
                ]
            }
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        frames = [part async for part in _transport(client).stream_messages(_request())]
    events = parse_sse_text("".join(frames))
    assert (
        "".join(
            e.data["delta"]["partial_json"]
            for e in events
            if e.data.get("delta", {}).get("type") == "input_json_delta"
        )
        == arguments
    )
    assert events[-1].event == "message_stop"


@pytest.mark.parametrize("valid", [False, True])
async def test_argument_alias_flush_precedes_completion_validation(monkeypatch, valid):
    from free_claude_code.providers.openai_chat.behavior import OpenAIChatBehavior

    monkeypatch.setattr(
        OpenAIChatBehavior,
        "tool_argument_aliases",
        lambda self, body: {"Task": {"wire_flag": "flag"}},
    )
    arguments = '{"wire_flag":true}' if valid else '{"wire_flag":'

    def reply(request):
        return _reply(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call",
                        "type": "function",
                        "function": {"name": "Task", "arguments": arguments},
                    }
                ]
            }
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        transport = _transport(client)
        if not valid:
            with pytest.raises(ExecutionFailure, match="valid JSON object"):
                _ = [part async for part in transport.stream_messages(_request())]
        else:
            frames = [part async for part in transport.stream_messages(_request())]
            assert "wire_flag" not in "".join(frames)
            assert any(
                e.event == "message_stop" for e in parse_sse_text("".join(frames))
            )
