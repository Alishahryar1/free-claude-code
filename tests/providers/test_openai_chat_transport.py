"""Chat execution borrows HTTP resources and releases shared admission."""

import asyncio
import json
from collections.abc import AsyncIterator
from functools import partial

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers import stream_recovery
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
)
from tests.providers.request_factory import make_messages_request

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("function_fields", [{}, {"function": None}])
@pytest.mark.parametrize("alias_collision", [False, True])
async def test_stream_preserves_tool_call_with_functionless_deltas(
    wire, function_fields, alias_collision
):
    tool_deltas = [
        {"index": 0, "id": "call_lookup", "type": "function", **function_fields},
        {"index": 0, "function": {"name": "lookup", "arguments": '{"q":'}},
        {"index": 0, **function_fields},
        {"index": 0, "function": {"arguments": '"fcc"}'}},
    ]

    def reply(request: httpx2.Request) -> httpx2.Response:
        return _tool_response([{"tool_calls": [delta]} for delta in tool_deltas])

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        transport = _transport(client)
        names = ("lookup", "lookup details") if alias_collision else ("lookup",)
        stream = _tool_stream(transport, wire, names)
        events = parse_sse_text("".join([event async for event in stream]))

    if wire == "messages":
        blocks = [
            event.data["content_block"]
            for event in events
            if event.event == "content_block_start"
        ]
        assert [(block["id"], block["name"]) for block in blocks] == [
            ("call_lookup", "lookup")
        ]
        arguments = "".join(
            event.data["delta"]["partial_json"]
            for event in events
            if event.event == "content_block_delta"
        )
        assert arguments == '{"q":"fcc"}'
        assert events[-2].data["delta"]["stop_reason"] == "tool_use"
        assert events[-1].event == "message_stop"
    else:
        assert events[-1].event == "response.completed"
        response = events[-1].data["response"]
        assert response["status"] == "completed"
        calls = response["output"]
        assert len(calls) == 1
        assert calls[0]["type"] == "function_call"
        assert calls[0]["call_id"] == "call_lookup"
        assert calls[0]["name"] == "lookup"
        assert calls[0]["arguments"] == '{"q":"fcc"}'


@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("function_fields", [{}, {"function": None}, {"function": {}}])
@pytest.mark.parametrize(
    "prefix", ["none", "text", "complete_tool", "complete_and_partial_tools"]
)
@pytest.mark.parametrize("committed", [False, True])
async def test_finished_stream_rejects_tool_call_without_name(
    wire, function_fields, prefix, committed, monkeypatch
):
    monkeypatch.setattr(
        stream_recovery,
        "RecoveryHoldbackBuffer",
        partial(
            stream_recovery.RecoveryHoldbackBuffer,
            holdback_seconds=0 if committed else float("inf"),
        ),
    )
    deltas = []
    if prefix == "text":
        deltas.append({"content": "Checking the weather."})
    elif prefix in {"complete_tool", "complete_and_partial_tools"}:
        deltas.append(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_complete",
                        "function": {"name": "lookup", "arguments": '{"q":"Paris"}'},
                    }
                ]
            }
        )
    deltas.append(
        {
            "tool_calls": [
                {
                    "index": 1 if prefix.startswith("complete") else 0,
                    "id": "call_incomplete",
                    **function_fields,
                }
            ]
        }
    )
    names = ("lookup",)
    if prefix == "complete_and_partial_tools":
        names = ("lookup", "other lookup")
        deltas.append(
            {
                "tool_calls": [
                    {
                        "index": 2,
                        "id": "call_partial_name",
                        "function": {"name": "other", "arguments": '{"q":"Boston"}'},
                    }
                ]
            }
        )
    calls = 0

    def reply(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return _tool_response(deltas)

    frames = []
    failure = None
    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        transport = _transport(client, max_attempts=3)
        try:
            async for frame in _tool_stream(transport, wire, names):
                frames.append(frame)
                assert not any(
                    event.event in {"message_stop", "response.completed"}
                    for event in parse_sse_text(frame)
                )
        except ExecutionFailure as error:
            failure = error

    events = parse_sse_text("".join(frames))
    assert calls == 1
    if failure is not None:
        assert failure.kind is FailureKind.UPSTREAM
        assert failure.status_code == 502
        assert not failure.retryable
    else:
        assert events[-1].event == "response.failed"
        assert events[-1].data["response"]["status"] == "failed"


def _tool_response(deltas) -> httpx2.Response:
    chunks = [
        {
            "id": "chat_test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "delta": delta or {},
                    "finish_reason": "tool_calls" if delta is None else None,
                }
            ],
        }
        for delta in [*deltas, None]
    ]
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n",
    )


def _tool_stream(transport, wire, names=("lookup",)):
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    if wire == "messages":
        return transport.stream_messages(
            make_messages_request(
                "model",
                tools=[{"name": name, "input_schema": schema} for name in names],
            )
        )
    return transport.stream_responses(
        OpenAIResponsesRequest.model_validate(
            {
                "model": "model",
                "input": "hello",
                "tools": [
                    {"type": "function", "name": name, "parameters": schema}
                    for name in names
                ],
            }
        )
    )


def _transport(client: AsyncOpenAI, *, max_attempts: int = 1) -> OpenAIChatTransport:
    return OpenAIChatTransport(
        client=client,
        admission=ProviderAdmissionController(
            provider_name="TEST",
            rate_limit=100,
            rate_window=1,
            max_concurrency=1,
            max_attempts=max_attempts,
        ),
        behavior=OpenAIChatBehavior(
            OpenAIChatProfile(
                OpenAIChatRequestPolicy("TEST", ReasoningReplayMode.DISABLED),
                NO_REASONING,
            )
        ),
        read_timeout_s=2,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
    )


def _success() -> httpx2.Response:
    chunk = {
        "id": "chat_test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "model",
        "choices": [
            {"index": 0, "delta": {"content": "hello"}, "finish_reason": "stop"}
        ],
    }
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
    )


async def _consume(transport: OpenAIChatTransport, wire: str) -> str:
    stream = (
        transport.stream_messages(make_messages_request("model"))
        if wire == "messages"
        else transport.stream_responses(
            OpenAIResponsesRequest(model="model", input="hello")
        )
    )
    return "".join([event async for event in stream])


@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_borrowed_client_remains_usable_after_failed_and_successful_turns(wire):
    calls = 0

    def reply(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json={"data": [{"id": "model"}]})
        calls += 1
        if calls == 1:
            return httpx2.Response(401, json={"error": {"message": "expired"}})
        return _success()

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        transport = _transport(client)
        with pytest.raises(ExecutionFailure):
            await _consume(transport, wire)
        async with asyncio.timeout(2):
            assert "hello" in await _consume(transport, wire)
        assert not client.is_closed()
        assert (await client.models.list()).data[0].id == "model"
        assert calls == 2


@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_cancelled_turn_closes_its_stream_and_releases_admission(wire):
    reading = asyncio.Event()
    closed = asyncio.Event()

    class BlockedStream(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            reading.set()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self) -> None:
            closed.set()

    calls = 0

    def reply(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=BlockedStream(),
            )
        return _success()

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    ) as client:
        transport = _transport(client)
        task = asyncio.create_task(_consume(transport, wire))
        try:
            async with asyncio.timeout(2):
                await reading.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert closed.is_set()
            assert not client.is_closed()
            # One concurrency slot: this hangs if the cancelled attempt retains it.
            async with asyncio.timeout(2):
                assert "hello" in await _consume(transport, wire)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
