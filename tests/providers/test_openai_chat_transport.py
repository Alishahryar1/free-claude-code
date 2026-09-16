"""Chat execution borrows HTTP resources and releases shared admission."""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
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


def _transport(client: AsyncOpenAI) -> OpenAIChatTransport:
    return OpenAIChatTransport(
        client=client,
        admission=ProviderAdmissionController(
            provider_name="TEST",
            rate_limit=100,
            rate_window=1,
            max_concurrency=1,
            max_attempts=1,
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
