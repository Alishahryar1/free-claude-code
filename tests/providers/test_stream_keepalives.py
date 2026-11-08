"""Upstream heartbeat traffic cannot extend the configured response read wait."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI
from starlette.responses import JSONResponse

from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    anthropic_sse_streaming_response,
    gemini_sse_streaming_response,
)
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.anthropic_messages.passthrough import (
    stream_native_messages,
)
from free_claude_code.providers.anthropic_messages.transport import (
    AnthropicMessagesTransport,
)
from free_claude_code.providers.gemini.transport import GeminiTransport
from free_claude_code.providers.history_negotiation import HistoryModeCache
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
)
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport
from tests.providers.request_factory import make_messages_request
from tests.providers.test_anthropic_messages_transport import Endpoint, _events, _sse
from tests.providers.test_openai_responses_transport import (
    _completed_event,
    _text_delta,
)
from tests.providers.test_sdk_stream_lifecycle import _chat_chunk

EGRESSES = ("chat", "responses", "messages", "native_messages", "gemini")
READ_TIMEOUT = 0.2


class Heartbeats(httpx.AsyncByteStream, httpx2.AsyncByteStream):
    def __init__(self, frame, *, prefix=b""):
        self.frame = frame
        self.prefix = prefix
        self.started = asyncio.Event()
        self.closed = False
        self.reads = 0

    async def __aiter__(self):
        if self.prefix:
            yield self.prefix
        while True:
            await asyncio.sleep(0.005)
            self.started.set()
            self.reads += 1
            yield self.frame

    async def aclose(self):
        self.closed = True


@asynccontextmanager
async def upstream(egress, body_factory, *, timeout=READ_TIMEOUT, attempts=1):
    admission = ProviderAdmissionController(
        provider_name="TEST",
        rate_limit=10000,
        rate_window=1,
        max_concurrency=1,
        max_attempts=attempts,
        base_delay=0,
        max_delay=0,
        jitter=0,
    )

    def reply(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=body_factory(),
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(reply))
    sdk = None
    transport = None
    if egress in {"chat", "responses"}:

        def sdk_reply(request):
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=body_factory(),
            )

        sdk = AsyncOpenAI(
            api_key="test",
            base_url="https://provider.invalid/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(sdk_reply)),
        )
        if egress == "chat":
            transport = OpenAIChatTransport(
                client=sdk,
                admission=admission,
                read_timeout_s=timeout,
                behavior=OpenAIChatBehavior(
                    OpenAIChatProfile(
                        OpenAIChatRequestPolicy(
                            "TEST", ReasoningReplayMode.REASONING_CONTENT
                        ),
                        NO_REASONING,
                    )
                ),
                log_raw_sse_events=False,
                log_api_error_tracebacks=False,
            )
        else:
            transport = OpenAIResponsesTransport(
                client=sdk,
                admission=admission,
                read_timeout_s=timeout,
                provider_name="TEST",
                log_raw_sse_events=False,
            )
    elif egress == "messages":
        transport = AnthropicMessagesTransport(
            client=http,
            admission=admission,
            provider_name="TEST",
            replay_scope="test",
            read_timeout_s=timeout,
        )
    elif egress == "gemini":
        transport = GeminiTransport(
            http, admission, HistoryModeCache(), read_timeout=timeout
        )

    def stream():
        if egress == "native_messages":
            return stream_native_messages(
                http,
                admission,
                base_url="https://provider.invalid/v1",
                headers={},
                body={
                    "model": "model",
                    "stream": True,
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hello"}],
                },
                public_model="public",
                provider_name="TEST",
                read_timeout_s=timeout,
                request_id="keepalive",
            )
        if isinstance(transport, GeminiTransport):
            return transport.stream(
                {"model": "model", "contents": [{"parts": [{"text": "hello"}]}]},
                endpoint_context=Endpoint(),
                request_id="keepalive",
                response_model="public",
                output_factory=lambda: None,
            )
        if isinstance(transport, AnthropicMessagesTransport):
            return transport.stream_messages(
                make_messages_request("model"),
                request_id="keepalive",
                response_model="public",
                reasoning=ReasoningPolicy.provider_default(),
                endpoint_context=Endpoint(),
            )
        assert isinstance(transport, (OpenAIChatTransport, OpenAIResponsesTransport))
        return transport.stream_messages(
            make_messages_request("model"),
            request_id="keepalive",
            response_model="public",
            reasoning=ReasoningPolicy.provider_default(),
            input_tokens=0,
        )

    try:
        yield stream, admission
    finally:
        if sdk is not None:
            await sdk.close()
        await http.aclose()
        admission.close()


async def collect(stream):
    return [event async for event in stream]


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", EGRESSES)
@pytest.mark.parametrize(
    "frame",
    [
        b": keep-alive\n\n",
        b'event: ping\ndata: {"type":"ping"}\n\n',
        b"event: ping\ndata: {}\n\n",
    ],
)
async def test_keepalive_only_upstream_times_out_and_releases_capacity(egress, frame):
    if egress in {"chat", "responses"} and b"data: {}" in frame:
        pytest.skip("SDK does not retain arbitrary SSE names")
    wire = Heartbeats(frame)
    async with upstream(egress, lambda: wire) as (stream, admission):
        with pytest.raises(ExecutionFailure) as caught:
            async with asyncio.timeout(1):
                await collect(stream())
        assert caught.value.kind is FailureKind.TIMEOUT
        assert wire.reads >= 1
        assert wire.closed
        async with asyncio.timeout(1):
            attempt = await admission.start_execution().open_attempt(
                ProviderOperationKind.GENERATION
            )
        await attempt.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", EGRESSES)
async def test_keepalive_timeout_uses_existing_attempt_budget(egress):
    wires = []

    def body():
        assert all(wire.closed for wire in wires)
        wire = Heartbeats(b": keep-alive\n\n")
        wires.append(wire)
        return wire

    async with upstream(egress, body, attempts=2) as (stream, _):
        with pytest.raises(ExecutionFailure) as caught:
            async with asyncio.timeout(1):
                await collect(stream())
        assert caught.value.kind is FailureKind.TIMEOUT
        assert len(wires) == 2 and all(wire.closed for wire in wires)


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", EGRESSES)
async def test_cancellation_during_heartbeats_closes_and_releases(egress):
    wire = Heartbeats(b": keep-alive\n\n")
    async with upstream(egress, lambda: wire, timeout=10) as (stream, admission):
        task = asyncio.create_task(collect(stream()))
        try:
            async with asyncio.timeout(1):
                await wire.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert wire.closed
            async with asyncio.timeout(1):
                attempt = await admission.start_execution().open_attempt(
                    ProviderOperationKind.GENERATION
                )
            await attempt.aclose()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def successful_frames(egress):
    if egress == "chat":
        return [
            _chat_chunk("hello"),
            _chat_chunk("", finish_reason="stop"),
            b"data: [DONE]\n\n",
        ]
    if egress == "responses":
        return [
            b"data: " + json.dumps(event).encode() + b"\n\n"
            for event in (_text_delta("hello"), _completed_event())
        ]
    if egress in {"messages", "native_messages"}:
        return [_sse(event) for event in _events()]
    return [
        b"data: " + json.dumps(payload).encode() + b"\n\n"
        for payload in (
            {
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "hello"}]}}
                ]
            },
            {"candidates": [{"finishReason": "STOP"}]},
        )
    ]


class Frames(httpx.AsyncByteStream, httpx2.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", EGRESSES)
async def test_pings_are_discarded_before_and_during_valid_responses(egress):
    ping = b'event: ping\ndata: {"type":"ping"}\n\n'
    raw = b"".join(ping + frame for frame in successful_frames(egress))
    # Exercise framing across arbitrary network chunk boundaries as well.
    wire = Frames([raw[index : index + 7] for index in range(0, len(raw), 7)])
    async with upstream(egress, lambda: wire, timeout=1) as (stream, _):
        async with asyncio.timeout(2):
            result = "".join(await collect(stream()))
        assert "hello" in result
        assert '"ping"' not in result
        assert wire.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", EGRESSES)
async def test_timeout_after_public_output_preserves_prefix_and_reports_failure(egress):
    frames = successful_frames(egress)
    prefix = b"".join(frames[:3] if "messages" in egress else frames[:1])
    wire = Heartbeats(b": keep-alive\n\n", prefix=prefix)
    async with upstream(egress, lambda: wire) as (stream, _):
        respond = (
            gemini_sse_streaming_response
            if egress == "gemini"
            else anthropic_sse_streaming_response
        )
        async with asyncio.timeout(2):
            response = await respond(
                stream(),
                request_id="keepalive",
                pre_start_error_response=lambda error: JSONResponse(
                    {"error": str(error)}, status_code=502
                ),
            )
            assert isinstance(response, ManagedStreamingResponse)
            try:
                result = "".join(await collect(response.body_iterator))
            finally:
                await response.aclose()
        assert result.count("hello") == 1
        assert '"error"' in result
        assert wire.closed


@pytest.mark.asyncio
async def test_chat_buffered_recovery_read_also_times_out():
    prefix = "x" * 70000  # Exceed the private workflow's early holdback.
    first = Frames([_chat_chunk(prefix)])
    recovery = Heartbeats(b": keep-alive\n\n")
    bodies = iter([first, recovery])
    async with upstream("chat", lambda: next(bodies), attempts=2) as (stream, _):
        # A failed buffered repair preserves the original truncation failure.
        with pytest.raises(ExecutionFailure, match="without finish_reason"):
            async with asyncio.timeout(2):
                await collect(stream())
        assert first.closed and recovery.closed
        assert recovery.reads >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", ["chat", "gemini"])
@pytest.mark.parametrize("done", [False, True])
async def test_stopped_generation_is_not_retried_when_tail_keeps_sending_pings(
    egress, done
):
    if egress == "gemini" and done:
        pytest.skip("Gemini has no DONE sentinel")
    frames = successful_frames(egress)
    prefix = b"".join(frames[:-1] if egress == "chat" and not done else frames)
    wire = Heartbeats(b": keep-alive\n\n", prefix=prefix)
    requests = 0

    def body():
        nonlocal requests
        requests += 1
        return wire

    # Deliberately shorter than the SDK's bounded HTTP/1.1 post-DONE drain.
    async with upstream(egress, body, timeout=0.02, attempts=2) as (stream, _):
        respond = (
            gemini_sse_streaming_response
            if egress == "gemini"
            else anthropic_sse_streaming_response
        )
        async with asyncio.timeout(2):
            response = await respond(
                stream(),
                request_id="keepalive",
                pre_start_error_response=lambda error: JSONResponse(
                    {"error": str(error)}, status_code=502
                ),
            )
            assert isinstance(response, ManagedStreamingResponse)
            try:
                result = "".join(await collect(response.body_iterator))
            finally:
                await response.aclose()
        assert '"error"' in result
        assert requests == 1 and wire.closed
