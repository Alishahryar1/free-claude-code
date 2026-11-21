"""Real SDK response ownership across provider stream boundaries."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Literal, overload, override
from unittest.mock import patch

import httpx2
import pytest
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk
from starlette.responses import StreamingResponse

from free_claude_code.api.response_streams import (
    anthropic_sse_streaming_response,
    openai_responses_sse_streaming_response,
)
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.chat_completions.response import chat_failure
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.mistral import MistralProvider
from free_claude_code.providers.mistral.reasoning import normalize_mistral_stream
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.openai_chat.provider import OpenAIChatProvider
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport
from free_claude_code.providers.openai_stream import OpenAIStreamAdapter
from free_claude_code.providers.recovery_errors import RetryableProviderProtocolError
from tests.api.test_response_streams import _json_error, _serve
from tests.providers.request_factory import make_messages_request
from tests.providers.support import make_provider_config, profiled_provider
from tests.providers.test_openai_responses_transport import (
    _completed_event,
    _sse,
    _text_delta,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["chat", "responses"])
async def test_sdk_reader_only_discards_identifiable_pings(kind: str):
    payloads: list[JsonObject] = [
        {},
        {"choices": []},
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"usage": None},
        {"usage": {"total_tokens": 3}},
        {"type": "response.in_progress", "opaque": {"state": "retained"}},
        {"choices": [{"index": 0, "delta": {"reasoning_content": "thought"}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": "{"}}]
                    },
                }
            ]
        },
    ]
    ping = 'event: ping\ndata: {"type":"ping"}\n\n'
    wire = "".join(ping + "data: " + json.dumps(event) + "\n\n" for event in payloads)
    # The SDK loses arbitrary SSE names. An unnamed empty object must survive.
    wire += "event: ping\ndata: {}\n\n"
    async with AsyncOpenAI(
        api_key="test",
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda request: httpx2.Response(
                    200, headers={"content-type": "text/event-stream"}, content=wire
                )
            )
        ),
    ) as client:
        if kind == "chat":
            adapter = OpenAIStreamAdapter(
                await client.chat.completions.create(
                    model="model", messages=[], stream=True
                ),
                read_timeout_s=1,
            )
        else:
            adapter = OpenAIStreamAdapter(
                await client.responses.create(
                    model="model", input="hello", stream=True
                ),
                read_timeout_s=1,
            )
        try:
            result = [event.model_dump(exclude_unset=True) async for event in adapter]
        finally:
            await adapter.aclose()
        assert result == [*payloads, {}]


class ResponseBody(httpx2.AsyncByteStream):
    def __init__(
        self,
        prefix: bytes = b"",
        suffix: bytes = b"",
        *,
        close_error: Exception | None = None,
        block_close: bool = False,
    ):
        self.prefix = prefix
        self.suffix = suffix
        self.read_started = asyncio.Event()
        self.read_release = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_error = close_error
        self.block_close = block_close
        self.close_count = 0

    @override
    async def __aiter__(self):
        if self.prefix:
            yield self.prefix
        self.read_started.set()
        await self.read_release.wait()
        yield self.suffix

    @override
    async def aclose(self):
        self.close_count += 1
        self.close_started.set()
        if self.block_close:
            await self.close_release.wait()
        if self.close_error is not None:
            raise self.close_error


def _chat_chunk(text: bool | int | str, *, finish_reason: str | None = None):
    return _sse(
        {
            "id": "chat_test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "model",
            "choices": [
                {"index": 0, "delta": {"content": text}, "finish_reason": finish_reason}
            ],
        }
    ).encode()


def _chunks(kind: str, text: str):
    if kind == "responses":
        return _sse(_text_delta(text)).encode(), _sse(_completed_event()).encode()
    return _chat_chunk(text), _chat_chunk(
        "", finish_reason="stop"
    ) + b"data: [DONE]\n\n"


def _response(body: ResponseBody):
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, stream=body
    )


@overload
@asynccontextmanager
def _harness(
    kind: Literal["chat", "mistral", "nim"],
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> AsyncGenerator[
    tuple[OpenAIChatProvider, AsyncOpenAI, ProviderAdmissionController]
]: ...


@overload
@asynccontextmanager
def _harness(
    kind: Literal["responses"],
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> AsyncGenerator[
    tuple[OpenAIResponsesTransport, AsyncOpenAI, ProviderAdmissionController]
]: ...


@overload
@asynccontextmanager
def _harness(
    kind: Literal["chat", "mistral", "nim", "responses"],
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> AsyncGenerator[
    tuple[
        OpenAIChatProvider | OpenAIResponsesTransport,
        AsyncOpenAI,
        ProviderAdmissionController,
    ]
]: ...


@asynccontextmanager
async def _harness(
    kind: Literal["chat", "mistral", "nim", "responses"],
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> AsyncGenerator[
    tuple[
        OpenAIChatProvider | OpenAIResponsesTransport,
        AsyncOpenAI,
        ProviderAdmissionController,
    ]
]:
    client = AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    admission = ProviderAdmissionController(
        provider_name="TEST",
        rate_limit=1_000_000,
        rate_window=1.0,
        max_concurrency=1,
        base_delay=0.0,
        max_delay=0.0,
        jitter=0.0,
    )
    retained_sdk_streams: list[AsyncStream[object]] = []
    endpoint = client.responses if kind == "responses" else client.chat.completions

    def retaining_streams[**P, Result](
        create: Callable[P, Awaitable[Result]],
    ) -> Callable[P, Awaitable[Result]]:
        async def retain(*args: P.args, **kwargs: P.kwargs) -> Result:
            stream = await create(*args, **kwargs)
            assert isinstance(stream, AsyncStream)
            retained_sdk_streams.append(stream)
            return stream

        return retain

    try:
        with (
            patch.object(
                endpoint, "create", side_effect=retaining_streams(endpoint.create)
            ),
            patch(
                "free_claude_code.providers.openai_chat.provider.create_chat_client",
                return_value=client,
            ),
        ):
            config = make_provider_config(
                api_key="test", base_url="https://provider.invalid/v1"
            )
            if kind == "responses":
                provider = OpenAIResponsesTransport(
                    client=client,
                    admission=admission,
                    provider_name="TEST",
                    read_timeout_s=2.0,
                    log_raw_sse_events=False,
                )
            elif kind == "mistral":
                provider = MistralProvider(config, admission=admission)
            elif kind == "nim":
                provider = NvidiaNimProvider(config, admission=admission)
            else:
                provider = profiled_provider("cerebras", config, admission=admission)
            yield provider, client, admission
    finally:
        # Keep references until after assertions; GC must not conceal missing cleanup.
        for stream in retained_sdk_streams:
            await stream.close()
        await client.close()


def _public_stream(
    provider: OpenAIChatProvider | OpenAIResponsesTransport,
    wire: Literal["messages", "responses"],
) -> AsyncIterator[StreamOutput]:
    if isinstance(provider, OpenAIResponsesTransport):
        if wire == "messages":
            return provider.stream_messages(
                make_messages_request("model"),
                input_tokens=1,
                request_id="lifecycle",
                response_model="model",
                reasoning=DEFAULT_REASONING_POLICY,
            )
        return provider.stream_responses(
            OpenAIResponsesRequest(model="model", input="hello"),
            input_tokens=1,
            request_id="lifecycle",
            response_model="model",
            reasoning=DEFAULT_REASONING_POLICY,
        )
    if wire == "messages":
        return provider.stream_messages(make_messages_request("model"))
    return provider.stream_responses(
        OpenAIResponsesRequest(model="model", input="hello")
    )


async def _assert_admission_available(admission: ProviderAdmissionController):
    async with asyncio.timeout(1):
        attempt = await admission.start_execution().open_attempt(
            ProviderOperationKind.GENERATION
        )
    await attempt.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["chat", "mistral", "nim", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("termination", ["close", "cancel", "complete"])
async def test_sdk_body_lifetime(
    kind: Literal["chat", "mistral", "nim", "responses"],
    wire: Literal["messages", "responses"],
    termination: str,
):
    prefix, suffix = _chunks(kind, "x" * 70_000)
    body = ResponseBody(prefix if termination != "cancel" else b"", suffix)
    if termination == "complete":
        body.read_release.set()

    def handler(request: httpx2.Request):
        if request.url.path.endswith("/health"):
            return httpx2.Response(200, json={"ok": True})
        return _response(body)

    async with _harness(kind, handler) as (
        provider,
        client,
        admission,
    ):
        stream = _public_stream(provider, wire)
        assert isinstance(stream, AsyncCloseable)
        task = None
        try:
            if termination == "cancel":
                task = asyncio.ensure_future(anext(stream))
                await asyncio.wait_for(body.read_started.wait(), timeout=2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif termination == "complete":
                async with asyncio.timeout(2):
                    output = "".join(
                        [frame async for event in stream for frame in event.frames]
                    )
                assert "x" * 100 in output
            else:
                async with asyncio.timeout(2):
                    assert await anext(stream)
                await stream.aclose()
                await stream.aclose()
            assert body.close_count == 1
            assert not client.is_closed()
            await _assert_admission_available(admission)
            # The same client must still be able to make another SDK request.
            body.read_release.set()
            fresh = await client.get("/health", cast_to=httpx2.Response)
            assert fresh.status_code == 200
        finally:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            body.read_release.set()
            await stream.aclose()


@pytest.mark.asyncio
async def test_cancelled_normalizer_construction_cleanup_releases_admission():
    body = ResponseBody(block_close=True)

    def callback(request: object):
        return _response(body)

    async with _harness("chat", callback) as (
        provider,
        client,
        admission,
    ):

        async def consume():
            async for _ in _public_stream(provider, "messages"):
                pass

        task = None
        with patch.object(
            provider._behavior,
            "normalize_stream",
            side_effect=ValueError("invalid normalization"),
        ):
            try:
                task = asyncio.create_task(consume())
                await asyncio.wait_for(body.close_started.wait(), timeout=2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                await _assert_admission_available(admission)
                assert not client.is_closed()
            finally:
                body.close_release.set()
                if task is not None:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_cancelled_retry_cleanup_releases_recovery_ownership(
    wire: Literal["messages", "responses"],
):
    failed_body = ResponseBody(block_close=True)
    healthy_body = ResponseBody(*_chunks("chat", "recovered"))
    healthy_body.read_release.set()
    bodies = iter((failed_body, healthy_body))

    def callback(request: object):
        return _response(next(bodies))

    async with _harness("chat", callback) as (
        provider,
        client,
        admission,
    ):

        async def consume():
            return "".join(
                [
                    frame
                    async for packet in _public_stream(provider, wire)
                    for frame in packet.frames
                ]
            )

        task = None
        try:
            with patch.object(
                provider._behavior,
                "normalize_stream",
                side_effect=TimeoutError("retryable stream setup failure"),
            ):
                task = asyncio.create_task(consume())
                await asyncio.wait_for(failed_body.close_started.wait(), 2)
                assert admission._episodes["model"].leader is not None
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert not admission._episodes
            assert admission._capacity._active_attempts == 0
            async with asyncio.timeout(2):
                assert "recovered" in await consume()
            assert not client.is_closed()
            assert failed_body.close_count == 1
        finally:
            failed_body.close_release.set()
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_mistral_cleanup_preserves_normalization_error():
    body = ResponseBody(
        _chat_chunk("hello"), close_error=RuntimeError("cleanup secret-token")
    )
    client = AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(lambda request: _response(body))
        ),
    )
    sdk_stream = await client.chat.completions.create(
        model="model", messages=[], stream=True
    )
    source = OpenAIStreamAdapter(sdk_stream, read_timeout_s=2)
    stream = normalize_mistral_stream(source)
    original = ValueError("invalid normalization")
    try:
        with (
            patch(
                "free_claude_code.providers.mistral.reasoning.normalize_mistral_chunk",
                side_effect=original,
            ),
            patch("free_claude_code.providers.http.trace_event") as trace,
            pytest.raises(ValueError) as caught,
        ):
            await anext(stream)
        assert caught.value is original
        assert body.close_count == 1
        assert not client.is_closed()
        assert trace.call_args.kwargs["preserved_exc_type"] == "ValueError"
        assert trace.call_args.kwargs["close_exc_type"] == "RuntimeError"
        assert "secret-token" not in repr(trace.call_args)
    finally:
        body.close_error = None
        body.read_release.set()
        assert isinstance(stream, AsyncCloseable)
        await stream.aclose()
        await sdk_stream.close()
        await client.close()


class _RejectingNormalizedStream:
    def __init__(self, source: OpenAIStreamAdapter[ChatCompletionChunk]):
        self.source = source

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await anext(self.source)
        if chunk.choices and chunk.choices[0].delta.content == "reject normalization":
            raise RetryableProviderProtocolError("invalid normalized chunk")
        return chunk

    async def aclose(self):
        await self.source.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_normalization_failure_closes_response_before_retry(
    wire: Literal["messages", "responses"],
):
    body = ResponseBody(_chat_chunk("reject normalization"))
    requests = 0

    def handler(request: object):
        nonlocal requests
        requests += 1
        if requests == 1:
            return _response(body)
        assert body.close_count == 1
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_chat_chunk("recovered", finish_reason="stop")
            + b"data: [DONE]\n\n",
        )

    async with _harness("nim", handler) as (provider, client, admission):

        def side_effect_2(
            stream: OpenAIStreamAdapter[ChatCompletionChunk], body: object
        ):
            return _RejectingNormalizedStream(stream)

        with patch.object(
            provider._behavior,
            "normalize_stream",
            side_effect=side_effect_2,
        ):
            async with asyncio.timeout(2):
                output = "".join(
                    [
                        frame
                        async for packet in _public_stream(provider, wire)
                        for frame in packet.frames
                    ]
                )
        assert "recovered" in output
        assert "reject normalization" not in output
        assert requests == 2
        assert not client.is_closed()
        await _assert_admission_available(admission)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_rejected_first_stopped_chunk_returns_error_and_releases_admission(
    wire: Literal["messages", "responses"],
):
    body = ResponseBody(_chat_chunk("reject normalization", finish_reason="length"))
    requests = 0

    def handler(request: object):
        nonlocal requests
        requests += 1
        if requests == 1:
            return _response(body)
        assert body.close_count == 1
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_chat_chunk("unexpected retry", finish_reason="stop")
            + b"data: [DONE]\n\n",
        )

    async with _harness("nim", handler) as (provider, client, admission):
        options = {"pre_start_error_response": _json_error, "request_id": "raw-stop"}
        source = _public_stream(provider, wire)

        def side_effect_2(
            stream: OpenAIStreamAdapter[ChatCompletionChunk], body: object
        ):
            return _RejectingNormalizedStream(stream)

        with patch.object(
            provider._behavior,
            "normalize_stream",
            side_effect=side_effect_2,
        ):
            async with asyncio.timeout(2):
                response = await (
                    anthropic_sse_streaming_response(
                        source,
                        pre_start_error_response=_json_error,
                        request_id="raw-stop",
                    )
                    if wire == "messages"
                    else openai_responses_sse_streaming_response(
                        source, headers={}, **options
                    )
                )
                if isinstance(response, StreamingResponse):
                    await _serve(response)
        assert response.status_code >= 400
        assert requests == 1
        assert body.close_count == 1 and not client.is_closed()
        await _assert_admission_available(admission)


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_normalization", [False, True])
async def test_sdk_observer_runs_before_normalization_without_owning_cleanup(
    reject_normalization: bool,
):
    text = "reject normalization" if reject_normalization else "hello"
    body = ResponseBody(_chat_chunk(text, finish_reason="stop"))
    seen: list[ChatCompletionChunk] = []

    def callback(_: object):
        return _response(body)

    async with _harness("nim", callback) as (_, client, admission):
        sdk_stream = await client.chat.completions.create(
            model="model", messages=[], stream=True
        )
        source = OpenAIStreamAdapter(sdk_stream, read_timeout_s=2, on_event=seen.append)
        normalized = (
            _RejectingNormalizedStream(source) if reject_normalization else source
        )
        try:
            if reject_normalization:
                with pytest.raises(RetryableProviderProtocolError):
                    await anext(normalized)
            else:
                event = await anext(normalized)
                assert seen[0] is event
            assert len(seen) == 1
            assert seen[0].choices[0].finish_reason == "stop"
        finally:
            assert isinstance(normalized, AsyncCloseable)
            await normalized.aclose()
        assert body.close_count == 1 and not client.is_closed()
        await _assert_admission_available(admission)


@pytest.mark.asyncio
async def test_chunk_transform_runs_before_observer_and_returns_same_event():
    body = ResponseBody(_chat_chunk(0, finish_reason="stop"))
    seen = []

    def transform(chunk: ChatCompletionChunk):
        choice = chunk.choices[0]
        return chunk.model_copy(
            update={
                "choices": [
                    choice.model_copy(
                        update={
                            "delta": choice.delta.model_copy(update={"content": "0"}),
                        }
                    )
                ]
            }
        )

    def callback(_: object):
        return _response(body)

    async with _harness("chat", callback) as (_, client, _):
        sdk_stream = await client.chat.completions.create(
            model="model", messages=[], stream=True
        )
        source = OpenAIStreamAdapter(
            sdk_stream,
            read_timeout_s=2,
            on_event=seen.append,
            transform_event=transform,
        )
        try:
            event = await anext(source)
            assert seen == [event]
            assert seen[0] is event
            assert event.choices[0].delta.content == "0"
        finally:
            await source.aclose()
        assert body.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_chunk_transform_failure_releases_response_without_retry(
    wire: Literal["messages", "responses"],
):
    body = ResponseBody(_chat_chunk(True))
    requests = 0

    def handler(_):
        nonlocal requests
        requests += 1
        return _response(body)

    async with _harness("chat", handler) as (provider, client, admission):
        with (
            patch.object(
                provider._behavior,
                "normalize_chunk",
                side_effect=chat_failure("invalid text"),
            ),
            pytest.raises(ExecutionFailure, match="invalid text"),
        ):
            async for _ in _public_stream(provider, wire):
                pass
        assert requests == 1
        assert body.close_count == 1 and not client.is_closed()
        await _assert_admission_available(admission)


@pytest.mark.asyncio
async def test_normalizer_construction_failure_closes_response_before_retry():
    bodies = [ResponseBody(), ResponseBody(*_chunks("chat", "recovered"))]
    bodies[1].read_release.set()
    remaining = iter(bodies)

    def handler(request: object):
        body = next(remaining)
        if body is bodies[1]:
            assert bodies[0].close_count == 1
        return _response(body)

    async with _harness("chat", handler) as (provider, client, admission):
        normalize = provider._behavior.normalize_stream
        calls = 0

        def fail_once(
            stream: OpenAIStreamAdapter[ChatCompletionChunk], body: Mapping[str, object]
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RetryableProviderProtocolError("invalid normalization")
            assert bodies[0].close_count == 1
            return normalize(stream, body)

        with patch.object(
            provider._behavior, "normalize_stream", side_effect=fail_once
        ):
            async with asyncio.timeout(2):
                frames = "".join(
                    [
                        frame
                        async for packet in _public_stream(provider, "messages")
                        for frame in packet.frames
                    ]
                )
        assert "recovered" in frames
        assert calls == 2
        assert bodies[0].close_count == 1
        assert bodies[1].close_count == 1
        assert not client.is_closed()
        await _assert_admission_available(admission)
