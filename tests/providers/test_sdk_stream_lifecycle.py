"""Real SDK response ownership across provider stream boundaries."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.mistral import MistralProvider
from free_claude_code.providers.mistral.reasoning import normalize_mistral_stream
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.nvidia_nim.native_tool_stream import (
    NimNativeToolProtocolError,
)
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport
from free_claude_code.providers.openai_stream import OpenAIStreamAdapter
from tests.providers.request_factory import make_messages_request
from tests.providers.support import make_provider_config, profiled_provider
from tests.providers.test_openai_responses_transport import (
    _completed_event,
    _sse,
    _text_delta,
)


class ResponseBody(httpx2.AsyncByteStream):
    def __init__(self, prefix=b"", suffix=b"", *, close_error=None, block_close=False):
        self.prefix = prefix
        self.suffix = suffix
        self.read_started = asyncio.Event()
        self.read_release = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_error = close_error
        self.block_close = block_close
        self.close_count = 0

    async def __aiter__(self):
        if self.prefix:
            yield self.prefix
        self.read_started.set()
        await self.read_release.wait()
        yield self.suffix

    async def aclose(self):
        self.close_count += 1
        self.close_started.set()
        if self.block_close:
            await self.close_release.wait()
        if self.close_error is not None:
            raise self.close_error


def _chat_chunk(text, *, finish_reason=None):
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


def _chunks(kind, text):
    if kind == "responses":
        return _sse(_text_delta(text)).encode(), _sse(_completed_event()).encode()
    return _chat_chunk(text), _chat_chunk(
        "", finish_reason="stop"
    ) + b"data: [DONE]\n\n"


def _response(body):
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, stream=body
    )


@asynccontextmanager
async def _harness(kind, handler):
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
    retained_sdk_streams = []
    endpoint = client.responses if kind == "responses" else client.chat.completions
    create = endpoint.create

    async def retain_stream(**kwargs):
        stream = await create(**kwargs)
        retained_sdk_streams.append(stream)
        return stream

    try:
        with (
            patch.object(endpoint, "create", side_effect=retain_stream),
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
                provider = NvidiaNimProvider(
                    config, nim_settings=NimSettings(), admission=admission
                )
            else:
                provider = profiled_provider("cerebras", config, admission=admission)
            yield provider, client, admission
    finally:
        # Keep references until after assertions; GC must not conceal missing cleanup.
        for stream in retained_sdk_streams:
            await stream.close()
        await client.close()


def _public_stream(provider, wire):
    kwargs = {}
    if isinstance(provider, OpenAIResponsesTransport):
        kwargs = {
            "input_tokens": 1,
            "request_id": "lifecycle",
            "response_model": "model",
            "reasoning": DEFAULT_REASONING_POLICY,
        }
    request = (
        make_messages_request("model")
        if wire == "messages"
        else OpenAIResponsesRequest(model="model", input="hello")
    )
    return getattr(provider, f"stream_{wire}")(request, **kwargs)


async def _assert_admission_available(admission):
    async with asyncio.timeout(1):
        attempt = await admission.start_execution().open_attempt(
            ProviderOperationKind.GENERATION
        )
    await attempt.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["chat", "mistral", "nim", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("termination", ["close", "cancel", "complete"])
async def test_sdk_body_lifetime(kind, wire, termination):
    prefix, suffix = _chunks(kind, "x" * 70_000)
    body = ResponseBody(prefix if termination != "cancel" else b"", suffix)
    if termination == "complete":
        body.read_release.set()

    def handler(request):
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
                task = asyncio.create_task(anext(stream))
                await asyncio.wait_for(body.read_started.wait(), timeout=2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif termination == "complete":
                async with asyncio.timeout(2):
                    output = "".join([event async for event in stream])
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
    async with _harness("chat", lambda request: _response(body)) as (
        provider,
        client,
        admission,
    ):
        task = None
        with patch.object(
            provider._behavior,
            "normalize_stream",
            side_effect=ValueError("invalid normalization"),
        ):
            try:
                task = asyncio.create_task(
                    provider._chat._create_stream(
                        {"model": "model", "messages": []},
                        admission.start_execution(),
                        ProviderOperationKind.GENERATION,
                    )
                )
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
    source = OpenAIStreamAdapter(sdk_stream)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_nim_parser_failure_closes_response_before_retry(wire):
    namespace = "]<]minimax[>["
    malformed = (
        f"{namespace}<tool_call>"
        f'{namespace}<invoke name="">'
        f"{namespace}</invoke>"
        f"{namespace}</tool_call>"
    )
    body = ResponseBody(_chat_chunk(malformed, finish_reason="stop"))
    requests = 0

    def handler(request):
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
        async with asyncio.timeout(2):
            output = "".join([event async for event in _public_stream(provider, wire)])
        assert "recovered" in output
        assert namespace not in output
        assert requests == 2
        assert not client.is_closed()
        await _assert_admission_available(admission)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        ProviderOperationKind.GENERATION,
        ProviderOperationKind.CONTINUATION,
        ProviderOperationKind.TOOL_REPAIR,
    ],
)
async def test_normalizer_construction_failure_closes_response_before_retry(operation):
    bodies = [ResponseBody(), ResponseBody()]
    remaining = iter(bodies)

    def handler(request):
        body = next(remaining)
        if body is bodies[1]:
            assert bodies[0].close_count == 1
        return _response(body)

    async with _harness("chat", handler) as (provider, client, admission):
        normalize = provider._behavior.normalize_stream
        calls = 0

        def fail_once(stream, body):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise NimNativeToolProtocolError("invalid normalization")
            assert bodies[0].close_count == 1
            return normalize(stream, body)

        with patch.object(
            provider._behavior, "normalize_stream", side_effect=fail_once
        ):
            async with asyncio.timeout(2):
                stream, _, attempt, _ = await provider._chat._create_stream(
                    {"model": "model", "messages": []},
                    admission.start_execution(),
                    operation,
                )
        try:
            assert calls == 2
            assert bodies[0].close_count == 1
            assert not client.is_closed()
        finally:
            await stream.aclose()
            await attempt.aclose()
        assert bodies[1].close_count == 1
        await _assert_admission_available(admission)
