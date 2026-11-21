"""Tests for LM Studio (OpenAI-compatible chat completions) provider."""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import override
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.config.provider_catalog import LMSTUDIO_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.gemini import GeminiRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
)
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.lmstudio import LMStudioProvider
from tests.application.test_execution import (
    FakeProvider,
    ResponsesFakeProvider,
    _routed_request,
    _routed_responses_request,
    _target,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    SDKStreamDouble,
    continuation_seed,
    immediate_admission,
    make_provider_config,
    sdk_chat_chunk,
)
from tests.wire_payload import array_payload, object_array, object_payload


def make_request(**overrides: object):
    return make_messages_request("lmstudio-community/qwen2.5-7b-instruct", **overrides)


@pytest.fixture
def lmstudio_config():
    return make_provider_config(
        api_key="lm-studio",
        base_url=LMSTUDIO_DEFAULT_BASE,
    )


@pytest.fixture
def lmstudio_provider(lmstudio_config: ProviderConfig):
    provider = LMStudioProvider(lmstudio_config, admission=immediate_admission())
    with patch.object(
        provider._client,
        "get",
        side_effect=httpx2.ConnectError("offline test"),
    ):
        yield provider


def metadata_response(context: int = 100_000):
    return httpx2.Response(
        200, json={"data": [{"state": "loaded", "loaded_context_length": context}]}
    )


@asynccontextmanager
async def provider_with_http(
    handler: Callable[[httpx2.Request], httpx2.Response],
    *,
    base_url: str = LMSTUDIO_DEFAULT_BASE,
):
    config = make_provider_config(api_key="lm-studio", base_url=base_url)
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=base_url,
        organization="test-org",
        project="test-project",
        max_retries=3,
        timeout=17.0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    with patch(
        "free_claude_code.providers.openai_chat.provider.create_chat_client",
        return_value=client,
    ):
        provider = LMStudioProvider(
            config,
            admission=ProviderAdmissionController(
                provider_name="LMSTUDIO",
                rate_limit=1_000_000,
                rate_window=1.0,
                max_concurrency=1,
            ),
        )
    try:
        yield provider
    finally:
        await provider.cleanup()


class GatedBody(httpx2.AsyncByteStream):
    def __init__(
        self,
        *,
        prefix: bytes = b"",
        content: bytes = b'{"data": [{"state": "loaded", "loaded_context_length": 1000}]}',
    ):
        self.prefix = prefix
        self.content = content
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.close_count = 0

    @override
    async def __aiter__(self):
        if self.prefix:
            yield self.prefix
        self.started.set()
        await self.release.wait()
        yield self.content

    @override
    async def aclose(self):
        self.close_count += 1


def input_request(wire: str):
    if wire == "gemini":
        return GeminiRequest.model_validate(
            {"model": "model", "contents": [{"parts": [{"text": "hello"}]}]}
        )
    return (
        make_request()
        if wire == "messages"
        else OpenAIResponsesRequest(model="model", input="hello")
    )


def stream_request(
    provider: LMStudioProvider,
    wire: str,
    *,
    request: MessagesRequest | OpenAIResponsesRequest | GeminiRequest | None = None,
    reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    continuation: ContinuationSeed | None = None,
) -> AsyncIterator[StreamOutput]:
    if request is None:
        request = input_request(wire)
    if isinstance(request, MessagesRequest):
        return provider.stream_messages(
            request, reasoning=reasoning, continuation=continuation
        )
    if isinstance(request, GeminiRequest):
        return provider.stream_gemini(
            request, reasoning=reasoning, continuation=continuation
        )
    return provider.stream_responses(
        request, reasoning=reasoning, continuation=continuation
    )


def execute_stream(
    executor: ProviderExecutor,
    wire: str,
    request_id: str,
    *fallbacks: ProviderModelTarget,
) -> AsyncIterator[StreamOutput]:
    if wire == "messages":
        return executor.stream_messages(
            _routed_request(*fallbacks), raw_log_payload=dict, request_id=request_id
        )
    return executor.stream_responses(
        _routed_responses_request(*fallbacks),
        raw_log_payload=dict,
        request_id=request_id,
    )


def completion_chunk(*, finish_reason: str | None = "stop", content: str = "hello"):
    chunk = {
        "id": "chat_test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "model",
        "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def completion_response():
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=completion_chunk() + b"data: [DONE]\n\n",
    )


def test_init(lmstudio_config: ProviderConfig):
    """Test provider initialization."""
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_openai:
        provider = LMStudioProvider(lmstudio_config, admission=immediate_admission())
        assert provider._api_key == "lm-studio"
        assert provider._base_url == LMSTUDIO_DEFAULT_BASE
        assert provider._provider_name == "LMSTUDIO"
        mock_openai.assert_called_once()


def test_default_base_url_constant():
    assert LMSTUDIO_DEFAULT_BASE == "http://localhost:1234/v1"


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_context_metadata_wait_keeps_event_loop_responsive(
    lmstudio_provider: LMStudioProvider, wire: str
):
    events = []
    loop = asyncio.get_running_loop()
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "http://localhost:1234/api/v0/models"),
        json={"data": [{"state": "loaded", "loaded_context_length": 1}]},
    )

    def synchronous_lookup(*args: object, **kwargs: object):
        loop.call_soon_threadsafe(events.append, "peer ran")
        time.sleep(0.02)
        events.append("metadata returned")
        return response

    async def asynchronous_lookup(*args: object, **kwargs: object):
        loop.call_soon_threadsafe(events.append, "peer ran")
        await asyncio.sleep(0)
        events.append("metadata returned")
        return response

    request = input_request(wire)
    stream = None
    with (
        patch("httpx.get", side_effect=synchronous_lookup),
        patch.object(lmstudio_provider._client, "get", side_effect=asynchronous_lookup),
    ):
        try:
            with pytest.raises(ExecutionFailure) as error:
                stream = stream_request(lmstudio_provider, wire, request=request)
                await anext(stream)
            assert error.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
        finally:
            if stream is not None:
                assert isinstance(stream, AsyncCloseable)
                await stream.aclose()
            await asyncio.sleep(0)
    assert events == ["peer ran", "metadata returned"]


def test_build_request_body_basic(lmstudio_provider: LMStudioProvider):
    req = make_request()
    body = lmstudio_provider._chat._build_request_body(req)

    assert body["model"] == "lmstudio-community/qwen2.5-7b-instruct"
    assert object_payload(array_payload(body["messages"])[0])["role"] == "system"


def test_adaptive_client_reasoning_uses_documented_named_effort(
    lmstudio_provider: LMStudioProvider,
):
    req = make_request()

    body = lmstudio_provider._chat._build_request_body(req, reasoning=REASONING_ON)

    assert object_payload(body["extra_body"])["reasoning_effort"] == "high"
    assert "reasoning_effort" not in body


def test_exact_client_budget_is_not_derived_from_output_tokens(
    lmstudio_provider: LMStudioProvider,
):
    req = make_request(max_tokens=8192)

    body = lmstudio_provider._chat._build_request_body(
        req,
        reasoning=ReasoningPolicy.on(
            effort=ReasoningEffort.HIGH,
            budget_tokens=1024,
        ),
    )

    assert object_payload(body["extra_body"])["reasoning_tokens"] == 1024
    assert "reasoning_tokens" not in body
    assert body["max_tokens"] == 8192


def test_build_request_body_never_replays_prior_thinking(
    lmstudio_provider: LMStudioProvider,
):
    """Retain prior thinking as ordinary context without a reasoning role."""
    req = make_request(
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "prior reasoning",
                        "signature": "s",
                    }
                ],
            },
        ]
    )
    body = lmstudio_provider._chat._build_request_body(req)

    messages: JsonValue = body.get("messages", [])
    roles = [m.get("role") for m in object_array(messages)]
    assert "assistant_reasoning_content" not in roles
    assert "[Earlier reasoning]" not in str(body)
    assert "prior reasoning" in str(body)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_startup_builds_before_context_budget_and_preserves_policy(
    lmstudio_provider: LMStudioProvider, wire: str
):
    request = input_request(wire)
    calls: list[tuple[str, ReasoningPolicy | JsonObject]] = []
    name = (
        "_build_request_body" if wire == "messages" else "_build_responses_request_body"
    )
    messages_builder = lmstudio_provider._chat._build_request_body
    responses_builder = lmstudio_provider._chat._build_responses_request_body

    def build(
        request_arg: MessagesRequest | OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy,
    ):
        assert request_arg is request
        calls.append(("build", reasoning))
        if isinstance(request_arg, MessagesRequest):
            return messages_builder(request_arg, reasoning=reasoning)
        return responses_builder(request_arg, reasoning=reasoning)

    async def check_context(body: JsonObject):
        calls.append(("context", body))
        raise ExecutionFailure(
            FailureKind.CONTEXT_WINDOW_EXCEEDED, 400, "stop before generation", False
        )

    with (
        patch.object(lmstudio_provider._chat, name, side_effect=build),
        patch.object(
            lmstudio_provider._chat,
            "_validate_prepared_body",
            side_effect=check_context,
        ),
        patch.object(
            lmstudio_provider._chat._admission,
            "start_execution",
            wraps=lmstudio_provider._chat._admission.start_execution,
        ) as admission,
    ):
        stream = stream_request(
            lmstudio_provider, wire, request=request, reasoning=REASONING_OFF
        )
        assert calls == [("build", REASONING_OFF)]
        with pytest.raises(ExecutionFailure, match="stop before generation"):
            await anext(stream)
    assert calls[0] == ("build", REASONING_OFF)
    assert calls[1][0] == "context"
    body = calls[1][1]
    assert isinstance(body, dict)
    assert body["messages"]
    assert object_payload(body["extra_body"])["reasoning_effort"] == "none"
    admission.assert_called_once()
    assert lmstudio_provider._chat._admission._capacity._active_attempts == 0


@pytest.mark.parametrize("wire", ["messages", "responses"])
def test_startup_conversion_failure_skips_context_budget(
    lmstudio_provider: LMStudioProvider, wire: str
):
    request = input_request(wire)
    conversion_error = InvalidRequestError("invalid request conversion")

    with (
        patch.object(
            lmstudio_provider._chat,
            "_build_request_body"
            if wire == "messages"
            else "_build_responses_request_body",
            side_effect=conversion_error,
        ),
        patch.object(lmstudio_provider._chat, "_validate_prepared_body") as context,
        pytest.raises(InvalidRequestError, match="invalid request conversion"),
    ):
        stream_request(lmstudio_provider, wire, request=request, reasoning=REASONING_ON)

    context.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "gemini"])
async def test_context_rejection_releases_admission_without_generation(
    lmstudio_provider: LMStudioProvider, wire: str
):
    request = input_request(wire)
    with (
        patch.object(
            lmstudio_provider, "_loaded_context_length", new=AsyncMock(return_value=1)
        ),
        patch.object(
            lmstudio_provider._chat._admission,
            "start_execution",
            wraps=lmstudio_provider._chat._admission.start_execution,
        ) as admission,
        patch.object(
            lmstudio_provider._client.chat.completions, "create"
        ) as generation,
        pytest.raises(ExecutionFailure) as error,
    ):
        await anext(stream_request(lmstudio_provider, wire, request=request))
    assert error.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    admission.assert_called_once()
    assert lmstudio_provider._chat._admission._capacity._active_attempts == 0
    generation.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses", "gemini"])
async def test_continuation_prefix_counts_before_generation(
    lmstudio_provider: LMStudioProvider, wire: str
):
    request = input_request(wire)
    original = request.model_dump(mode="json")
    with (
        patch.object(
            lmstudio_provider,
            "_loaded_context_length",
            new=AsyncMock(return_value=2000),
        ),
        patch.object(
            lmstudio_provider._chat._admission,
            "start_execution",
            wraps=lmstudio_provider._chat._admission.start_execution,
        ) as admission,
        patch.object(
            lmstudio_provider._client.chat.completions, "create"
        ) as generation,
        pytest.raises(ExecutionFailure) as error,
    ):
        await anext(
            stream_request(
                lmstudio_provider,
                wire,
                request=request,
                continuation=continuation_seed("large " * 10000, "consider this"),
            )
        )
    assert error.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    assert request.model_dump(mode="json") == original
    admission.assert_called_once()
    assert lmstudio_provider._chat._admission._capacity._active_attempts == 0
    generation.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_cancelling_primary_context_validation_closes_lookup(
    wire: str,
):
    body = GatedBody()
    requests = []

    def handler(request: httpx2.Request):
        requests.append(request)
        return (
            httpx2.Response(200, stream=body)
            if len(requests) == 1
            else metadata_response()
        )

    async with provider_with_http(handler) as provider:
        fallback = FakeProvider() if wire == "messages" else ResponsesFakeProvider()
        providers = {"provider": provider, "fallback": fallback}

        def resolve_provider(name: str, _api_key_number: int):
            return providers[name]

        executor = ProviderExecutor(
            AsyncMock(side_effect=resolve_provider),
        )
        stream = execute_stream(
            executor, wire, "context-cancel", _target("fallback", "fallback-model")
        )
        async with asyncio.timeout(2):
            task = asyncio.ensure_future(anext(stream))
            await body.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert body.started.is_set()
        assert body.close_count == 1
        assert fallback.stream_calls == []
        assert [request.url.path for request in requests] == ["/api/v0/models"]
        async with asyncio.timeout(1):
            assert await provider._loaded_context_length() == 100_000
        assert len(requests) == 2


@pytest.mark.asyncio
async def test_stream_messages_text(lmstudio_provider: LMStudioProvider):
    """Text content deltas are emitted through the shared OpenAI-chat provider."""
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello back!", reasoning_content=None, tool_calls=None
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield sdk_chat_chunk(mock_chunk)

    with patch.object(
        lmstudio_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(mock_stream())

        events = [
            frame
            async for event in lmstudio_provider.stream_messages(req)
            for frame in event.frames
        ]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_messages_passes_exact_reasoning_budget_via_extra_body(
    lmstudio_provider: LMStudioProvider,
):
    req = make_request()
    policy = ReasoningPolicy.on(
        effort=ReasoningEffort.HIGH,
        budget_tokens=1024,
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Done", reasoning_content=None, tool_calls=None),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=1, prompt_tokens=1)

    async def mock_stream():
        yield sdk_chat_chunk(mock_chunk)

    with patch.object(
        lmstudio_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(mock_stream())

        events = [
            frame
            async for event in lmstudio_provider.stream_messages(
                req,
                reasoning=policy,
            )
            for frame in event.frames
        ]

    await_args = mock_create.await_args
    assert await_args is not None
    create_kwargs = await_args.kwargs
    assert create_kwargs["extra_body"] == {"reasoning_tokens": 1024}
    assert "reasoning_tokens" not in create_kwargs
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_cleanup(lmstudio_provider: LMStudioProvider):
    assert not lmstudio_provider._client.is_closed()
    await lmstudio_provider.cleanup()
    assert lmstudio_provider._client.is_closed()


# --- Context-budget validation (new: guards against LM Studio's silent
# mid-stream truncation when a prompt exceeds the loaded model's context) ---


@pytest.mark.asyncio
async def test_validate_context_budget_noop_when_context_length_unknown(
    lmstudio_provider: LMStudioProvider,
):
    """No LM Studio /api/v0/models data available -> validation is a no-op (fail open)."""
    with patch.object(
        lmstudio_provider, "_loaded_context_length", new=AsyncMock(return_value=None)
    ):
        await lmstudio_provider._validate_context_budget({"messages": []})


@pytest.mark.asyncio
@pytest.mark.parametrize("words", [1, 80_000])
async def test_validate_context_budget_allows_request_under_budget(
    lmstudio_provider: LMStudioProvider, words: int
):
    with patch.object(
        lmstudio_provider, "_loaded_context_length", new=AsyncMock(return_value=100_000)
    ):
        await lmstudio_provider._validate_context_budget(
            {"messages": [{"content": "word " * words}]}
        )


@pytest.mark.asyncio
async def test_validate_context_budget_rejects_request_over_90_percent(
    lmstudio_provider: LMStudioProvider,
):
    with (
        patch.object(
            lmstudio_provider,
            "_loaded_context_length",
            new=AsyncMock(return_value=1000),
        ),
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        await lmstudio_provider._validate_context_budget(
            {"messages": [{"content": "word " * 1000}]}
        )

    failure = exc_info.value
    assert failure.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    assert failure.status_code == 400
    assert failure.retryable is False
    assert "context budget (900 tokens; 90% of loaded context 1000)" in failure.message


@pytest.mark.asyncio
async def test_loaded_context_length_reads_max_across_loaded_models(
    lmstudio_provider: LMStudioProvider,
):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "data": [
            {"state": "loaded", "loaded_context_length": 40960},
            {"state": "loaded", "loaded_context_length": 8192},
            {"state": "not-loaded", "loaded_context_length": 999999},
        ]
    }
    with patch.object(
        lmstudio_provider._client, "get", return_value=response
    ) as mock_get:
        value = await lmstudio_provider._loaded_context_length()

    assert value == 40960
    mock_get.assert_called_once()
    assert mock_get.call_args[0][0] == "http://localhost:1234/api/v0/models"


@pytest.mark.asyncio
async def test_loaded_context_length_fails_open_on_error(
    lmstudio_provider: LMStudioProvider,
):
    with patch.object(
        lmstudio_provider._client,
        "get",
        side_effect=httpx2.ConnectError("refused"),
    ):
        assert await lmstudio_provider._loaded_context_length() is None


@pytest.mark.asyncio
async def test_loaded_context_length_is_cached_within_ttl(
    lmstudio_provider: LMStudioProvider,
):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "data": [{"state": "loaded", "loaded_context_length": 40960}]
    }
    with patch.object(
        lmstudio_provider._client, "get", return_value=response
    ) as mock_get:
        first = await lmstudio_provider._loaded_context_length()
        second = await lmstudio_provider._loaded_context_length()

    assert first == second == 40960
    mock_get.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_url", "metadata_url"),
    [
        ("http://localhost:1234/v1", "http://localhost:1234/api/v0/models"),
        (
            "http://localhost:1234/prefix/v1/",
            "http://localhost:1234/prefix/api/v0/models",
        ),
        ("http://localhost:1234", "http://localhost:1234/api/v0/models"),
    ],
)
@pytest.mark.parametrize("content_type", [None, "text/plain"])
async def test_metadata_http_policy_preserves_the_generation_client(
    provider_url: str, metadata_url: str, content_type: str | None
):
    requests = []

    def handler(request: httpx2.Request):
        requests.append(request)
        if request.url.path.endswith("/api/v0/models"):
            return httpx2.Response(
                200,
                headers={"content-type": content_type} if content_type else {},
                content=b'{"data": [{"state": "loaded", "loaded_context_length": 4096}]}',
            )
        return completion_response()

    async with provider_with_http(handler, base_url=provider_url) as provider:
        assert await provider._loaded_context_length() == 4096
        output = "".join(
            [
                frame
                async for packet in stream_request(provider, "messages")
                for frame in packet.frames
            ]
        )
        assert "hello" in output
        assert "message_stop" in output
        assert len(requests) == 2
        metadata, generation = requests
        assert str(metadata.url) == metadata_url
        assert not {
            "authorization",
            "openai-organization",
            "openai-project",
        }.intersection(metadata.headers)
        assert metadata.extensions["timeout"] == {
            "connect": 2.0,
            "read": 2.0,
            "write": 2.0,
            "pool": 2.0,
        }
        assert generation.headers["authorization"] == "Bearer lm-studio"
        assert generation.headers["openai-organization"] == "test-org"
        assert generation.headers["openai-project"] == "test-project"
        assert generation.extensions["timeout"]["read"] == 17.0
        assert not provider._client.is_closed()
    assert provider._client.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["timeout", "status", "redirect", "json", "shape", "empty"]
)
async def test_metadata_failure_is_cached_without_retry_and_allows_generation(
    failure: str,
):
    requests = []

    def handler(request: httpx2.Request):
        requests.append(request)
        if not request.url.path.endswith("/api/v0/models"):
            return completion_response()
        if failure == "timeout":
            raise httpx2.ReadTimeout("metadata unavailable", request=request)
        if failure == "status":
            return httpx2.Response(503, json={"error": "unavailable"})
        if failure == "redirect":
            return httpx2.Response(
                302, headers={"location": "http://other.invalid/models"}
            )
        if failure == "json":
            return httpx2.Response(200, content=b"not JSON")
        return httpx2.Response(200, json={"data": None if failure == "shape" else []})

    async with provider_with_http(handler) as provider:
        assert await provider._loaded_context_length() is None
        assert await provider._loaded_context_length() is None
        output = "".join(
            [
                frame
                async for packet in stream_request(provider, "messages")
                for frame in packet.frames
            ]
        )
        assert "hello" in output
        assert "message_stop" in output
        assert [request.url.path for request in requests] == [
            "/api/v0/models",
            "/v1/chat/completions",
        ]


@pytest.mark.asyncio
async def test_context_cache_expires_from_lookup_completion(
    monkeypatch: pytest.MonkeyPatch,
):
    from types import SimpleNamespace

    now = 0.0
    calls = 0

    def handler(request: object):
        nonlocal now, calls
        calls += 1
        now += 10.0
        return metadata_response(4096 * calls)

    monkeypatch.setattr(
        "free_claude_code.providers.lmstudio.client.time",
        SimpleNamespace(monotonic=lambda: now),
    )
    async with provider_with_http(handler) as provider:
        assert await provider._loaded_context_length() == 4096
        now = 39.9
        assert await provider._loaded_context_length() == 4096
        now = 40.0
        assert await provider._loaded_context_length() == 8192
        assert calls == 2


@pytest.mark.asyncio
async def test_concurrent_requests_share_lookup_but_validate_their_own_prompt():
    body = GatedBody()
    calls = 0

    def handler(request: object):
        nonlocal calls
        calls += 1
        return httpx2.Response(200, stream=body)

    async def validate(provider: LMStudioProvider, estimate: int):
        try:
            await provider._validate_context_budget(
                {"messages": [{"content": "word " * estimate}]}
            )
        except ExecutionFailure as failure:
            return failure.kind
        return None

    async with provider_with_http(handler) as provider, asyncio.TaskGroup() as group:
        small = group.create_task(validate(provider, 1))
        await asyncio.wait_for(body.started.wait(), 2)
        large = group.create_task(validate(provider, 901))
        await asyncio.sleep(0)
        assert not small.done() and not large.done()
        assert calls == 1
        body.release.set()
        assert await small is None
        assert await large is FailureKind.CONTEXT_WINDOW_EXCEEDED
        assert calls == 1
        assert body.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_owner", [False, True])
async def test_cancelled_context_lookup_does_not_poison_other_requests(
    cancel_owner: bool,
):
    body = GatedBody()
    calls = 0

    def handler(request: object):
        nonlocal calls
        calls += 1
        return (
            httpx2.Response(200, stream=body) if calls == 1 else metadata_response(4096)
        )

    async with provider_with_http(handler) as provider, asyncio.TaskGroup() as group:
        owner = group.create_task(provider._loaded_context_length())
        await asyncio.wait_for(body.started.wait(), 2)
        waiter = group.create_task(provider._loaded_context_length())
        await asyncio.sleep(0)
        cancelled = owner if cancel_owner else waiter
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        if cancel_owner:
            assert await asyncio.wait_for(waiter, 2) == 4096
            assert calls == 2
        else:
            assert not owner.done()
            assert body.close_count == 0
            body.release.set()
            assert await owner == 1000
            assert calls == 1
        assert body.close_count == 1
        assert not provider._client.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_request_cancellation_while_waiting_for_cache_leaves_refresh_running(
    wire: str,
):
    body = GatedBody()
    calls = 0

    def handler(request: object):
        nonlocal calls
        calls += 1
        return httpx2.Response(200, stream=body)

    async with provider_with_http(handler) as provider, asyncio.TaskGroup() as group:
        owner = group.create_task(provider._loaded_context_length())
        await asyncio.wait_for(body.started.wait(), 2)
        executor = ProviderExecutor(AsyncMock(return_value=provider))
        stream = execute_stream(executor, wire, "cache-wait-cancel")
        async with asyncio.timeout(2):
            task = asyncio.ensure_future(anext(stream))
            await asyncio.sleep(0)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not owner.done()
        assert body.close_count == 0
        body.release.set()
        assert await owner == 1000
        assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_closing_validated_stream_closes_upstream_and_releases_admission(
    wire: str,
):
    # Cross the transport's 64 KiB holdback before closing a publicly started stream.
    body = GatedBody(
        prefix=completion_chunk(finish_reason=None, content="hello " * 12_000),
        content=b"data: [DONE]\n\n",
    )
    generations = 0

    def handler(request: httpx2.Request):
        nonlocal generations
        if request.url.path.endswith("/api/v0/models"):
            return metadata_response()
        generations += 1
        if generations > 1:
            return completion_response()
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=body
        )

    async with provider_with_http(handler) as provider:
        stream = stream_request(provider, wire)
        try:
            async with asyncio.timeout(2):
                first = "".join((await anext(stream)).frames)
            assert (
                "message_start" in first
                if wire == "messages"
                else "response.created" in first
            )
        finally:
            assert isinstance(stream, AsyncCloseable)
            await stream.aclose()
        assert body.close_count == 1
        assert not provider._client.is_closed()
        async with asyncio.timeout(2):
            output = "".join(
                [
                    frame
                    async for packet in stream_request(provider, wire)
                    for frame in packet.frames
                ]
            )
        assert "hello" in output
        assert generations == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_context_rejection_before_first_event_allows_fallback(wire: str):
    requests = []

    def handler(request: httpx2.Request):
        requests.append(request)
        return metadata_response(1)

    async with provider_with_http(handler) as provider:
        fallback = FakeProvider() if wire == "messages" else ResponsesFakeProvider()
        providers = {"provider": provider, "fallback": fallback}

        def resolve_provider(name: str, _api_key_number: int):
            return providers[name]

        executor = ProviderExecutor(
            AsyncMock(side_effect=resolve_provider),
        )
        stream = execute_stream(
            executor, wire, "context-fallback", _target("fallback", "fallback-model")
        )
        output = [frame async for event in stream for frame in event.frames]
        assert output == [
            "event: message_stop\ndata: {}\n\n"
            if wire == "messages"
            else "event: response.completed\ndata: {}\n\n"
        ]
        assert len(fallback.stream_calls) == 1
        assert fallback.stream_close_calls == 1
        assert [request.url.path for request in requests] == ["/api/v0/models"]


@pytest.mark.asyncio
async def test_context_cache_belongs_to_each_provider_instance():
    def callback(request: object):
        return metadata_response(4096)

    def callback_2(request: object):
        return metadata_response(8192)

    async with (
        provider_with_http(callback) as first,
        provider_with_http(callback_2) as second,
    ):
        assert await first._loaded_context_length() == 4096
        assert await second._loaded_context_length() == 8192
