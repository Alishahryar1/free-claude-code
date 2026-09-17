"""Tests for LM Studio (OpenAI-compatible chat completions) provider."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.config.provider_catalog import LMSTUDIO_DEFAULT_BASE
from free_claude_code.core.anthropic import MessagesRequest, get_token_count
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    estimate_responses_input_tokens,
)
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
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
    immediate_admission,
    make_provider_config,
)


def make_request(**overrides):
    return make_messages_request("lmstudio-community/qwen2.5-7b-instruct", **overrides)


@pytest.fixture
def lmstudio_config():
    return make_provider_config(
        api_key="lm-studio",
        base_url=LMSTUDIO_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def lmstudio_provider(lmstudio_config):
    provider = LMStudioProvider(lmstudio_config, admission=immediate_admission())
    with patch.object(
        provider._client,
        "get",
        side_effect=httpx2.ConnectError("offline test"),
    ):
        yield provider


def metadata_response(context=100_000):
    return httpx2.Response(
        200, json={"data": [{"state": "loaded", "loaded_context_length": context}]}
    )


@asynccontextmanager
async def provider_with_http(handler, *, base_url=LMSTUDIO_DEFAULT_BASE):
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
        prefix=b"",
        content=b'{"data": [{"state": "loaded", "loaded_context_length": 1000}]}',
    ):
        self.prefix = prefix
        self.content = content
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.close_count = 0

    async def __aiter__(self):
        if self.prefix:
            yield self.prefix
        self.started.set()
        await self.release.wait()
        yield self.content

    async def aclose(self):
        self.close_count += 1


def stream_request(provider, wire):
    request = (
        make_request()
        if wire == "messages"
        else OpenAIResponsesRequest(model="model", input="hello")
    )
    return getattr(provider, f"stream_{wire}")(request)


def completion_chunk(*, finish_reason="stop", content="hello"):
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


def test_init(lmstudio_config):
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
    lmstudio_provider, wire
):
    events = []
    loop = asyncio.get_running_loop()
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "http://localhost:1234/api/v0/models"),
        json={"data": [{"state": "loaded", "loaded_context_length": 1}]},
    )

    def synchronous_lookup(*args, **kwargs):
        loop.call_soon_threadsafe(events.append, "peer ran")
        time.sleep(0.02)
        events.append("metadata returned")
        return response

    async def asynchronous_lookup(*args, **kwargs):
        loop.call_soon_threadsafe(events.append, "peer ran")
        await asyncio.sleep(0)
        events.append("metadata returned")
        return response

    request = (
        make_request()
        if wire == "messages"
        else OpenAIResponsesRequest(model="model", input="hello")
    )
    stream = None
    with (
        patch("httpx.get", side_effect=synchronous_lookup),
        patch.object(lmstudio_provider._client, "get", side_effect=asynchronous_lookup),
    ):
        try:
            with pytest.raises(ExecutionFailure) as error:
                stream = getattr(lmstudio_provider, f"stream_{wire}")(request)
                await anext(stream)
            assert error.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
        finally:
            if stream is not None:
                await stream.aclose()
            await asyncio.sleep(0)
    assert events == ["peer ran", "metadata returned"]


def test_build_request_body_basic(lmstudio_provider):
    req = make_request()
    body = lmstudio_provider._chat._build_request_body(req)

    assert body["model"] == "lmstudio-community/qwen2.5-7b-instruct"
    assert body["messages"][0]["role"] == "system"


def test_adaptive_client_reasoning_uses_documented_named_effort(lmstudio_provider):
    req = make_request()

    body = lmstudio_provider._chat._build_request_body(req, reasoning=REASONING_ON)

    assert body["extra_body"]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in body


def test_exact_client_budget_is_not_derived_from_output_tokens(lmstudio_provider):
    req = make_request(max_tokens=8192)

    body = lmstudio_provider._chat._build_request_body(
        req,
        reasoning=ReasoningPolicy.on(
            effort=ReasoningEffort.HIGH,
            budget_tokens=1024,
        ),
    )

    assert body["extra_body"]["reasoning_tokens"] == 1024
    assert "reasoning_tokens" not in body
    assert body["max_tokens"] == 8192


def test_build_request_body_never_replays_prior_thinking(lmstudio_provider):
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

    roles = [m.get("role") for m in body.get("messages", [])]
    assert "assistant_reasoning_content" not in roles
    assert "[Earlier reasoning]" in str(body)
    assert "prior reasoning" in str(body)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_startup_builds_before_context_budget_and_preserves_policy(
    lmstudio_provider, wire
):
    request = (
        make_request()
        if wire == "messages"
        else OpenAIResponsesRequest(model="model", input="hello")
    )
    calls = []
    name = (
        "_build_request_body" if wire == "messages" else "_build_responses_request_body"
    )
    original = getattr(lmstudio_provider._chat, name)

    def build(request_arg, *, reasoning):
        assert request_arg is request
        calls.append(("build", reasoning))
        return original(request_arg, reasoning=reasoning)

    async def check_context(estimate):
        calls.append(("context", estimate))
        raise RuntimeError("stop before generation")

    with (
        patch.object(lmstudio_provider._chat, name, side_effect=build),
        patch.object(
            lmstudio_provider, "_validate_context_budget", side_effect=check_context
        ),
        patch.object(
            lmstudio_provider._chat._admission, "start_execution"
        ) as admission,
    ):
        stream = getattr(lmstudio_provider, f"stream_{wire}")(
            request, reasoning=REASONING_OFF
        )
        assert calls == [("build", REASONING_OFF)]
        with pytest.raises(RuntimeError, match="stop before generation"):
            await anext(stream)
    estimate = (
        get_token_count(request.messages, request.system, request.tools)
        if isinstance(request, MessagesRequest)
        else estimate_responses_input_tokens(request)
    )
    assert calls == [("build", REASONING_OFF), ("context", estimate)]
    admission.assert_not_called()


@pytest.mark.parametrize("wire", ["messages", "responses"])
def test_startup_conversion_failure_skips_context_budget(lmstudio_provider, wire):
    request = (
        make_request()
        if wire == "messages"
        else OpenAIResponsesRequest(model="model", input="hello")
    )
    conversion_error = InvalidRequestError("invalid request conversion")

    with (
        patch.object(
            lmstudio_provider._chat,
            "_build_request_body"
            if wire == "messages"
            else "_build_responses_request_body",
            side_effect=conversion_error,
        ),
        patch.object(lmstudio_provider, "_validate_context_budget") as context,
        pytest.raises(InvalidRequestError, match="invalid request conversion"),
    ):
        getattr(lmstudio_provider, f"stream_{wire}")(request, reasoning=REASONING_ON)

    context.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_context_rejection_never_starts_admission_or_generation(
    lmstudio_provider, wire
):
    request = (
        make_request()
        if wire == "messages"
        else OpenAIResponsesRequest(model="model", input="hello")
    )
    with (
        patch.object(
            lmstudio_provider, "_loaded_context_length", new=AsyncMock(return_value=1)
        ),
        patch.object(
            lmstudio_provider._chat._admission, "start_execution"
        ) as admission,
        patch.object(
            lmstudio_provider._client.chat.completions, "create"
        ) as generation,
        pytest.raises(ExecutionFailure) as error,
    ):
        await anext(getattr(lmstudio_provider, f"stream_{wire}")(request))
    assert error.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    admission.assert_not_called()
    generation.assert_not_called()


@pytest.mark.asyncio
async def test_bound_response_continuations_validate_enlarged_context():
    from tests.api.test_responses_web_tools import chat_wire

    generations = []

    def handler(request):
        if request.url.path == "/api/v0/models":
            return metadata_response(context=1000)
        generations.append(json.loads(request.content))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=chat_wire({"role": "assistant", "content": "ok"}, "stop"),
        )

    async with provider_with_http(handler) as provider:
        request = OpenAIResponsesRequest(model="model", input="hello")
        async with provider.bind_responses(
            request,
            request_id="bound",
            response_model="public",
            reasoning=REASONING_OFF,
        ) as binding:
            assert "ok" in "".join(
                [chunk async for chunk in binding.stream(request, input_tokens=1)]
            )
            enlarged = request.model_copy(update={"input": "web result " * 10000})
            with pytest.raises(ExecutionFailure) as error:
                await anext(binding.stream(enlarged, input_tokens=1))
            assert error.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
        assert len(generations) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_primary_context_validation_counts_toward_progress_timeout(
    wire,
):
    body = GatedBody()
    requests = []

    def handler(request):
        requests.append(request)
        return (
            httpx2.Response(200, stream=body)
            if len(requests) == 1
            else metadata_response()
        )

    async with provider_with_http(handler) as provider:
        fallback = FakeProvider() if wire == "messages" else ResponsesFakeProvider()
        providers = {"provider": provider, "fallback": fallback}
        executor = ProviderExecutor(
            AsyncMock(side_effect=lambda name: providers[name]),
            progress_timeout_seconds=0.05,
        )
        route = _routed_request if wire == "messages" else _routed_responses_request
        routed = route(_target("fallback", "fallback-model"))
        stream = getattr(executor, f"stream_{wire}")(
            routed, raw_log_payload={}, request_id="context-timeout"
        )
        async with asyncio.timeout(2):
            with pytest.raises(ExecutionFailure) as error:
                await anext(stream)
        assert error.value.kind is FailureKind.TIMEOUT
        assert body.started.is_set()
        assert body.close_count == 1
        assert fallback.stream_calls == []
        assert [request.url.path for request in requests] == ["/api/v0/models"]
        async with asyncio.timeout(1):
            assert await provider._loaded_context_length() == 100_000
        assert len(requests) == 2


@pytest.mark.asyncio
async def test_stream_messages_text(lmstudio_provider):
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
        yield mock_chunk

    with patch.object(
        lmstudio_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(mock_stream())

        events = [event async for event in lmstudio_provider.stream_messages(req)]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_messages_passes_exact_reasoning_budget_via_extra_body(
    lmstudio_provider,
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
        yield mock_chunk

    with patch.object(
        lmstudio_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(mock_stream())

        events = [
            event
            async for event in lmstudio_provider.stream_messages(
                req,
                reasoning=policy,
            )
        ]

    await_args = mock_create.await_args
    assert await_args is not None
    create_kwargs = await_args.kwargs
    assert create_kwargs["extra_body"] == {"reasoning_tokens": 1024}
    assert "reasoning_tokens" not in create_kwargs
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_cleanup(lmstudio_provider):
    assert not lmstudio_provider._client.is_closed()
    await lmstudio_provider.cleanup()
    assert lmstudio_provider._client.is_closed()


# --- Context-budget validation (new: guards against LM Studio's silent
# mid-stream truncation when a prompt exceeds the loaded model's context) ---


@pytest.mark.asyncio
async def test_validate_context_budget_noop_when_context_length_unknown(
    lmstudio_provider,
):
    """No LM Studio /api/v0/models data available -> validation is a no-op (fail open)."""
    with patch.object(
        lmstudio_provider, "_loaded_context_length", new=AsyncMock(return_value=None)
    ):
        await lmstudio_provider._validate_context_budget(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("estimate", [1, 90_000])
async def test_validate_context_budget_allows_request_under_budget(
    lmstudio_provider, estimate
):
    with patch.object(
        lmstudio_provider, "_loaded_context_length", new=AsyncMock(return_value=100_000)
    ):
        await lmstudio_provider._validate_context_budget(estimate)


@pytest.mark.asyncio
async def test_validate_context_budget_rejects_request_over_90_percent(
    lmstudio_provider,
):
    with (
        patch.object(
            lmstudio_provider,
            "_loaded_context_length",
            new=AsyncMock(return_value=1000),
        ),
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        await lmstudio_provider._validate_context_budget(901)

    failure = exc_info.value
    assert failure.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    assert failure.status_code == 400
    assert failure.retryable is False
    assert failure.message == (
        "Estimated provider input (901 tokens) exceeds the safe LM Studio "
        "context budget (900 tokens; 90% of loaded context 1000)."
    )
    assert "prompt is too long" not in failure.message


@pytest.mark.asyncio
async def test_loaded_context_length_reads_max_across_loaded_models(lmstudio_provider):
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
async def test_loaded_context_length_fails_open_on_error(lmstudio_provider):
    with patch.object(
        lmstudio_provider._client,
        "get",
        side_effect=httpx2.ConnectError("refused"),
    ):
        assert await lmstudio_provider._loaded_context_length() is None


@pytest.mark.asyncio
async def test_loaded_context_length_is_cached_within_ttl(lmstudio_provider):
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
    provider_url, metadata_url, content_type
):
    requests = []

    def handler(request):
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
            [event async for event in stream_request(provider, "messages")]
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
async def test_metadata_failure_is_cached_without_retry_and_allows_generation(failure):
    requests = []

    def handler(request):
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
            [event async for event in stream_request(provider, "messages")]
        )
        assert "hello" in output
        assert "message_stop" in output
        assert [request.url.path for request in requests] == [
            "/api/v0/models",
            "/v1/chat/completions",
        ]


@pytest.mark.asyncio
async def test_context_cache_expires_from_lookup_completion(monkeypatch):
    from types import SimpleNamespace

    now = 0.0
    calls = 0

    def handler(request):
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

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx2.Response(200, stream=body)

    async def validate(provider, estimate):
        try:
            await provider._validate_context_budget(estimate)
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
async def test_cancelled_context_lookup_does_not_poison_other_requests(cancel_owner):
    body = GatedBody()
    calls = 0

    def handler(request):
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
async def test_request_timeout_while_waiting_for_cache_leaves_refresh_running(wire):
    body = GatedBody()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx2.Response(200, stream=body)

    async with provider_with_http(handler) as provider, asyncio.TaskGroup() as group:
        owner = group.create_task(provider._loaded_context_length())
        await asyncio.wait_for(body.started.wait(), 2)
        executor = ProviderExecutor(
            AsyncMock(return_value=provider), progress_timeout_seconds=0.05
        )
        routed = (
            _routed_request() if wire == "messages" else _routed_responses_request()
        )
        stream = getattr(executor, f"stream_{wire}")(
            routed, raw_log_payload={}, request_id="cache-wait-timeout"
        )
        async with asyncio.timeout(2):
            with pytest.raises(ExecutionFailure) as error:
                await anext(stream)
        assert error.value.kind is FailureKind.TIMEOUT
        assert not owner.done()
        assert body.close_count == 0
        body.release.set()
        assert await owner == 1000
        assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_closing_validated_stream_closes_upstream_and_releases_admission(wire):
    # Cross the transport's 64 KiB holdback before closing a publicly started stream.
    body = GatedBody(
        prefix=completion_chunk(finish_reason=None, content="hello " * 12_000),
        content=b"data: [DONE]\n\n",
    )
    generations = 0

    def handler(request):
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
                first = await anext(stream)
            assert (
                "message_start" in first
                if wire == "messages"
                else "response.created" in first
            )
        finally:
            await stream.aclose()
        assert body.close_count == 1
        assert not provider._client.is_closed()
        async with asyncio.timeout(2):
            output = "".join([event async for event in stream_request(provider, wire)])
        assert "hello" in output
        assert generations == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_context_rejection_before_first_event_allows_fallback(wire):
    requests = []

    def handler(request):
        requests.append(request)
        return metadata_response(1)

    async with provider_with_http(handler) as provider:
        fallback = FakeProvider() if wire == "messages" else ResponsesFakeProvider()
        providers = {"provider": provider, "fallback": fallback}
        executor = ProviderExecutor(
            AsyncMock(side_effect=lambda name: providers[name]),
            progress_timeout_seconds=2,
        )
        route = _routed_request if wire == "messages" else _routed_responses_request
        stream = getattr(executor, f"stream_{wire}")(
            route(_target("fallback", "fallback-model")),
            raw_log_payload={},
            request_id="context-fallback",
        )
        output = [event async for event in stream]
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
    async with (
        provider_with_http(lambda request: metadata_response(4096)) as first,
        provider_with_http(lambda request: metadata_response(8192)) as second,
    ):
        assert await first._loaded_context_length() == 4096
        assert await second._loaded_context_length() == 8192
