"""Provider test helpers with explicit admission ownership."""

import json
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx2
from openai import AsyncOpenAI

from free_claude_code.application.ports import ProviderCandidate
from free_claude_code.application.reasoning import client_reasoning_policy
from free_claude_code.application.recovery import RecoveryCoordinator, RecoveryWriter
from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.passthrough import NativeMessagesRequest
from free_claude_code.core.anthropic.recovery_stream import (
    MessagesRecoveryWriter,
    NativeMessagesCompletionWriter,
)
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import (
    ResponsesRecoveryWriter,
)
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.recovery import RecoveryCheckpoint
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    OpenAIChatProvider,
    create_openai_chat_provider,
)

REASONING_DEFAULT = ReasoningPolicy.provider_default()
REASONING_ON = ReasoningPolicy.on()
REASONING_OFF = ReasoningPolicy.off()


async def successful_chat_stream():
    """A minimal SDK stream with an explicit content-completion boundary."""
    yield SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="answer", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


async def exercise_chat_body(provider: Any, body: dict) -> dict:
    """Exercise correction of a prepared body through the real logical owner."""
    request = MessagesRequest.model_validate(
        {
            "model": body["model"],
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "test"}],
        }
    )
    async with provider.open_messages(request) as candidate:
        with patch.object(
            provider._chat, "_build_request_body", return_value=deepcopy(body)
        ):
            await candidate.prepare(RecoveryCheckpoint("messages"))

        @asynccontextmanager
        async def prepared():
            yield candidate

        frames = [
            frame
            async for frame in candidate_stream(
                prepared(),
                writer=MessagesRecoveryWriter(model=request.model, input_tokens=0),
            )
        ]
        assert any("message_stop" in frame for frame in frames)
        assert not any("event: error" in frame for frame in frames)
        return deepcopy(candidate.body)


def provider_stream(
    provider: Any, wire: str, request: Any, **kwargs: Any
) -> AsyncIterator[str]:
    return (stream_messages if wire == "messages" else stream_responses)(
        provider, request, **kwargs
    )


def stream_messages(
    provider: Any,
    request: MessagesRequest,
    input_tokens: int = 0,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """Exercise a provider through the application owner of its retry lifecycle."""
    context = provider.open_messages(request, input_tokens=input_tokens, **kwargs)
    return candidate_stream(
        context,
        writer=MessagesRecoveryWriter(
            model=kwargs.get("response_model") or request.model,
            input_tokens=input_tokens,
        ),
    )


def stream_responses(
    provider: Any,
    request: OpenAIResponsesRequest,
    input_tokens: int = 0,
    **kwargs: Any,
) -> AsyncIterator[str]:
    context = provider.open_responses(request, input_tokens=input_tokens, **kwargs)
    return candidate_stream(
        context,
        writer=ResponsesRecoveryWriter(
            model=kwargs.get("response_model") or request.model,
            input_tokens=input_tokens,
        ),
    )


def stream_native_messages(
    provider: Any,
    request: NativeMessagesRequest,
    **kwargs: Any,
) -> AsyncIterator[str]:
    model = kwargs.get("response_model") or request.model
    writer = (
        MessagesRecoveryWriter(model=model, input_tokens=0, native=True)
        if request.stream
        else NativeMessagesCompletionWriter(model=model)
    )
    return candidate_stream(
        provider.open_native_messages(request, **kwargs), writer=writer
    )


def candidate_stream(
    context: AbstractAsyncContextManager[ProviderCandidate],
    *,
    writer: RecoveryWriter,
) -> AsyncIterator[str]:
    async def open_candidate(
        index: int,
        target: ProviderModelTarget,
    ) -> AbstractAsyncContextManager[ProviderCandidate]:
        return context

    return RecoveryCoordinator(
        candidates=(ProviderModelTarget("test", "model", "test/model"),),
        opener=open_candidate,
        writer=writer,
        progress_timeout_seconds=30,
        timeout_failure=lambda _provider: ExecutionFailure(
            FailureKind.TIMEOUT, 504, "Test provider made no progress.", False
        ),
        request_id="test",
    ).stream()


class SDKStreamDouble[EventT](AsyncIterator[EventT]):
    """Model the SDK iterator and async close API at mocked create boundaries."""

    def __init__(
        self,
        source: AsyncIterable[EventT],
        *,
        close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._iterator = aiter(source)
        self._close = close

    def __aiter__(self) -> AsyncIterator[EventT]:
        return self

    async def __anext__(self) -> EventT:
        return await anext(self._iterator)

    async def close(self) -> None:
        try:
            if isinstance(self._iterator, AsyncCloseable):
                await self._iterator.aclose()
        finally:
            if self._close is not None:
                await self._close()


def make_provider_config(
    api_key: str | None,
    base_url: str,
    http_read_timeout: float = 120.0,
    http_write_timeout: float = 10.0,
    http_connect_timeout: float = 10.0,
    proxy: str | None = None,
    log_raw_sse_events: bool = False,
    log_api_error_tracebacks: bool = False,
) -> ProviderConfig:
    """Build a complete resolved config for isolated provider tests."""

    return ProviderConfig(
        api_key=api_key,
        base_url=base_url,
        http_read_timeout=http_read_timeout,
        http_write_timeout=http_write_timeout,
        http_connect_timeout=http_connect_timeout,
        proxy=proxy,
        log_raw_sse_events=log_raw_sse_events,
        log_api_error_tracebacks=log_api_error_tracebacks,
    )


async def capture_openai_chat_wire_body(body: dict) -> dict:
    """Return the JSON body serialized by the OpenAI chat client."""
    captured: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        assert isinstance(payload, dict)
        captured.append(payload)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: [DONE]\n\n",
        )

    client = AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    try:
        stream = await client.chat.completions.create(**body, stream=True)
        await stream.close()
    finally:
        await client.close()

    assert len(captured) == 1
    return captured[0]


def immediate_admission(
    *,
    provider_name: str = "TEST",
    max_attempts: int = 5,
) -> ProviderAdmissionController:
    """Return a real controller with deterministic zero-delay recovery."""
    return ProviderAdmissionController(
        provider_name=provider_name,
        rate_limit=1_000_000,
        rate_window=1.0,
        max_concurrency=1_000,
        max_attempts=max_attempts,
        base_delay=0.0,
        max_delay=0.0,
        jitter=0.0,
    )


def profiled_provider(
    provider_id: str,
    config: ProviderConfig,
    *,
    admission: ProviderAdmissionController | None = None,
) -> OpenAIChatProvider:
    """Construct one declarative provider for a focused behavior test."""
    return create_openai_chat_provider(
        provider_id,
        config,
        admission or immediate_admission(provider_name=provider_id),
    )


def reasoning_for(request: MessagesRequest) -> ReasoningPolicy:
    """Resolve provider-test input through the production client boundary."""

    return client_reasoning_policy(request)
