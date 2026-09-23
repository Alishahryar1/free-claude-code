"""Provider test helpers with explicit admission ownership."""

import json
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable

import httpx2
from openai import AsyncOpenAI

from free_claude_code.application.reasoning import client_reasoning_policy
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    OpenAIChatProvider,
    create_openai_chat_provider,
)

REASONING_DEFAULT = ReasoningPolicy.provider_default()
REASONING_ON = ReasoningPolicy.on()
REASONING_OFF = ReasoningPolicy.off()


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
    rate_limit: int = 1_000_000,
    rate_window: int = 1,
    max_concurrency: int = 1_000,
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
        rate_limit=rate_limit,
        rate_window=rate_window,
        max_concurrency=max_concurrency,
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
