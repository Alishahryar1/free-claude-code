"""Provider test helpers with explicit admission ownership."""

import json
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import Any

import httpx2
from openai import AsyncOpenAI

from free_claude_code.application.reasoning import client_reasoning_policy
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.admission import (
    UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS,
    ProviderAdmissionController,
)
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
    max_attempts: int = UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS,
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


def without_empty_reasoning(value: Any) -> Any:
    """Compare unrelated wire contracts without negotiated empty placeholders.

    Populated reasoning and all other request fields remain observable. Projection
    tests assert the empty representations themselves.
    """
    empty_items = (
        {"type": "thinking", "thinking": ""},
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": ""}],
            "summary": [],
        },
    )
    if isinstance(value, list):
        return [
            without_empty_reasoning(item) for item in value if item not in empty_items
        ]
    if isinstance(value, dict):
        return {
            key: without_empty_reasoning(item)
            for key, item in value.items()
            if not (
                value.get("role") == "assistant"
                and key in {"reasoning_content", "reasoning"}
                and item == ""
            )
        }
    return value


def continuation_seed(text: str = "", thinking: str = ""):
    from free_claude_code.core.stream_recovery import ContinuationSeed

    items: list[JsonObject] = []
    if thinking:
        items.append({"type": "thinking", "thinking": thinking})
    if text:
        items.append({"type": "text", "text": text})
    return ContinuationSeed("messages", tuple(items))


def assistant_history_text(rows):
    """Read actual assistant answer text across both upstream request formats."""
    parts = []
    for row in rows:
        if row.get("role") != "assistant":
            continue
        content = row.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(
                part.get("text", "")
                for part in content
                if part.get("type") in {"text", "output_text"}
            )
    return "".join(parts)
