"""API Route contracts at the real SDK/HTTP boundary, without external calls."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.config.provider_catalog import APIROUTE_DEFAULT_BASE
from free_claude_code.core.anthropic import aggregate_anthropic_sse_to_message
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)

_MODEL = "gpt-6.1-sol"


@asynccontextmanager
async def _provider(
    requests: list[httpx2.Request],
) -> AsyncIterator[OpenAIChatProvider]:
    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if request.method == "GET":
            assert str(request.url) == f"{APIROUTE_DEFAULT_BASE}/models"
            return httpx2.Response(
                200,
                json={"data": [{"id": _MODEL}, {"id": "vendor/another-model"}]},
            )
        assert request.method == "POST"
        assert str(request.url) == f"{APIROUTE_DEFAULT_BASE}/chat/completions"
        chunks = [
            {
                "id": "chat_api_route",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": _MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            for delta, finish in [
                ({"role": "assistant", "reasoning_content": "Plan the answer."}, None),
                ({"content": "API Route OK"}, None),
                ({}, "stop"),
            ]
        ]
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            + "data: [DONE]\n\n",
        )

    async with AsyncOpenAI(
        api_key="api-route-test-key",
        base_url=APIROUTE_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
    ) as client:
        with patch(
            "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
            return_value=client,
        ):
            provider = profiled_provider(
                "apiroute",
                make_provider_config(
                    api_key="api-route-test-key", base_url=APIROUTE_DEFAULT_BASE
                ),
                admission=immediate_admission(max_attempts=1),
            )
        try:
            yield provider
        finally:
            await provider.cleanup()


@pytest.mark.asyncio
async def test_apiroute_discovers_nonempty_model_ids_over_authenticated_http() -> None:
    requests: list[httpx2.Request] = []
    async with _provider(requests) as provider:
        infos = await provider.list_model_infos()

    assert {info.model_id for info in infos} == {_MODEL, "vendor/another-model"}
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer api-route-test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    ("reasoning", "effort"),
    [
        (ReasoningPolicy.provider_default(), None),
        (ReasoningPolicy.off(), None),
        (ReasoningPolicy.on(), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "high"),
    ],
)
async def test_apiroute_serializes_chat_and_reasoning_fields(
    wire: str, reasoning: ReasoningPolicy, effort: str | None
) -> None:
    requests: list[httpx2.Request] = []
    async with _provider(requests) as provider:
        stream = (
            provider.stream_messages(
                make_messages_request(_MODEL, thinking=None), reasoning=reasoning
            )
            if wire == "messages"
            else provider.stream_responses(
                OpenAIResponsesRequest(model=_MODEL, input="Hello"),
                reasoning=reasoning,
            )
        )
        result = "".join([event async for event in stream])

    assert "API Route OK" in result
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer api-route-test-key"
    body = json.loads(requests[0].content)
    assert body["model"] == _MODEL
    assert body["stream"] is True
    assert {"role": "user", "content": "Hello"} in body["messages"]
    if effort is None:
        assert "reasoning_effort" not in body
    else:
        assert body["reasoning_effort"] == effort
    assert "thinking" not in body
    assert "reasoning" not in body


@pytest.mark.asyncio
async def test_apiroute_preserves_streamed_thinking_as_readable_history() -> None:
    requests: list[httpx2.Request] = []
    async with _provider(requests) as provider:
        message, error, stopped = await aggregate_anthropic_sse_to_message(
            provider.stream_messages(
                make_messages_request(_MODEL), reasoning=ReasoningPolicy.on()
            )
        )
        assert error is None and stopped
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": message["content"]},
            {"role": "user", "content": "Continue"},
        ]
        result = "".join(
            [
                event
                async for event in provider.stream_messages(
                    make_messages_request(_MODEL, messages=history),
                    reasoning=ReasoningPolicy.on(),
                )
            ]
        )

    assert "API Route OK" in result
    assert len(requests) == 2
    body = json.loads(requests[1].content)
    assistant = next(msg for msg in body["messages"] if msg["role"] == "assistant")
    # This profile emits unsigned reasoning, so shared history validation keeps
    # it as readable context rather than replaying it as authenticated native data.
    assert (
        assistant["content"] == "API Route OK\n\n[Earlier reasoning]\nPlan the answer."
    )
    assert "reasoning_content" not in assistant
    assert "fcc:history" not in requests[1].content.decode()
