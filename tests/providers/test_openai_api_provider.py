"""Direct OpenAI Platform API provider wiring across both FCC request protocols."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    parse_sse_text,
    text_content,
)
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.openai_api.provider import OpenAIAPIProvider
from tests.providers.support import immediate_admission, make_provider_config


def _sse(*events: tuple[str, dict[str, Any]]) -> str:
    return "".join(
        f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events
    )


def _complete_stream(text: str) -> str:
    return _sse(
        (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        ),
        (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": text},
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            },
        ),
    )


def _provider(
    handler: httpx2.MockTransport, *, max_attempts: int = 1
) -> OpenAIAPIProvider:
    return OpenAIAPIProvider(
        make_provider_config("platform-key", "https://api.openai.com/v1"),
        admission=immediate_admission(
            provider_name="openai_api", max_attempts=max_attempts
        ),
        transport=handler,
    )


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in stream])


@pytest.mark.asyncio
@pytest.mark.parametrize("ingress", ["messages", "responses"])
async def test_api_key_provider_uses_public_responses_endpoint(
    ingress: str,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            text=_complete_stream("hello"),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    provider = _provider(httpx2.MockTransport(handler))
    try:
        if ingress == "messages":
            output = await _collect(
                provider.stream_messages(
                    MessagesRequest.model_validate(
                        {
                            "model": "gpt-test",
                            "max_tokens": 64,
                            "messages": [{"role": "user", "content": "hello"}],
                        }
                    )
                )
            )
            assert text_content(parse_sse_text(output)) == "hello"
        else:
            output = await _collect(
                provider.stream_responses(
                    OpenAIResponsesRequest.model_validate(
                        {
                            "model": "gpt-test",
                            "input": [{"role": "user", "content": "hello"}],
                            "max_output_tokens": 64,
                            "metadata": {"source": "api-key-test"},
                        }
                    )
                )
            )
            assert "response.completed" in output
    finally:
        await provider.cleanup()

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == "Bearer platform-key"
    assert "originator" not in request.headers
    assert "chatgpt-account-id" not in request.headers
    body = json.loads(request.content)
    assert body["model"] == "gpt-test"
    assert body["store"] is False
    assert body["max_output_tokens"] == 64
    if ingress == "responses":
        assert body["metadata"] == {"source": "api-key-test"}


@pytest.mark.asyncio
async def test_api_key_model_discovery_keeps_every_returned_id_unknown() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "gpt-test", "object": "model", "owned_by": "openai"},
                    {
                        "id": "text-embedding-test",
                        "object": "model",
                        "owned_by": "openai",
                    },
                ],
            },
            request=request,
        )

    provider = _provider(httpx2.MockTransport(handler))
    try:
        models = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert models == frozenset(
        {ProviderModelInfo("gpt-test"), ProviderModelInfo("text-embedding-test")}
    )
    assert str(requests[0].url) == "https://api.openai.com/v1/models"
    assert requests[0].headers["authorization"] == "Bearer platform-key"


@pytest.mark.asyncio
async def test_api_key_provider_preserves_tool_calls() -> None:
    tool_name = "mcp__example__search"
    upstream_bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        upstream_bodies.append(body)
        return httpx2.Response(
            200,
            text=_sse(
                (
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": tool_name,
                        },
                    },
                ),
                (
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": tool_name,
                            "arguments": "{}",
                        },
                    },
                ),
                (
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": {"id": "resp_1"},
                    },
                ),
            ),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    provider = _provider(httpx2.MockTransport(handler))
    try:
        events = parse_sse_text(
            await _collect(
                provider.stream_messages(
                    MessagesRequest.model_validate(
                        {
                            "model": "gpt-test",
                            "messages": [{"role": "user", "content": "search"}],
                            "tools": [
                                {
                                    "name": tool_name,
                                    "input_schema": {"type": "object"},
                                }
                            ],
                        }
                    )
                )
            )
        )
    finally:
        await provider.cleanup()

    assert upstream_bodies[0]["tools"][0]["name"] == tool_name
    tool_start = next(
        event.data["content_block"]
        for event in events
        if event.event == "content_block_start"
    )
    assert tool_start["name"] == tool_name
