"""Tests for the SiliconFlow OpenAI-chat profile and model catalog."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import SILICONFLOW_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_MODEL = "Qwen/Qwen3-32B"
_CATALOG_URL = "https://api.siliconflow.com/v1/models?sub_type=chat"


def _request(**overrides: Any) -> MessagesRequest:
    payload: dict[str, Any] = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


@pytest.fixture
def siliconflow_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "siliconflow",
        make_provider_config(
            api_key="test-siliconflow-key",
            base_url=SILICONFLOW_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="siliconflow"),
    )


def test_init_uses_documented_endpoint(
    siliconflow_provider: OpenAIChatProvider,
) -> None:
    assert siliconflow_provider._api_key == "test-siliconflow-key"
    assert siliconflow_provider._base_url == SILICONFLOW_DEFAULT_BASE
    assert siliconflow_provider._provider_name == "SILICONFLOW"


def test_build_request_body_preserves_common_chat_tools_and_images(
    siliconflow_provider: OpenAIChatProvider,
) -> None:
    request = _request(
        system="Be concise.",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this image."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aW1hZ2U=",
                        },
                    },
                ],
            }
        ],
        tools=[
            {
                "name": "inspect_image",
                "description": "Inspect an image",
                "input_schema": {
                    "type": "object",
                    "properties": {"detail": {"type": "string"}},
                },
            }
        ],
    )

    body = siliconflow_provider._chat._build_request_body(
        request,
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert body["model"] == _MODEL
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["messages"][0] == {"role": "system", "content": "Be concise."}
    assert body["messages"][1]["content"] == [
        {"type": "text", "text": "Inspect this image."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
        },
    ]
    assert body["tools"][0]["function"]["name"] == "inspect_image"
    assert "extra_body" not in body


@pytest.mark.parametrize(
    "reasoning",
    [
        ReasoningPolicy.provider_default(),
        REASONING_OFF,
        REASONING_ON,
        ReasoningPolicy.on(budget_tokens=777),
        *[ReasoningPolicy.on(effort=effort) for effort in ReasoningEffort],
    ],
)
def test_build_request_body_omits_model_specific_thinking_controls(
    siliconflow_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
) -> None:
    body = siliconflow_provider._chat._build_request_body(
        _request(), reasoning=reasoning
    )

    assert "extra_body" not in body


def test_build_request_body_preserves_unrelated_extra_body(
    siliconflow_provider: OpenAIChatProvider,
) -> None:
    request = _request(extra_body={"min_p": 0.05})

    body = siliconflow_provider._chat._build_request_body(
        request,
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )

    assert body["extra_body"] == {
        "min_p": 0.05,
    }


@pytest.mark.parametrize("field", ["enable_thinking", "thinking_budget"])
def test_build_request_body_rejects_caller_thinking_override(
    siliconflow_provider: OpenAIChatProvider,
    field: str,
) -> None:
    request = _request(extra_body={field: "caller-owned"})

    with pytest.raises(InvalidRequestError, match="must not override reasoning"):
        siliconflow_provider._chat._build_request_body(request, reasoning=REASONING_ON)


def test_build_request_body_replays_reasoning_content_verbatim(
    siliconflow_provider: OpenAIChatProvider,
) -> None:
    request = _request(
        messages=[
            {"role": "user", "content": "Use a tool."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need the tool."},
                    {"type": "text", "text": "Calling it."},
                ],
            },
            {"role": "user", "content": "Continue."},
        ]
    )

    body = siliconflow_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "Calling it.",
        "reasoning_content": "Need the tool.",
    }
    assert (
        siliconflow_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_model_catalog_uses_chat_query(
    siliconflow_provider: OpenAIChatProvider,
) -> None:
    siliconflow_provider._client.get = AsyncMock(
        return_value={"data": [{"id": _MODEL}]}
    )

    model_infos = await siliconflow_provider.list_model_infos()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    siliconflow_provider._client.get.assert_awaited_once_with(
        "/models",
        cast_to=object,
        options={"params": {"sub_type": "chat"}},
    )


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_endpoint_query_and_auth() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": _MODEL}]})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def build_client(*args: Any, **kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = http_client
        return AsyncOpenAI(*args, **kwargs)

    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        side_effect=build_client,
    ):
        provider = profiled_provider(
            "siliconflow",
            make_provider_config(
                api_key="wire-siliconflow-key",
                base_url=SILICONFLOW_DEFAULT_BASE,
                rate_limit=10,
                rate_window=60,
            ),
            admission=immediate_admission(provider_name="siliconflow"),
        )
    try:
        model_infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == _CATALOG_URL
    assert requests[0].headers["authorization"] == "Bearer wire-siliconflow-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"models": []}, "expected top-level data array"),
        ({"data": [{}]}, "include id"),
        ({"data": []}, "did not include any model ids"),
    ],
)
async def test_rejects_malformed_or_empty_catalog_atomically(
    siliconflow_provider: OpenAIChatProvider,
    payload: object,
    message: str,
) -> None:
    siliconflow_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await siliconflow_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_shared_retry_admission(
    siliconflow_provider: OpenAIChatProvider,
) -> None:
    request = httpx.Request("GET", _CATALOG_URL)
    response = httpx.Response(429, request=request)
    rate_limit_error = httpx.HTTPStatusError(
        "rate limited",
        request=request,
        response=response,
    )
    siliconflow_provider._client.get = AsyncMock(
        side_effect=[rate_limit_error, {"data": [{"id": _MODEL}]}]
    )

    model_infos = await siliconflow_provider.list_model_infos()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert siliconflow_provider._client.get.await_count == 2
