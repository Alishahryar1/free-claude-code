"""Tests for the ModelScope OpenAI-chat profile and model catalog."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import MODELSCOPE_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    capture_openai_chat_wire_body,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_MODEL = "Qwen/Qwen3.8-27B"
_CATALOG_URL = "https://api-inference.modelscope.cn/v1/models"


@pytest.fixture
def modelscope_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "modelscope",
        make_provider_config(
            api_key="test-modelscope-key",
            base_url=MODELSCOPE_DEFAULT_BASE,
        ),
        admission=immediate_admission(provider_name="modelscope"),
    )


def _request(**overrides: Any) -> MessagesRequest:
    payload: dict[str, Any] = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def test_init_uses_documented_endpoint(
    modelscope_provider: OpenAIChatProvider,
) -> None:
    assert modelscope_provider._api_key == "test-modelscope-key"
    assert modelscope_provider._base_url == MODELSCOPE_DEFAULT_BASE
    assert modelscope_provider._provider_name == "MODELSCOPE"


def test_build_request_body_preserves_common_chat_tools_and_images(
    modelscope_provider: OpenAIChatProvider,
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

    body = modelscope_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
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


@pytest.mark.parametrize(
    ("reasoning", "enabled"),
    [(REASONING_OFF, False), (REASONING_ON, True)],
)
def test_build_request_body_encodes_documented_thinking_control(
    modelscope_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    enabled: bool,
) -> None:
    request = _request()

    body = modelscope_provider._chat._build_request_body(request, reasoning=reasoning)

    assert body["extra_body"] == {"chat_template_kwargs": {"enable_thinking": enabled}}


def test_build_request_body_omits_thinking_control_for_provider_default(
    modelscope_provider: OpenAIChatProvider,
) -> None:
    request = _request()

    body = modelscope_provider._chat._build_request_body(
        request,
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert "extra_body" not in body


def test_build_request_body_replays_reasoning_content_verbatim(
    modelscope_provider: OpenAIChatProvider,
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

    body = modelscope_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "Calling it.",
        "reasoning_content": "Need the tool.",
    }
    assert (
        modelscope_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


def test_build_request_body_preserves_unrelated_extra_body(
    modelscope_provider: OpenAIChatProvider,
) -> None:
    request = _request(extra_body={"min_p": 0.05})

    body = modelscope_provider._chat._build_request_body(
        request,
        reasoning=REASONING_ON,
    )

    assert body["extra_body"]["min_p"] == 0.05
    assert body["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize("field", ["enable_thinking", "reasoning_effort"])
def test_build_request_body_rejects_caller_thinking_override(
    modelscope_provider: OpenAIChatProvider,
    field: str,
) -> None:
    request = _request(extra_body={field: "caller-owned"})

    with pytest.raises(InvalidRequestError, match="must not override reasoning"):
        modelscope_provider._chat._build_request_body(request, reasoning=REASONING_ON)


@pytest.mark.asyncio
async def test_thinking_control_reaches_wire_as_top_level_extension(
    modelscope_provider: OpenAIChatProvider,
) -> None:
    body = modelscope_provider._chat._build_request_body(
        _request(),
        reasoning=REASONING_OFF,
    )

    wire = await capture_openai_chat_wire_body(body)

    assert wire["chat_template_kwargs"] == {"enable_thinking": False}
    assert "extra_body" not in wire


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_endpoint_and_auth() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": _MODEL,
                        "object": "",
                        "created": 1787158178,
                        "owned_by": "system",
                    }
                ],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def build_client(*args: Any, **kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = http_client
        return AsyncOpenAI(*args, **kwargs)

    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        side_effect=build_client,
    ):
        provider = profiled_provider(
            "modelscope",
            make_provider_config(
                api_key="wire-modelscope-key",
                base_url=MODELSCOPE_DEFAULT_BASE,
            ),
            admission=immediate_admission(provider_name="modelscope"),
        )
    try:
        model_infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == _CATALOG_URL
    assert requests[0].headers["authorization"] == "Bearer wire-modelscope-key"


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
    modelscope_provider: OpenAIChatProvider,
    payload: object,
    message: str,
) -> None:
    modelscope_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await modelscope_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_shared_retry_admission(
    modelscope_provider: OpenAIChatProvider,
) -> None:
    request = httpx.Request("GET", _CATALOG_URL)
    response = httpx.Response(429, request=request)
    rate_limit_error = httpx.HTTPStatusError(
        "rate limited",
        request=request,
        response=response,
    )
    modelscope_provider._client.get = AsyncMock(
        side_effect=[rate_limit_error, {"data": [{"id": _MODEL}]}]
    )

    model_infos = await modelscope_provider.list_model_infos()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert modelscope_provider._client.get.await_count == 2
