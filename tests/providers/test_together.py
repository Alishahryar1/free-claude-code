"""Tests for the Together AI OpenAI-chat profile and model catalog."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import TOGETHER_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
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


@pytest.fixture
def together_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "together",
        make_provider_config(
            api_key="test-together-key",
            base_url=TOGETHER_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="together"),
    )


def test_init_uses_openai_chat_provider(
    together_provider: OpenAIChatProvider,
) -> None:
    assert together_provider._api_key == "test-together-key"
    assert together_provider._base_url == TOGETHER_DEFAULT_BASE
    assert together_provider._provider_name == "TOGETHER"


def test_build_request_body_preserves_common_chat_tools_and_images(
    together_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "zai-org/GLM-5.2",
            "system": "Be concise.",
            "messages": [
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
            "tools": [
                {
                    "name": "inspect_image",
                    "description": "Inspect an image",
                    "input_schema": {
                        "type": "object",
                        "properties": {"detail": {"type": "string"}},
                    },
                }
            ],
        }
    )

    body = together_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["model"] == "zai-org/GLM-5.2"
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


@pytest.mark.parametrize("reasoning", [REASONING_OFF, REASONING_ON])
def test_build_request_body_does_not_invent_catalog_wide_reasoning_control(
    together_provider: OpenAIChatProvider,
    reasoning,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "zai-org/GLM-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    body = together_provider._chat._build_request_body(request, reasoning=reasoning)
    extra_body = body.get("extra_body", {})

    for field in ("reasoning", "reasoning_effort", "chat_template_kwargs"):
        assert field not in body
        assert field not in extra_body


def test_build_request_body_replays_documented_reasoning_field(
    together_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "zai-org/GLM-5.2",
            "messages": [
                {"role": "user", "content": "Solve it."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Work through it."},
                        {"type": "text", "text": "The answer is 42."},
                    ],
                },
                {"role": "user", "content": "Continue."},
            ],
        }
    )

    body = together_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning": "Work through it.",
    }
    assert (
        together_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_lists_only_documented_chat_models(
    together_provider: OpenAIChatProvider,
) -> None:
    together_provider._client.get = AsyncMock(
        return_value=[
            {"id": "zai-org/GLM-5.2", "type": "chat"},
            {"id": "openai/gpt-oss-120b", "type": "chat"},
            {"id": "codellama/CodeLlama-70b", "type": "code"},
            {"id": "black-forest-labs/FLUX.2-dev", "type": "image"},
            {"id": "intfloat/multilingual-e5-large", "type": "embedding"},
            {"id": "meta-llama/Llama-Guard-4-12B", "type": "moderation"},
            {"id": "mixedbread-ai/Mxbai-Rerank-Large-V2", "type": "rerank"},
        ]
    )
    together_provider._client.models.list = AsyncMock()

    model_infos = await together_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo("zai-org/GLM-5.2"),
            ProviderModelInfo("openai/gpt-oss-120b"),
        }
    )
    together_provider._client.get.assert_awaited_once_with(
        "/models",
        cast_to=object,
    )
    together_provider._client.models.list.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_catalog_uses_configured_base_url_and_auth(
    together_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json=[
                {
                    "id": "zai-org/GLM-5.2",
                    "object": "model",
                    "created": 0,
                    "type": "chat",
                }
            ],
        )

    await together_provider._client.close()
    together_provider._client = AsyncOpenAI(
        api_key="wire-together-key",
        base_url=TOGETHER_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await together_provider.list_model_infos()
    finally:
        await together_provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo("zai-org/GLM-5.2")})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.together.ai/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-together-key"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"data": []}, "expected root array"),
        ([{"type": "chat"}], "include id"),
        ([{"id": "model"}], "include type as str"),
        ([{"id": "model", "type": 7}], "include type as str"),
        ([], "did not include any model ids"),
        (
            [{"id": "black-forest-labs/FLUX.2-dev", "type": "image"}],
            "did not include any model ids",
        ),
    ],
)
@pytest.mark.asyncio
async def test_rejects_malformed_or_unusable_catalog_atomically(
    together_provider: OpenAIChatProvider,
    payload: object,
    message: str,
) -> None:
    together_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await together_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_shared_retry_admission(
    together_provider: OpenAIChatProvider,
) -> None:
    request = httpx.Request("GET", "https://api.together.ai/v1/models")
    response = httpx.Response(429, request=request)
    rate_limit_error = httpx.HTTPStatusError(
        "rate limited",
        request=request,
        response=response,
    )
    together_provider._client.get = AsyncMock(
        side_effect=[
            rate_limit_error,
            [{"id": "zai-org/GLM-5.2", "type": "chat"}],
        ]
    )

    model_infos = await together_provider.list_model_infos()

    assert model_infos == frozenset({ProviderModelInfo("zai-org/GLM-5.2")})
    assert together_provider._client.get.await_count == 2
