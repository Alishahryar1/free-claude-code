"""Tests for the xAI OpenAI-chat provider profile and model catalog."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import XAI_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.model_capabilities import ModelInputModality
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
def xai_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "xai",
        make_provider_config(
            api_key="test-xai-key",
            base_url=XAI_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="xai"),
    )


def test_init_uses_openai_chat_provider(xai_provider: OpenAIChatProvider) -> None:
    assert xai_provider._api_key == "test-xai-key"
    assert xai_provider._base_url == "https://api.x.ai/v1"
    assert xai_provider._provider_name == "XAI"


def test_build_request_body_preserves_common_chat_tools_and_images(
    xai_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "grok-4.5",
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

    body = xai_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["model"] == "grok-4.5"
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["messages"][0] == {"role": "system", "content": "Be concise."}
    user_content = body["messages"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "Inspect this image."}
    assert user_content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
    }
    assert body["tools"][0]["function"]["name"] == "inspect_image"


@pytest.mark.parametrize("reasoning", [REASONING_OFF, REASONING_ON])
def test_build_request_body_does_not_invent_catalog_wide_reasoning_control(
    xai_provider: OpenAIChatProvider,
    reasoning,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "grok-4.5",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    body = xai_provider._chat._build_request_body(request, reasoning=reasoning)

    assert "reasoning_effort" not in body
    assert "reasoning" not in body.get("extra_body", {})


def test_build_request_body_replays_prior_reasoning_content(
    xai_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "grok-4.5",
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

    body = xai_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": "Work through it.",
    }
    assert (
        xai_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_lists_language_models_and_routable_aliases(
    xai_provider: OpenAIChatProvider,
) -> None:
    xai_provider._client.get = AsyncMock(
        return_value={
            "models": [
                {
                    "id": "latest",
                    "aliases": ["grok-4.5", "grok-latest"],
                    "input_modalities": ["text", "image"],
                },
                {
                    "id": "grok-4.5",
                    "aliases": [],
                    "input_modalities": ["text"],
                },
            ]
        }
    )
    xai_provider._client.models.list = AsyncMock()

    model_infos = await xai_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                "latest",
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
            ),
            ProviderModelInfo(
                "grok-4.5",
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
            ),
            ProviderModelInfo(
                "grok-latest",
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
            ),
        }
    )
    xai_provider._client.get.assert_awaited_once_with(
        "/language-models",
        cast_to=object,
    )
    xai_provider._client.models.list.assert_not_awaited()


@pytest.mark.asyncio
async def test_language_model_catalog_uses_configured_base_url_and_auth(
    xai_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={"models": [{"id": "grok-4.5", "aliases": []}]},
        )

    await xai_provider._client.close()
    xai_provider._client = AsyncOpenAI(
        api_key="wire-xai-key",
        base_url=XAI_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await xai_provider.list_model_infos()
    finally:
        await xai_provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo("grok-4.5")})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.x.ai/v1/language-models"
    assert requests[0].headers["authorization"] == "Bearer wire-xai-key"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"data": []}, "top-level models array"),
        ({"models": [{}]}, "models item to include id"),
        ({"models": [{"id": "grok"}]}, "include aliases array"),
        (
            {"models": [{"id": "grok", "aliases": [""]}]},
            "aliases item to be a model id",
        ),
        ({"models": []}, "did not include any model ids"),
    ],
)
@pytest.mark.asyncio
async def test_rejects_malformed_language_model_catalog_atomically(
    xai_provider: OpenAIChatProvider,
    payload: dict,
    message: str,
) -> None:
    xai_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await xai_provider.list_model_infos()


@pytest.mark.asyncio
async def test_language_model_catalog_uses_shared_retry_admission(
    xai_provider: OpenAIChatProvider,
) -> None:
    request = httpx.Request("GET", "https://api.x.ai/v1/language-models")
    response = httpx.Response(429, request=request)
    rate_limit_error = httpx.HTTPStatusError(
        "rate limited",
        request=request,
        response=response,
    )
    xai_provider._client.get = AsyncMock(
        side_effect=[
            rate_limit_error,
            {"models": [{"id": "grok-4.5", "aliases": []}]},
        ]
    )

    model_infos = await xai_provider.list_model_infos()

    assert model_infos == frozenset({ProviderModelInfo("grok-4.5")})
    assert xai_provider._client.get.await_count == 2
