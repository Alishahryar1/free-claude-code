"""Tests for the QwenCloud Token Plan OpenAI-chat provider profile."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import QWENCLOUD_DEFAULT_BASE
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
def qwencloud_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "qwencloud",
        make_provider_config(
            api_key="test-qwencloud-key",
            base_url=QWENCLOUD_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="qwencloud"),
    )


def test_init_uses_openai_chat_provider(
    qwencloud_provider: OpenAIChatProvider,
) -> None:
    assert qwencloud_provider._api_key == "test-qwencloud-key"
    assert qwencloud_provider._base_url == QWENCLOUD_DEFAULT_BASE
    assert qwencloud_provider._provider_name == "QWENCLOUD"


def test_build_request_body_preserves_common_chat_tools_and_images(
    qwencloud_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "qwen3.7-plus",
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

    body = qwencloud_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["model"] == "qwen3.7-plus"
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
    qwencloud_provider: OpenAIChatProvider,
    reasoning,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "qwen3.7-plus",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    body = qwencloud_provider._chat._build_request_body(request, reasoning=reasoning)
    extra_body = body.get("extra_body", {})

    for field in (
        "enable_thinking",
        "preserve_thinking",
        "reasoning_effort",
        "thinking_budget",
    ):
        assert field not in body
        assert field not in extra_body


def test_build_request_body_replays_prior_reasoning_content(
    qwencloud_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "qwen3.7-plus",
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

    body = qwencloud_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": "Work through it.",
    }
    assert (
        qwencloud_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_model_catalog_uses_standard_endpoint_base_url_and_auth(
    qwencloud_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "qwen3.7-plus",
                        "object": "model",
                        "created": 0,
                        "owned_by": "qwencloud",
                    },
                    {
                        "id": "deepseek-v4-pro",
                        "object": "model",
                        "created": 0,
                        "owned_by": "qwencloud",
                    },
                ],
            },
        )

    await qwencloud_provider._client.close()
    qwencloud_provider._client = AsyncOpenAI(
        api_key="wire-qwencloud-key",
        base_url=QWENCLOUD_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await qwencloud_provider.list_model_infos()
    finally:
        await qwencloud_provider.cleanup()

    assert model_infos == frozenset(
        {
            ProviderModelInfo("qwen3.7-plus"),
            ProviderModelInfo("deepseek-v4-pro"),
        }
    )
    assert len(requests) == 1
    assert str(requests[0].url) == (
        "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/models"
    )
    assert requests[0].headers["authorization"] == "Bearer wire-qwencloud-key"


@pytest.mark.asyncio
async def test_empty_model_catalog_fails_instead_of_publishing_partial_state(
    qwencloud_provider: OpenAIChatProvider,
) -> None:
    qwencloud_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(data=[])
    )

    with pytest.raises(ModelListResponseError, match="did not include any model ids"):
        await qwencloud_provider.list_model_infos()
