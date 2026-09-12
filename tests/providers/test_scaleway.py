"""Tests for the Scaleway Generative APIs OpenAI-chat provider profile."""

from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import SCALEWAY_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_DEFAULT,
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_MODEL = "deepseek/deepseek-v4-flash"


@pytest.fixture
def scaleway_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "scaleway",
        make_provider_config(
            api_key="test-scaleway-key",
            base_url=SCALEWAY_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="scaleway"),
    )


def test_constructs_standard_openai_chat_provider(
    scaleway_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(scaleway_provider, OpenAIChatProvider)
    assert scaleway_provider._provider_name == "SCALEWAY"
    assert scaleway_provider._api_key == "test-scaleway-key"
    assert scaleway_provider._base_url == SCALEWAY_DEFAULT_BASE


@pytest.mark.parametrize(
    "reasoning",
    [REASONING_DEFAULT, REASONING_ON, REASONING_OFF],
)
def test_preserves_standard_fields_and_omits_reasoning_controls(
    scaleway_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "Summarize the diff."}],
        }
    )

    body = scaleway_provider._build_request_body(request, reasoning=reasoning)

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["model"] == _MODEL
    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    assert "thinking" not in body
    assert "extra_body" not in body


def test_drops_thinking_replay_for_plain_chat_history(
    scaleway_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
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

    body = scaleway_provider._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "[Earlier reasoning]\nWork through it.\n\nThe answer is 42.",
    }


@pytest.mark.asyncio
async def test_lists_models_from_models_endpoint(
    scaleway_provider: OpenAIChatProvider,
) -> None:
    scaleway_provider._client.get = AsyncMock(return_value={"data": [{"id": _MODEL}]})

    model_infos = await scaleway_provider.list_model_infos()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    scaleway_provider._client.get.assert_awaited_once()
    call = scaleway_provider._client.get.await_args
    assert call is not None
    assert call.args == ("/models",)


@pytest.mark.asyncio
async def test_rejects_empty_model_ids(
    scaleway_provider: OpenAIChatProvider,
) -> None:
    scaleway_provider._client.get = AsyncMock(return_value={"data": [{"id": ""}]})

    with pytest.raises(ModelListResponseError, match="include id"):
        await scaleway_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    scaleway_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [{"id": _MODEL}]})

    await scaleway_provider._client.close()
    scaleway_provider._client = AsyncOpenAI(
        api_key="wire-scaleway-key",
        base_url=SCALEWAY_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await scaleway_provider.list_model_infos()
    finally:
        await scaleway_provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.scaleway.ai/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-scaleway-key"
