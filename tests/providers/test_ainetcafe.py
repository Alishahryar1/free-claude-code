"""Tests for the ainetcafe (Kimi K3) OpenAI-chat provider profile."""

from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import AINETCAFE_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.model_capabilities import ModelInputModality
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

_MODEL = "Kimi-K3"
_KIMI_K3_INFO = ProviderModelInfo(
    _MODEL,
    input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
)


@pytest.fixture
def ainetcafe_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "ainetcafe",
        make_provider_config(
            api_key="test-ainetcafe-key",
            base_url=AINETCAFE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="ainetcafe"),
    )


def test_constructs_standard_openai_chat_provider(
    ainetcafe_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(ainetcafe_provider, OpenAIChatProvider)
    assert ainetcafe_provider._provider_name == "AINETCAFE"
    assert ainetcafe_provider._api_key == "test-ainetcafe-key"
    assert ainetcafe_provider._base_url == AINETCAFE_DEFAULT_BASE


@pytest.mark.parametrize(
    ("reasoning", "expected_effort"),
    [
        (REASONING_DEFAULT, None),
        (REASONING_ON, "high"),
        (REASONING_OFF, "none"),
        (
            reasoning_for(
                MessagesRequest.model_validate(
                    {
                        "model": _MODEL,
                        "messages": [{"role": "user", "content": "x"}],
                        "thinking": {"type": "enabled", "budget_tokens": 1024},
                    }
                )
            ),
            "high",
        ),
    ],
)
def test_encodes_kimi_reasoning_effort(
    ainetcafe_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected_effort: str | None,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "Summarize the diff."}],
        }
    )

    body = ainetcafe_provider._chat._build_request_body(request, reasoning=reasoning)

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["model"] == _MODEL
    assert body.get("reasoning_effort") == expected_effort
    assert "reasoning" not in body
    assert "thinking" not in body
    assert "extra_body" not in body


def test_replays_thinking_in_reasoning_content_field(
    ainetcafe_provider: OpenAIChatProvider,
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

    body = ainetcafe_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": "Work through it.",
    }


@pytest.mark.asyncio
async def test_lists_models_from_models_endpoint(
    ainetcafe_provider: OpenAIChatProvider,
) -> None:
    ainetcafe_provider._client.get = AsyncMock(
        return_value={"data": [{"id": _MODEL}, {"id": "some-other-model"}]}
    )

    model_infos = await ainetcafe_provider.list_model_infos()

    assert model_infos == frozenset({_KIMI_K3_INFO})
    ainetcafe_provider._client.get.assert_awaited_once()
    call = ainetcafe_provider._client.get.await_args
    assert call is not None
    assert call.args == ("/models",)


@pytest.mark.asyncio
async def test_rejects_empty_model_ids(
    ainetcafe_provider: OpenAIChatProvider,
) -> None:
    ainetcafe_provider._client.get = AsyncMock(return_value={"data": [{"id": ""}]})

    with pytest.raises(ModelListResponseError, match="include id"):
        await ainetcafe_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    ainetcafe_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [{"id": _MODEL}]})

    await ainetcafe_provider._client.close()
    ainetcafe_provider._client = AsyncOpenAI(
        api_key="wire-ainetcafe-key",
        base_url=AINETCAFE_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await ainetcafe_provider.list_model_infos()
    finally:
        await ainetcafe_provider.cleanup()

    assert model_infos == frozenset({_KIMI_K3_INFO})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://microquickjs.com/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-ainetcafe-key"
