"""Tests for the Nebius Token Factory OpenAI-chat profile and catalog."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import NEBIUS_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.model_capabilities import ModelInputModality
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

_MODEL = "Qwen/Qwen3-30B-A3B"


@pytest.fixture
def nebius_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "nebius",
        make_provider_config(
            api_key="test-nebius-key",
            base_url=NEBIUS_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="nebius"),
    )


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (REASONING_OFF, "none"),
        (REASONING_ON, "xhigh"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL), "minimal"),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "xhigh"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), "xhigh"),
    ],
)
def test_encodes_documented_reasoning_effort(
    nebius_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: str,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    body = nebius_provider._chat._build_request_body(request, reasoning=reasoning)

    assert body["reasoning_effort"] == expected
    assert "extra_body" not in body


def test_replays_reasoning_content(nebius_provider: OpenAIChatProvider) -> None:
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

    body = nebius_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": "Work through it.",
    }
    assert (
        nebius_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_lists_only_text_output_models(
    nebius_provider: OpenAIChatProvider,
) -> None:
    nebius_provider._client.get = AsyncMock(
        return_value={
            "data": [
                {
                    "id": _MODEL,
                    "architecture": {"modality": "text->text"},
                },
                {
                    "id": "black-forest-labs/flux",
                    "architecture": {"modality": "text->image"},
                },
            ]
        }
    )

    model_infos = await nebius_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            )
        }
    )
    nebius_provider._client.get.assert_awaited_once_with(
        "/models",
        cast_to=object,
        options={"params": {"verbose": "true"}},
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"data": [{"id": _MODEL}]}, "architecture.modality"),
        (
            {"data": [{"id": _MODEL, "architecture": {}}]},
            "architecture.modality",
        ),
        (
            {
                "data": [
                    {"id": _MODEL, "architecture": {"modality": True}},
                ]
            },
            "architecture.modality",
        ),
        (
            {
                "data": [
                    {"id": "", "architecture": {"modality": "text->text"}},
                ]
            },
            "include id",
        ),
        (
            {
                "data": [
                    {
                        "id": "image-model",
                        "architecture": {"modality": "text->image"},
                    },
                ]
            },
            "did not include any model ids",
        ),
    ],
)
@pytest.mark.asyncio
async def test_rejects_malformed_or_unusable_catalog_atomically(
    nebius_provider: OpenAIChatProvider,
    payload: object,
    message: str,
) -> None:
    nebius_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await nebius_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_auth_and_verbose_query(
    nebius_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "data": [
                    {
                        "id": _MODEL,
                        "architecture": {"modality": "text->text"},
                    }
                ]
            },
        )

    await nebius_provider._client.close()
    nebius_provider._client = AsyncOpenAI(
        api_key="wire-nebius-key",
        base_url=NEBIUS_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await nebius_provider.list_model_infos()
    finally:
        await nebius_provider.cleanup()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            )
        }
    )
    assert len(requests) == 1
    assert str(requests[0].url) == (
        "https://api.tokenfactory.nebius.com/v1/models?verbose=true"
    )
    assert requests[0].headers["authorization"] == "Bearer wire-nebius-key"
