"""Tests for the Chutes OpenAI-chat profile and semantic model catalog."""

from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import CHUTES_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningPolicy
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

_MODEL = "Qwen/Qwen3-32B-TEE"


@pytest.fixture
def chutes_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "chutes",
        make_provider_config(
            api_key="test-chutes-key",
            base_url=CHUTES_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="chutes"),
    )


def _catalog_model(
    model_id: str = _MODEL,
    *,
    input_modalities: object = ("text",),
    output_modalities: object = ("text",),
    supported_features: object = ("tools", "reasoning"),
) -> dict[str, object]:
    return {
        "id": model_id,
        "input_modalities": input_modalities,
        "output_modalities": output_modalities,
        "supported_features": supported_features,
    }


@pytest.mark.parametrize("reasoning", [REASONING_OFF, REASONING_ON])
def test_preserves_upstream_reasoning_default(
    chutes_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    body = chutes_provider._chat._build_request_body(request, reasoning=reasoning)

    assert "reasoning_effort" not in body
    assert "reasoning" not in body
    assert "extra_body" not in body


def test_does_not_replay_reasoning_but_preserves_tool_history(
    chutes_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [
                {"role": "user", "content": "Inspect the file."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Read it first."},
                        {"type": "text", "text": "I will inspect it."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "read_file",
                            "input": {"path": "example.py"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "print('hello')",
                        }
                    ],
                },
            ],
        }
    )

    body = chutes_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assistant = body["messages"][1]
    assert (
        assistant["content"]
        == "[Earlier reasoning]\nRead it first.\n\nI will inspect it."
    )
    assert assistant["tool_calls"][0]["id"] == "toolu_1"
    assert "reasoning_content" not in assistant
    assert "reasoning" not in assistant
    assert body["messages"][2] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "print('hello')",
    }


@pytest.mark.asyncio
async def test_lists_only_semantically_agent_capable_models(
    chutes_provider: OpenAIChatProvider,
) -> None:
    chutes_provider._client.get = AsyncMock(
        return_value={
            "data": [
                _catalog_model(input_modalities=("text", "image")),
                _catalog_model(
                    "plain-agent",
                    supported_features=("tools",),
                ),
                _catalog_model(
                    "image-generator",
                    output_modalities=("image",),
                ),
                _catalog_model(
                    "chat-without-tools",
                    supported_features=("reasoning",),
                ),
                {"id": "legacy-without-capability-metadata"},
            ]
        }
    )

    model_infos = await chutes_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
            ),
            ProviderModelInfo(
                "plain-agent",
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
        }
    )


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (_catalog_model(input_modalities="text"), "input_modalities string array"),
        (
            _catalog_model(output_modalities=("text", 7)),
            "output_modalities string array",
        ),
        (
            _catalog_model(supported_features=("tools", "")),
            "supported_features string array",
        ),
        (_catalog_model(""), "include id"),
        ({"id": "legacy-without-capability-metadata"}, "did not include any model ids"),
        (
            _catalog_model("image-only", output_modalities=("image",)),
            "did not include any model ids",
        ),
    ],
)
@pytest.mark.asyncio
async def test_rejects_malformed_or_unusable_catalog_atomically(
    chutes_provider: OpenAIChatProvider,
    item: dict[str, object],
    message: str,
) -> None:
    chutes_provider._client.get = AsyncMock(return_value={"data": [item]})

    with pytest.raises(ModelListResponseError, match=message):
        await chutes_provider.list_model_infos()


@pytest.mark.parametrize(
    "supported_features",
    [("tools", 7), "tools"],
)
@pytest.mark.asyncio
async def test_incomplete_record_cannot_bypass_later_metadata_validation(
    chutes_provider: OpenAIChatProvider,
    supported_features: object,
) -> None:
    chutes_provider._client.get = AsyncMock(
        return_value={
            "data": [
                _catalog_model(),
                {
                    "id": "incomplete-malformed-model",
                    "output_modalities": ["text"],
                    "supported_features": supported_features,
                },
            ]
        }
    )

    with pytest.raises(
        ModelListResponseError,
        match="supported_features string array",
    ):
        await chutes_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    chutes_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [_catalog_model()]})

    await chutes_provider._client.close()
    chutes_provider._client = AsyncOpenAI(
        api_key="wire-chutes-key",
        base_url=CHUTES_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await chutes_provider.list_model_infos()
    finally:
        await chutes_provider.cleanup()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                supports_thinking=True,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            )
        }
    )
    assert len(requests) == 1
    assert str(requests[0].url) == "https://llm.chutes.ai/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-chutes-key"
