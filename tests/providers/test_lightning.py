"""Tests for the Lightning AI Model APIs OpenAI-chat provider profile."""

from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import LIGHTNING_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_DEFAULT,
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    profiled_provider,
)

_MODEL = "lightning-ai/gpt-oss-120b"


@pytest.fixture
def lightning_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "lightning",
        make_provider_config(
            api_key="test-lightning-key",
            base_url=LIGHTNING_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="lightning", max_attempts=1),
    )


def _request(**overrides: JsonValue) -> MessagesRequest:
    payload: JsonObject = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "Inspect the file."}],
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def _catalog_model(
    model_id: str = _MODEL,
    *,
    context_length: int | None = None,
    max_tokens: int | None = None,
    input_modalities: object = ("text",),
) -> dict[str, object]:
    model: dict[str, object] = {"id": model_id}
    if input_modalities is not None:
        model["architecture"] = {"input_modalities": input_modalities}
    if context_length is not None:
        model["context_length"] = context_length
    if max_tokens is not None:
        model["max_tokens"] = max_tokens
    return model


def test_constructs_standard_openai_chat_provider(
    lightning_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(lightning_provider, OpenAIChatProvider)
    assert lightning_provider._provider_name == "LIGHTNING"
    assert lightning_provider._api_key == "test-lightning-key"
    assert lightning_provider._base_url == LIGHTNING_DEFAULT_BASE


@pytest.mark.parametrize(
    ("reasoning", "expected_reasoning_effort"),
    [
        (REASONING_DEFAULT, None),
        (REASONING_ON, "medium"),
        (REASONING_OFF, "none"),
    ],
)
def test_encodes_documented_reasoning_effort_and_standard_request_fields(
    lightning_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected_reasoning_effort: str | None,
) -> None:
    body = lightning_provider._build_request_body(_request(), reasoning=reasoning)

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["model"] == _MODEL
    assert body["tools"][0]["function"]["name"] == "read_file"
    if expected_reasoning_effort is None:
        assert "reasoning_effort" not in body
    else:
        assert body["reasoning_effort"] == expected_reasoning_effort
    assert "reasoning" not in body
    assert "thinking" not in body
    assert "extra_body" not in body


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), "high"),
    ],
)
def test_reasoning_uses_lightning_documented_effort_vocabulary(
    lightning_provider: OpenAIChatProvider,
    policy: ReasoningPolicy,
    expected: str,
) -> None:
    body = lightning_provider._build_request_body(_request(), reasoning=policy)

    assert body["reasoning_effort"] == expected


@pytest.mark.asyncio
async def test_extracts_catalog_metadata(
    lightning_provider: OpenAIChatProvider,
) -> None:
    lightning_provider._client.get = AsyncMock(
        return_value={
            "data": [
                _catalog_model(
                    "openai/gpt-5-mini",
                    context_length=400_000,
                    max_tokens=128_000,
                    input_modalities=("text", "image"),
                ),
                _catalog_model("bare-model", input_modalities=None),
            ]
        }
    )

    model_infos = await lightning_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                "openai/gpt-5-mini",
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=400_000,
                max_output_tokens=128_000,
            ),
            ProviderModelInfo("bare-model"),
        }
    )


@pytest.mark.asyncio
async def test_rejects_catalog_item_without_model_id(
    lightning_provider: OpenAIChatProvider,
) -> None:
    item = _catalog_model()
    item.pop("id")
    lightning_provider._client.get = AsyncMock(return_value={"data": [item]})

    with pytest.raises(ModelListResponseError, match="include id"):
        await lightning_provider.list_model_infos()


@pytest.mark.asyncio
async def test_empty_catalog_rejected(lightning_provider: OpenAIChatProvider) -> None:
    lightning_provider._client.get = AsyncMock(return_value={"data": []})

    with pytest.raises(ModelListResponseError, match="did not include any model ids"):
        await lightning_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    lightning_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [_catalog_model()]})

    await lightning_provider._client.close()
    lightning_provider._client = AsyncOpenAI(
        api_key="wire-lightning-key",
        base_url=LIGHTNING_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await lightning_provider.list_model_infos()
    finally:
        await lightning_provider.cleanup()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                input_modalities=frozenset({ModelInputModality.TEXT}),
            )
        }
    )
    assert len(requests) == 1
    assert str(requests[0].url) == "https://lightning.ai/api/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-lightning-key"
