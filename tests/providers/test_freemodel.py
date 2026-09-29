"""Tests for the FreeModel by Aiglade OpenAI-chat provider profile."""

from unittest.mock import AsyncMock

import pytest

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import FREEMODEL_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
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
)

_MODEL = "fm-v1-lite"
_NON_CHAT_MODALITIES = ("image", "video", "tts", "asr", "embedding", "rerank")


@pytest.fixture
def freemodel_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "freemodel",
        make_provider_config(
            api_key="test-freemodel-key",
            base_url=FREEMODEL_DEFAULT_BASE,
        ),
        admission=immediate_admission(provider_name="freemodel", max_attempts=1),
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


def _catalog_item(model_id: str, **extra: object) -> dict[str, object]:
    """Mirror one entry of https://freemodel.online/v1/models."""

    item: dict[str, object] = {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "aggregator",
        "modality": "chat",
    }
    item.update(extra)
    return item


def test_constructs_standard_openai_chat_provider(
    freemodel_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(freemodel_provider, OpenAIChatProvider)
    assert freemodel_provider._provider_name == "FREEMODEL"
    assert freemodel_provider._api_key == "test-freemodel-key"
    assert freemodel_provider._base_url == FREEMODEL_DEFAULT_BASE


@pytest.mark.parametrize(
    "reasoning",
    [REASONING_DEFAULT, REASONING_ON, REASONING_OFF],
)
def test_standard_request_fields_carry_no_reasoning_controls(
    freemodel_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
) -> None:
    body = freemodel_provider._chat._build_request_body(_request(), reasoning=reasoning)

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["model"] == _MODEL
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    assert "thinking" not in body


@pytest.mark.asyncio
async def test_filters_catalog_to_chat_modality(
    freemodel_provider: OpenAIChatProvider,
) -> None:
    freemodel_provider._client.get = AsyncMock(
        return_value={
            "data": [
                _catalog_item("fm-v1-lite", tier="lite"),
                _catalog_item("fm-v1-standard", tier="standard"),
                _catalog_item("qwen/qwen-plus"),
                *[
                    _catalog_item(f"auto/{modality}", modality=modality)
                    for modality in _NON_CHAT_MODALITIES
                ],
            ]
        }
    )

    model_infos = await freemodel_provider.list_model_infos()

    assert {info.model_id for info in model_infos} == {
        "fm-v1-lite",
        "fm-v1-standard",
        "qwen/qwen-plus",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("modality", None),
        ("modality", 1),
    ],
)
@pytest.mark.asyncio
async def test_rejects_missing_or_wrongly_typed_modality(
    freemodel_provider: OpenAIChatProvider,
    field: str,
    value: object,
) -> None:
    item = _catalog_item("fm-v1-lite")
    if value is None:
        item.pop(field)
    else:
        item[field] = value
    freemodel_provider._client.get = AsyncMock(return_value={"data": [item]})

    with pytest.raises(ModelListResponseError, match=f"include {field} as"):
        await freemodel_provider.list_model_infos()


@pytest.mark.asyncio
async def test_rejects_catalog_without_any_chat_model(
    freemodel_provider: OpenAIChatProvider,
) -> None:
    freemodel_provider._client.get = AsyncMock(
        return_value={"data": [_catalog_item("auto/image", modality="image")]}
    )

    with pytest.raises(ModelListResponseError, match="did not include any model ids"):
        await freemodel_provider.list_model_infos()
