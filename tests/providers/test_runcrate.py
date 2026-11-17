"""Tests for the Runcrate OpenAI-chat profile and semantic model catalog."""

from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import RUNCRATE_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
_TEXT = frozenset({ModelInputModality.TEXT})
_TEXT_AND_IMAGE = frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE})


@pytest.fixture
def runcrate_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "runcrate",
        make_provider_config(
            api_key="test-runcrate-key",
            base_url=RUNCRATE_DEFAULT_BASE,
        ),
        admission=immediate_admission(provider_name="runcrate", max_attempts=1),
    )


def _catalog_model(
    model_id: str = _MODEL,
    *,
    input_modalities: object = ("text",),
    output_modalities: object = ("text",),
    supported_features: object = ("tools", "json_mode", "reasoning"),
    context_length: int = 1048576,
    max_output_length: int = 131072,
) -> dict[str, object]:
    """Mirror the record shape served by ``GET https://api.runcrate.ai/v1/models``."""
    return {
        "id": model_id,
        "object": "model",
        "input_modalities": input_modalities,
        "output_modalities": output_modalities,
        "supported_features": supported_features,
        "context_length": context_length,
        "max_output_length": max_output_length,
    }


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


@pytest.mark.asyncio
async def test_lists_only_text_tool_capable_models_with_token_limits(
    runcrate_provider: OpenAIChatProvider,
) -> None:
    runcrate_provider._client.get = AsyncMock(
        return_value={
            "data": [
                _catalog_model(),
                _catalog_model(
                    "google/gemini-3.1-flash-lite",
                    input_modalities=("text", "image"),
                    supported_features=("tools", "json_mode", "structured_outputs"),
                    context_length=1000000,
                    max_output_length=65535,
                ),
                _catalog_model(
                    "Qwen/Qwen-Image",
                    output_modalities=("image",),
                    supported_features=(),
                    context_length=0,
                    max_output_length=0,
                ),
                _catalog_model(
                    "Qwen/Qwen3-TTS",
                    output_modalities=("audio",),
                    supported_features=("json_mode", "tools"),
                    context_length=0,
                    max_output_length=0,
                ),
                _catalog_model(
                    "BAAI/bge-m3",
                    output_modalities=("embeddings",),
                    supported_features=(),
                    context_length=8192,
                    max_output_length=8192,
                ),
                _catalog_model(
                    "openai/whisper-large-v3",
                    input_modalities=("audio",),
                    supported_features=("json_mode",),
                ),
                _catalog_model(
                    "meta-llama/Llama-Guard-4-12B",
                    input_modalities=("text", "image"),
                    supported_features=("logprobs",),
                ),
                {"id": "legacy-without-capability-metadata"},
            ]
        }
    )

    model_infos = await runcrate_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                supports_thinking=True,
                input_modalities=_TEXT,
                context_window_tokens=1048576,
                max_output_tokens=131072,
            ),
            ProviderModelInfo(
                "google/gemini-3.1-flash-lite",
                input_modalities=_TEXT_AND_IMAGE,
                context_window_tokens=1000000,
                max_output_tokens=65535,
            ),
        }
    )


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (_catalog_model(input_modalities="text"), "input_modalities string array"),
        (
            _catalog_model(supported_features=("tools", "")),
            "supported_features string array",
        ),
        ({"id": "legacy-without-capability-metadata"}, "did not include any model ids"),
        (
            _catalog_model("Qwen/Qwen-Image", output_modalities=("image",)),
            "did not include any model ids",
        ),
    ],
)
@pytest.mark.asyncio
async def test_rejects_malformed_or_unusable_catalog(
    runcrate_provider: OpenAIChatProvider,
    item: dict[str, object],
    message: str,
) -> None:
    runcrate_provider._client.get = AsyncMock(return_value={"data": [item]})

    with pytest.raises(ModelListResponseError, match=message):
        await runcrate_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_public_models_url_and_bearer_auth(
    runcrate_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [_catalog_model()]})

    await runcrate_provider._client.close()
    runcrate_provider._client = AsyncOpenAI(
        api_key="wire-runcrate-key",
        base_url=RUNCRATE_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await runcrate_provider.list_model_infos()
    finally:
        await runcrate_provider.cleanup()

    assert [info.model_id for info in model_infos] == [_MODEL]
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.runcrate.ai/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-runcrate-key"


def test_provider_default_reasoning_omits_effort(
    runcrate_provider: OpenAIChatProvider,
) -> None:
    body = runcrate_provider._chat._build_request_body(
        _request(), reasoning=ReasoningPolicy.provider_default()
    )

    assert body["model"] == _MODEL
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "reasoning_effort" not in body
    assert "reasoning" not in body
    assert "extra_body" not in body


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (ReasoningPolicy.off(), "none"),
        (ReasoningPolicy.on(), "high"),
        *(
            (ReasoningPolicy.on(effort=effort), effort.value)
            for effort in ReasoningEffort
        ),
    ],
)
def test_reasoning_policy_encodes_reasoning_effort(
    runcrate_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: str,
) -> None:
    body = runcrate_provider._chat._build_request_body(_request(), reasoning=reasoning)

    assert body["reasoning_effort"] == expected
    assert "reasoning" not in body
    assert "extra_body" not in body


def test_replays_reasoning_content_with_tool_history(
    runcrate_provider: OpenAIChatProvider,
) -> None:
    request = _request(
        messages=[
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
        ]
    )

    body = runcrate_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "I will inspect it.",
        "reasoning_content": "Read it first.",
        "tool_calls": [
            {
                "id": "toolu_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "example.py"}',
                },
            }
        ],
    }
    assert body["messages"][2] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "print('hello')",
    }
