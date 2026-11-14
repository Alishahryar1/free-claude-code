"""Tests for the LLMTR OpenAI-chat gateway profile and catalog."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import LLMTR_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
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

_MODEL = "llmtr/qwen3-8-27b"
_FREE_MODEL = "qwen/qwen3.8-27b-free"


@pytest.fixture
def llmtr_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "llmtr",
        make_provider_config(
            api_key="test-llmtr-key",
            base_url=LLMTR_DEFAULT_BASE,
        ),
        admission=immediate_admission(provider_name="llmtr", max_attempts=1),
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


def test_constructs_standard_openai_chat_provider(
    llmtr_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(llmtr_provider, OpenAIChatProvider)
    assert llmtr_provider._provider_name == "LLMTR"
    assert llmtr_provider._api_key == "test-llmtr-key"
    assert llmtr_provider._base_url == LLMTR_DEFAULT_BASE


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (REASONING_DEFAULT, None),
        (REASONING_OFF, False),
        (REASONING_ON, True),
    ],
)
def test_encodes_reasoning_as_a_boolean(
    llmtr_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: bool | None,
) -> None:
    body = llmtr_provider._chat._build_request_body(
        _request(),
        reasoning=reasoning,
    )

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["model"] == _MODEL
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "reasoning_effort" not in body
    assert "thinking" not in body
    if expected is None:
        assert "reasoning" not in body.get("extra_body", {})
        return
    assert body["extra_body"]["reasoning"] is expected


def test_rejects_reasoning_object_from_caller_extra_body(
    llmtr_provider: OpenAIChatProvider,
) -> None:
    request = _request(extra_body={"reasoning": {"effort": "high"}})

    with pytest.raises(InvalidRequestError, match="extra_body must not override"):
        llmtr_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )


def test_forwards_free_model_id_without_adding_another_prefix(
    llmtr_provider: OpenAIChatProvider,
) -> None:
    body = llmtr_provider._chat._build_request_body(
        _request(model=_FREE_MODEL),
        reasoning=REASONING_DEFAULT,
    )

    assert body["model"] == _FREE_MODEL


def test_replays_reasoning_content_with_tool_history(
    llmtr_provider: OpenAIChatProvider,
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

    body = llmtr_provider._chat._build_request_body(
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
    assert (
        llmtr_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_model_catalog_keeps_chat_models_and_prefixed_ids(
    llmtr_provider: OpenAIChatProvider,
) -> None:
    llmtr_provider._client.get = AsyncMock(
        return_value={
            "data": [
                {
                    "id": _FREE_MODEL,
                    "supported_operations": ["CHAT_COMPLETIONS"],
                    "context_length": 131042,
                    "architecture": {"input_modalities": ["text", "image"]},
                    "top_provider": {"max_completion_tokens": 16384},
                },
                {
                    "id": _MODEL,
                    "supported_operations": ["CHAT_COMPLETIONS"],
                    "context_length": 262144,
                    "architecture": {"input_modalities": ["text"]},
                    "top_provider": {"max_completion_tokens": 262144},
                    "reasoning": {
                        "mandatory": False,
                        "default_enabled": False,
                        "supported_efforts": [],
                    },
                },
                {
                    "id": "openai/gpt-5.6-sol",
                    "supported_operations": ["RESPONSES"],
                    "context_length": 128000,
                    "architecture": {"input_modalities": ["text", "image", "video"]},
                    "top_provider": {"max_completion_tokens": 16384},
                },
                {
                    "id": "llmtr/embeddingmagibu-200m",
                    "supported_operations": ["EMBEDDINGS"],
                    "context_length": 8192,
                    "architecture": {"input_modalities": ["text"]},
                    "top_provider": {"max_completion_tokens": 8192},
                },
            ]
        }
    )

    model_infos = await llmtr_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo(
                _FREE_MODEL,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=131042,
                max_output_tokens=16384,
            ),
            ProviderModelInfo(
                _MODEL,
                input_modalities=frozenset({ModelInputModality.TEXT}),
                context_window_tokens=262144,
                max_output_tokens=262144,
            ),
        }
    )
    llmtr_provider._client.get.assert_awaited_once_with(
        "/models",
        cast_to=object,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"models": []}, "expected top-level data array"),
        ({"data": [{}]}, "include id"),
        ({"data": []}, "did not include any model ids"),
        (
            {"data": [{"id": _FREE_MODEL}]},
            "supported_operations string array",
        ),
        (
            {
                "data": [
                    {
                        "id": "llmtr/embeddingmagibu-200m",
                        "supported_operations": ["EMBEDDINGS"],
                    }
                ]
            },
            "did not include any model ids",
        ),
    ],
)
async def test_rejects_malformed_or_empty_catalog_atomically(
    llmtr_provider: OpenAIChatProvider,
    payload: object,
    message: str,
) -> None:
    llmtr_provider._client.get = AsyncMock(return_value=payload)

    with pytest.raises(ModelListResponseError, match=message):
        await llmtr_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    llmtr_provider: OpenAIChatProvider,
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
                        "supported_operations": ["CHAT_COMPLETIONS"],
                    }
                ]
            },
        )

    await llmtr_provider._client.close()
    llmtr_provider._client = AsyncOpenAI(
        api_key="wire-llmtr-key",
        base_url=LLMTR_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await llmtr_provider.list_model_infos()
    finally:
        await llmtr_provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://llmtr.com/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-llmtr-key"
