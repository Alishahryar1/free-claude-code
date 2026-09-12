"""Tests for the Experiential Labs OpenAI-chat provider profile."""

from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import EXPERIENTIAL_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_MODEL = "gemma-4-26b-a4b-it-free"


@pytest.fixture
def experiential_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "experiential",
        make_provider_config(
            api_key="test-experiential-key",
            base_url=EXPERIENTIAL_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="experiential", max_attempts=1),
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
    experiential_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(experiential_provider, OpenAIChatProvider)
    assert experiential_provider._provider_name == "EXPERIENTIAL"
    assert experiential_provider._api_key == "test-experiential-key"
    assert experiential_provider._base_url == EXPERIENTIAL_DEFAULT_BASE


@pytest.mark.parametrize(
    "reasoning",
    [ReasoningPolicy.provider_default(), ReasoningPolicy.off()],
)
def test_default_and_off_reasoning_omit_effort_and_keep_standard_fields(
    experiential_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
) -> None:
    body = experiential_provider._build_request_body(_request(), reasoning=reasoning)

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["model"] == _MODEL
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    assert "thinking" not in body
    assert "extra_body" not in body


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    (
        (ReasoningPolicy.on(), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), "high"),
    ),
)
def test_reasoning_maps_efforts_onto_documented_vocabulary(
    experiential_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: str,
) -> None:
    body = experiential_provider._build_request_body(_request(), reasoning=reasoning)

    assert body["reasoning_effort"] == expected


def test_replays_reasoning_content_with_tool_history(
    experiential_provider: OpenAIChatProvider,
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

    body = experiential_provider._build_request_body(
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


@pytest.mark.asyncio
async def test_model_catalog_returns_bare_ids(
    experiential_provider: OpenAIChatProvider,
) -> None:
    experiential_provider._client.get = AsyncMock(
        return_value={"data": [{"id": "dots-3-note-preview-free"}, {"id": _MODEL}]}
    )

    assert await experiential_provider.list_model_infos() == frozenset(
        {ProviderModelInfo("dots-3-note-preview-free"), ProviderModelInfo(_MODEL)}
    )


@pytest.mark.asyncio
async def test_model_catalog_rejects_missing_id(
    experiential_provider: OpenAIChatProvider,
) -> None:
    experiential_provider._client.get = AsyncMock(return_value={"data": [{}]})

    with pytest.raises(ModelListResponseError, match="include id"):
        await experiential_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_rejects_empty_catalog(
    experiential_provider: OpenAIChatProvider,
) -> None:
    experiential_provider._client.get = AsyncMock(return_value={"data": []})

    with pytest.raises(ModelListResponseError, match="did not include any model ids"):
        await experiential_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    experiential_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [{"id": _MODEL}]})

    await experiential_provider._client.close()
    experiential_provider._client = AsyncOpenAI(
        api_key="wire-experiential-key",
        base_url=EXPERIENTIAL_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await experiential_provider.list_model_infos()
    finally:
        await experiential_provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.experientiallabs.ai/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-experiential-key"
