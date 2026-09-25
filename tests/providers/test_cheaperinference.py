"""Tests for the Cheaper Inference OpenAI-chat provider profile."""

from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import CHEAPERINFERENCE_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from free_claude_code.providers.openai_chat.profiles import OPENAI_CHAT_PROFILES
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_MODEL = "gpt-5.4-mini"


@pytest.fixture
def cheaperinference_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "cheaperinference",
        make_provider_config(
            api_key="test-cheaperinference-key",
            base_url=CHEAPERINFERENCE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="cheaperinference", max_attempts=1),
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
    cheaperinference_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(cheaperinference_provider, OpenAIChatProvider)
    assert cheaperinference_provider._provider_name == "CHEAPERINFERENCE"
    assert cheaperinference_provider._api_key == "test-cheaperinference-key"
    assert cheaperinference_provider._base_url == CHEAPERINFERENCE_DEFAULT_BASE


@pytest.mark.parametrize(
    "reasoning",
    [ReasoningPolicy.provider_default(), ReasoningPolicy.off()],
)
def test_default_and_off_reasoning_omit_effort_and_keep_standard_fields(
    cheaperinference_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
) -> None:
    body = cheaperinference_provider._chat._build_request_body(
        _request(), reasoning=reasoning
    )

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
    cheaperinference_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected: str,
) -> None:
    body = cheaperinference_provider._chat._build_request_body(
        _request(), reasoning=reasoning
    )

    assert body["reasoning_effort"] == expected


def test_replays_reasoning_content_with_tool_history(
    cheaperinference_provider: OpenAIChatProvider,
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

    body = cheaperinference_provider._chat._build_request_body(
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
async def test_model_catalog_keeps_text_models_with_token_limits(
    cheaperinference_provider: OpenAIChatProvider,
) -> None:
    cheaperinference_provider._client.get = AsyncMock(
        return_value={
            "data": [
                {
                    "id": _MODEL,
                    "type": "text",
                    "context_length": 400_000,
                    "max_output_tokens": 128_000,
                },
                {
                    "id": "claude-sonnet-5",
                    "type": "text",
                    "context_length": 1_000_000,
                    "max_output_tokens": 64_000,
                },
                {"id": "image-model", "type": "image"},
                {"id": "video-model", "type": "video"},
            ]
        }
    )

    assert await cheaperinference_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo(
                _MODEL,
                context_window_tokens=400_000,
                max_output_tokens=128_000,
            ),
            ProviderModelInfo(
                "claude-sonnet-5",
                context_window_tokens=1_000_000,
                max_output_tokens=64_000,
            ),
        }
    )


@pytest.mark.asyncio
async def test_model_catalog_rejects_missing_id(
    cheaperinference_provider: OpenAIChatProvider,
) -> None:
    cheaperinference_provider._client.get = AsyncMock(
        return_value={"data": [{"type": "text"}]}
    )

    with pytest.raises(ModelListResponseError, match="include id"):
        await cheaperinference_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_rejects_empty_catalog(
    cheaperinference_provider: OpenAIChatProvider,
) -> None:
    cheaperinference_provider._client.get = AsyncMock(return_value={"data": []})

    with pytest.raises(ModelListResponseError, match="did not include any model ids"):
        await cheaperinference_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_requests_streaming_text_models_with_bearer_auth() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        models = [{"id": _MODEL, "type": "text"}]
        if request.url.params.get("streaming") != "true":
            models.append({"id": "gpt-5.5-pro", "type": "text"})
        if request.url.params.get("type") != "text":
            models.append({"id": "image-model", "type": "image"})
        return httpx2.Response(200, json={"data": models})

    async with AsyncOpenAI(
        api_key="wire-cheaperinference-key",
        base_url=CHEAPERINFERENCE_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    ) as client:
        cheaperinference_provider = OpenAIChatProvider(
            make_provider_config(
                api_key="wire-cheaperinference-key",
                base_url=CHEAPERINFERENCE_DEFAULT_BASE,
            ),
            profile=OPENAI_CHAT_PROFILES["cheaperinference"],
            admission=immediate_admission(provider_name="cheaperinference"),
            client=client,
        )
        model_infos = await cheaperinference_provider.list_model_infos()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == (
        "https://api.cheaperinference.com/v1/models?type=text&streaming=true"
    )
    assert requests[0].headers["authorization"] == "Bearer wire-cheaperinference-key"
