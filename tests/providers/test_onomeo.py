"""Tests for the onomeo OpenAI-chat provider profile."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import ONOMEO_DEFAULT_BASE
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
    reasoning_for,
)

_MODEL = "deepseek-v4-flash"


@pytest.fixture
def onomeo_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "onomeo",
        make_provider_config(
            api_key="test-onomeo-key",
            base_url=ONOMEO_DEFAULT_BASE,
        ),
        admission=immediate_admission(provider_name="onomeo"),
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
    onomeo_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(onomeo_provider, OpenAIChatProvider)
    assert onomeo_provider._provider_name == "ONOMEO"
    assert onomeo_provider._api_key == "test-onomeo-key"
    assert onomeo_provider._base_url == "https://onomeo.com/v1"


@pytest.mark.parametrize(
    "reasoning",
    [REASONING_DEFAULT, REASONING_ON, REASONING_OFF],
)
def test_preserves_standard_fields_and_omits_reasoning_controls(
    onomeo_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
) -> None:
    body = onomeo_provider._chat._build_request_body(_request(), reasoning=reasoning)

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["model"] == _MODEL
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    assert "thinking" not in body
    assert "extra_body" not in body


def test_replays_earlier_reasoning_as_text_with_tool_history(
    onomeo_provider: OpenAIChatProvider,
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

    body = onomeo_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "[Earlier reasoning]\nRead it first.\n\nI will inspect it.",
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


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_reads_reasoning_deltas_from_either_field(
    onomeo_provider: OpenAIChatProvider,
    field: str,
) -> None:
    delta = SimpleNamespace(**{field: "next thought"})

    assert onomeo_provider._profile.reasoning_delta(delta) == "next thought"


@pytest.mark.asyncio
async def test_model_catalog_lists_ids_without_token_limits(
    onomeo_provider: OpenAIChatProvider,
) -> None:
    onomeo_provider._client.get = AsyncMock(
        return_value={
            "object": "list",
            "data": [
                {"id": "auto", "object": "model"},
                {"id": _MODEL, "object": "model"},
            ],
        }
    )

    model_infos = await onomeo_provider.list_model_infos()

    assert model_infos == frozenset(
        {ProviderModelInfo("auto"), ProviderModelInfo(_MODEL)}
    )
    onomeo_provider._client.get.assert_awaited_once()
    call = onomeo_provider._client.get.await_args
    assert call is not None
    assert call.args == ("/models",)


@pytest.mark.asyncio
async def test_model_catalog_rejects_missing_id(
    onomeo_provider: OpenAIChatProvider,
) -> None:
    onomeo_provider._client.get = AsyncMock(
        return_value={"object": "list", "data": [{"object": "model"}]}
    )

    with pytest.raises(ModelListResponseError, match="include id"):
        await onomeo_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_rejects_empty_catalog(
    onomeo_provider: OpenAIChatProvider,
) -> None:
    onomeo_provider._client.get = AsyncMock(return_value={"object": "list", "data": []})

    with pytest.raises(ModelListResponseError, match="did not include any model ids"):
        await onomeo_provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_catalog_uses_documented_url_and_bearer_auth(
    onomeo_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200, json={"object": "list", "data": [{"id": _MODEL, "object": "model"}]}
        )

    await onomeo_provider._client.close()
    onomeo_provider._client = AsyncOpenAI(
        api_key="wire-onomeo-key",
        base_url=ONOMEO_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await onomeo_provider.list_model_infos()
    finally:
        await onomeo_provider.cleanup()

    assert model_infos == frozenset({ProviderModelInfo(_MODEL)})
    assert len(requests) == 1
    assert str(requests[0].url) == "https://onomeo.com/v1/models"
    assert requests[0].headers["authorization"] == "Bearer wire-onomeo-key"
