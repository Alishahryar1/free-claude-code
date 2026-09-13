"""Tests for the Alibaba Model Studio OpenAI-chat provider profile."""

from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import ALIBABA_MODELSTUDIO_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.json_types import JsonObject, JsonValue
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

_MODEL = "qwen3-max"


@pytest.fixture
def modelstudio_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "alibaba_modelstudio",
        make_provider_config(
            api_key="test-modelstudio-key",
            base_url=ALIBABA_MODELSTUDIO_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(
            provider_name="alibaba_modelstudio", max_attempts=1
        ),
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
    modelstudio_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(modelstudio_provider, OpenAIChatProvider)
    assert modelstudio_provider._provider_name == "ALIBABA_MODELSTUDIO"
    assert modelstudio_provider._api_key == "test-modelstudio-key"
    assert modelstudio_provider._base_url == ALIBABA_MODELSTUDIO_DEFAULT_BASE


def test_base_url_constant() -> None:
    assert (
        ALIBABA_MODELSTUDIO_DEFAULT_BASE
        == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )


def test_build_request_body_openai_chat_shape(
    modelstudio_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest(
        model=_MODEL,
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        system="System prompt",
    )

    body = modelstudio_provider._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == _MODEL
    assert body["max_tokens"] == 100
    assert body["messages"] == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Hello"},
    ]


def test_build_request_body_default_max_tokens(
    modelstudio_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest(
        model=_MODEL,
        messages=[Message(role="user", content="x")],
    )

    body = modelstudio_provider._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_preserves_validated_extra_body(
    modelstudio_provider: OpenAIChatProvider,
) -> None:
    """DashScope accepts extra parameters (top_k, etc.) via extra_body."""

    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"top_k": 50},
        }
    )

    body = modelstudio_provider._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["extra_body"]["top_k"] == 50


def test_build_request_body_rejects_reserved_reasoning_extra_body_keys(
    modelstudio_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"enable_thinking": True},
        }
    )

    with pytest.raises(InvalidRequestError, match="extra_body must not override"):
        modelstudio_provider._build_request_body(
            request, reasoning=reasoning_for(request)
        )


def test_build_request_body_disables_thinking_when_reasoning_off(
    modelstudio_provider: OpenAIChatProvider,
) -> None:
    """Thinking models bill for reasoning, so OFF must disable it explicitly."""

    body = modelstudio_provider._build_request_body(_request(), reasoning=REASONING_OFF)

    assert body["extra_body"]["enable_thinking"] is False


def test_build_request_body_enables_thinking_when_reasoning_on(
    modelstudio_provider: OpenAIChatProvider,
) -> None:
    """Non-thinking Qwen models default to answering directly, so ON must opt in."""

    body = modelstudio_provider._build_request_body(_request(), reasoning=REASONING_ON)

    assert body["extra_body"]["enable_thinking"] is True


def test_build_request_body_omits_thinking_field_by_default(
    modelstudio_provider: OpenAIChatProvider,
) -> None:
    """Each Qwen model ships its own default, so FCC leaves it untouched."""

    body = modelstudio_provider._build_request_body(
        _request(), reasoning=REASONING_DEFAULT
    )

    assert "enable_thinking" not in body.get("extra_body", {})


def test_replays_reasoning_content_with_tool_history(
    modelstudio_provider: OpenAIChatProvider,
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

    body = modelstudio_provider._build_request_body(
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


def test_dashscope_model_ids_pass_through_untouched(
    modelstudio_provider: OpenAIChatProvider,
) -> None:
    """Model Studio ids are bare names (qwen3.8-max-0902), never aliases."""

    body = modelstudio_provider._build_request_body(
        _request(model="qwen3.8-max-0902"),
        reasoning=REASONING_OFF,
    )

    assert body["model"] == "qwen3.8-max-0902"
    assert body["extra_body"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_model_catalog_uses_default_listing_shape(
    modelstudio_provider: OpenAIChatProvider,
) -> None:
    """The default listing reads GET /models with collection 'data', id 'id'."""

    modelstudio_provider._client.models.list = AsyncMock(
        return_value={
            "data": [
                {
                    "id": "qwen3-max",
                    "object": "model",
                    "owned_by": "system",
                },
                {
                    "id": "qwen3.8-max-0902",
                    "object": "model",
                    "owned_by": "system",
                },
                {
                    "id": "deepseek-v4",
                    "object": "model",
                    "owned_by": "system",
                },
            ]
        }
    )

    model_infos = await modelstudio_provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo("qwen3-max"),
            ProviderModelInfo("qwen3.8-max-0902"),
            ProviderModelInfo("deepseek-v4"),
        }
    )
