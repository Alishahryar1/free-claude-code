"""Tests for the Novita AI OpenAI-chat provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import NOVITA_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)


@pytest.fixture
def novita_provider():
    return profiled_provider(
        "novita",
        make_provider_config(
            api_key="test_novita_key",
            base_url=NOVITA_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )


def test_init_uses_openai_chat_provider(novita_provider):
    assert isinstance(novita_provider, OpenAIChatProvider)
    assert novita_provider._api_key == "test_novita_key"
    assert novita_provider._base_url == NOVITA_DEFAULT_BASE


def test_base_url_constant():
    assert NOVITA_DEFAULT_BASE == "https://api.novita.ai/openai/v1"


def test_build_request_body_openai_chat_shape(novita_provider):
    request = MessagesRequest(
        model="deepseek/deepseek-v4-flash-0731",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        system="System prompt",
    )

    body = novita_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == "deepseek/deepseek-v4-flash-0731"
    assert body["max_tokens"] == 100
    assert body["messages"] == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Hello"},
    ]


def test_build_request_body_default_max_tokens(novita_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )

    body = novita_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_preserves_validated_extra_body(novita_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"custom_param": "value"},
        }
    )

    body = novita_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["extra_body"]["custom_param"] == "value"


def test_build_request_body_rejects_reserved_reasoning_extra_body_keys(
    novita_provider,
):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"reasoning_effort": "high"},
        }
    )

    with pytest.raises(InvalidRequestError, match="extra_body must not override"):
        novita_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )


def test_build_request_body_disables_thinking_when_reasoning_off(novita_provider):
    request = MessagesRequest(
        model="zai-org/glm-5.2",
        messages=[Message(role="user", content="x")],
    )

    body = novita_provider._chat._build_request_body(request, reasoning=REASONING_OFF)

    assert body["extra_body"]["enable_thinking"] is False


def test_build_request_body_enables_thinking_when_reasoning_on(novita_provider):
    request = MessagesRequest(
        model="zai-org/glm-5.2",
        messages=[Message(role="user", content="x")],
    )

    body = novita_provider._chat._build_request_body(request, reasoning=REASONING_ON)

    assert body["extra_body"]["enable_thinking"] is True


def test_build_request_body_omits_thinking_field_by_default(novita_provider):
    request = MessagesRequest(
        model="zai-org/glm-5.2",
        messages=[Message(role="user", content="x")],
    )

    body = novita_provider._chat._build_request_body(
        request, reasoning=ReasoningPolicy.provider_default()
    )

    assert "enable_thinking" not in body.get("extra_body", {})


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(novita_provider):
    novita_provider._client = MagicMock()
    novita_provider._client.close = AsyncMock()

    await novita_provider.cleanup()

    novita_provider._client.close.assert_awaited_once()
