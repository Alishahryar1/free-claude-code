"""Tests for the Fireworks AI OpenAI-chat provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import FIREWORKS_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_OFF,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)


@pytest.fixture
def fireworks_provider():
    return profiled_provider(
        "fireworks",
        make_provider_config(
            api_key="test_fireworks_key",
            base_url=FIREWORKS_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )


def test_init_uses_openai_chat_provider(fireworks_provider):
    assert isinstance(fireworks_provider, OpenAIChatProvider)
    assert fireworks_provider._api_key == "test_fireworks_key"
    assert fireworks_provider._base_url == FIREWORKS_DEFAULT_BASE


def test_base_url_constant():
    assert FIREWORKS_DEFAULT_BASE == "https://api.fireworks.ai/inference/v1"


def test_build_request_body_openai_chat_shape(fireworks_provider):
    request = MessagesRequest(
        model="accounts/fireworks/models/glm-5p1",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        system="System prompt",
    )

    body = fireworks_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == "accounts/fireworks/models/glm-5p1"
    assert body["max_tokens"] == 100
    assert body["messages"] == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Hello"},
    ]


def test_build_request_body_default_max_tokens(fireworks_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )

    body = fireworks_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_replay_is_independent_of_current_turn_reasoning_control():
    provider = profiled_provider(
        "fireworks",
        make_provider_config(
            api_key="k",
            base_url=FIREWORKS_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "hidden"}],
                }
            ],
        }
    )

    body = provider._chat._build_request_body(request, reasoning=REASONING_OFF)

    assert body["messages"][0]["reasoning_content"] == "hidden"
    assert body["reasoning_effort"] == "none"


def test_build_request_body_preserves_validated_extra_body(fireworks_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"custom_param": "value"},
        }
    )

    body = fireworks_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["extra_body"] == {"custom_param": "value"}


def test_build_request_body_rejects_reserved_extra_body_keys(fireworks_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"temperature": 0.1},
        }
    )

    with pytest.raises(InvalidRequestError, match="extra_body must not override"):
        fireworks_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(fireworks_provider):
    fireworks_provider._client = MagicMock()
    fireworks_provider._client.close = AsyncMock()

    await fireworks_provider.cleanup()

    fireworks_provider._client.close.assert_awaited_once()
