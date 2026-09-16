"""Tests for the Z.ai OpenAI-chat provider surfaces."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import (
    ZAI_API_DEFAULT_BASE,
    ZAI_CODING_DEFAULT_BASE,
)
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

_ZAI_BASES = {
    "zai": ZAI_CODING_DEFAULT_BASE,
    "zai_api": ZAI_API_DEFAULT_BASE,
}


@pytest.fixture(params=tuple(_ZAI_BASES))
def zai_provider(request: pytest.FixtureRequest):
    provider_id = request.param
    return profiled_provider(
        provider_id,
        make_provider_config(
            api_key="test_zai_key",
            base_url=_ZAI_BASES[provider_id],
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )


def test_init_uses_surface_specific_openai_chat_endpoint(zai_provider):
    assert isinstance(zai_provider, OpenAIChatProvider)
    assert zai_provider._api_key == "test_zai_key"
    expected_base = {
        "ZAI": ZAI_CODING_DEFAULT_BASE,
        "ZAI_API": ZAI_API_DEFAULT_BASE,
    }[zai_provider._provider_name]
    assert zai_provider._base_url == expected_base


def test_build_request_body_openai_chat(zai_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "glm-5.2",
            "max_tokens": 100,
            "messages": [Message(role="user", content="Hello")],
            "thinking": {"type": "enabled"},
        }
    )

    body = zai_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == "glm-5.2"
    assert body["max_tokens"] == 100
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert body["extra_body"]["thinking"] == {
        "type": "enabled",
        "clear_thinking": False,
    }


def test_build_request_body_default_max_tokens(zai_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )

    body = zai_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_rejects_caller_extra_body(zai_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"x": 1},
        }
    )

    with pytest.raises(InvalidRequestError, match=r"Z\.ai Chat Completions"):
        zai_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )


def test_build_request_body_disables_zai_thinking(zai_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "disabled"},
        }
    )

    body = zai_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["extra_body"]["thinking"] == {"type": "disabled"}


def test_build_request_body_replays_prior_reasoning_content(zai_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "glm-5.2",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "prior"}],
                },
                {"role": "user", "content": "continue"},
            ],
        }
    )

    body = zai_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["messages"][0]["reasoning_content"] == "prior"
    assert "extra_body" not in body


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(zai_provider):
    zai_provider._client = MagicMock()
    zai_provider._client.close = AsyncMock()

    await zai_provider.cleanup()

    zai_provider._client.close.assert_awaited_once()
