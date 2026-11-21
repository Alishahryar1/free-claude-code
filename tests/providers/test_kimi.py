"""Tests for the Kimi OpenAI-chat provider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import KIMI_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)
from tests.wire_payload import object_payload


@pytest.fixture
def kimi_provider():
    return profiled_provider(
        "kimi",
        make_provider_config(
            api_key="test_kimi_key",
            base_url=KIMI_DEFAULT_BASE,
        ),
        admission=immediate_admission(),
    )


def test_init_uses_openai_chat_provider(kimi_provider: OpenAIChatProvider):
    assert isinstance(kimi_provider, OpenAIChatProvider)
    assert kimi_provider._api_key == "test_kimi_key"
    assert kimi_provider._base_url == "https://api.moonshot.ai/v1"


def test_build_request_body_openai_chat(kimi_provider: OpenAIChatProvider):
    request = MessagesRequest(
        model="kimi-k2.5",
        max_tokens=50,
        messages=[Message(role="user", content="hi")],
    )

    body = kimi_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == "kimi-k2.5"
    assert body["max_tokens"] == 50
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "extra_body" not in body


def test_build_request_body_default_max_tokens(kimi_provider: OpenAIChatProvider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )

    body = kimi_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_rejects_caller_extra_body(
    kimi_provider: OpenAIChatProvider,
):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"x": 1},
        }
    )

    with pytest.raises(InvalidRequestError, match="Kimi Chat Completions"):
        kimi_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )


def test_build_request_body_disables_kimi_thinking(kimi_provider: OpenAIChatProvider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "disabled"},
        }
    )

    body = kimi_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert object_payload(body["extra_body"])["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_model_list_uses_openai_client_models_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    kimi_provider: OpenAIChatProvider,
):
    mock_list = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(id="kimi-k2.5")])
    )
    monkeypatch.setattr(kimi_provider._client.models, "list", mock_list)

    assert await kimi_provider.list_model_infos() == frozenset(
        {ProviderModelInfo("kimi-k2.5")}
    )

    mock_list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(kimi_provider: OpenAIChatProvider):
    kimi_provider._client = MagicMock()
    kimi_provider._client.close = AsyncMock()

    await kimi_provider.cleanup()

    kimi_provider._client.close.assert_awaited_once()
