"""Tests for the Kimi Code subscription provider profile."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import KIMI_CODE_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)


def _request(**overrides) -> MessagesRequest:
    payload = {
        "model": "k3",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


@pytest.fixture
def kimi_code_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "kimi_code",
        make_provider_config(
            api_key="test-subscription-key",
            base_url=KIMI_CODE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )


def test_init_uses_subscription_endpoint_and_identifies_fcc(kimi_code_provider):
    assert kimi_code_provider._api_key == "test-subscription-key"
    assert kimi_code_provider._base_url == "https://api.kimi.com/coding/v1"
    assert kimi_code_provider._provider_name == "KIMI_CODE"
    assert kimi_code_provider._client.default_headers["User-Agent"] == (
        "free-claude-code"
    )


def test_explicit_output_limit_uses_max_completion_tokens(kimi_code_provider):
    request = _request(max_tokens=32_768)

    body = kimi_code_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["max_completion_tokens"] == 32_768
    assert "max_tokens" not in body


def test_omitted_output_limit_preserves_kimi_default(kimi_code_provider):
    request = _request()

    body = kimi_code_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body


@pytest.mark.parametrize(
    ("client_effort", "upstream_effort"),
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "max"),
        ("max", "max"),
    ],
)
def test_named_reasoning_effort_uses_kimi_vocabulary(
    kimi_code_provider,
    client_effort: str,
    upstream_effort: str,
) -> None:
    request = _request(output_config={"effort": client_effort})

    body = kimi_code_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["reasoning_effort"] == upstream_effort


def test_enabled_reasoning_without_named_effort_uses_max(kimi_code_provider):
    request = _request(thinking={"type": "enabled", "budget_tokens": 4_096})

    body = kimi_code_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["reasoning_effort"] == "max"


def test_disabled_reasoning_uses_none(kimi_code_provider):
    request = _request(thinking={"type": "disabled"})

    body = kimi_code_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["reasoning_effort"] == "none"


def test_reasoning_history_uses_reasoning_content(kimi_code_provider):
    request = _request(
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Inspect the repository."},
                    {"type": "text", "text": "I found the relevant file."},
                ],
            },
            {"role": "user", "content": "Continue."},
        ]
    )

    body = kimi_code_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )
    assistant = body["messages"][0]

    assert assistant["reasoning_content"] == "Inspect the repository."
    assert assistant["content"] == "I found the relevant file."


def test_caller_extra_body_is_rejected(kimi_code_provider):
    request = _request(extra_body={"thinking": {"type": "enabled"}})

    with pytest.raises(InvalidRequestError, match="Kimi Code Chat Completions"):
        kimi_code_provider._chat._build_request_body(
            request, reasoning=reasoning_for(request)
        )


@pytest.mark.asyncio
async def test_model_list_uses_subscription_models_endpoint(kimi_code_provider):
    kimi_code_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(id="k3"),
                SimpleNamespace(id="kimi-for-coding"),
                SimpleNamespace(id="kimi-for-coding-highspeed"),
            ]
        )
    )

    assert await kimi_code_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo("k3"),
            ProviderModelInfo("kimi-for-coding"),
            ProviderModelInfo("kimi-for-coding-highspeed"),
        }
    )
    kimi_code_provider._client.models.list.assert_awaited_once_with()
