"""Tests for the Poe OpenAI-compatible provider profile.

Documented API behavior comes from
https://creator.poe.com/docs/external-applications/openai-compatible-api:
top-level ``reasoning_effort`` is ignored (efforts must travel through
``extra_body``), ``response_format``/``n``/penalties are silently ignored
server-side, and thinking is replayed through ``reasoning_content``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import POE_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
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

_MODEL = "gpt-5.4-nano"


@pytest.fixture
def poe_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "poe",
        make_provider_config(
            api_key="test-poe-key",
            base_url=POE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="poe"),
    )


def test_constructs_standard_openai_chat_provider(
    poe_provider: OpenAIChatProvider,
) -> None:
    assert isinstance(poe_provider, OpenAIChatProvider)
    assert poe_provider._provider_name == "POE"
    assert poe_provider._api_key == "test-poe-key"
    assert poe_provider._base_url == POE_DEFAULT_BASE


@pytest.mark.parametrize(
    ("reasoning", "expected_extra_body"),
    [
        (REASONING_DEFAULT, None),
        (REASONING_ON, {"reasoning_effort": "high"}),
        (REASONING_OFF, {"reasoning_effort": "none"}),
        (
            reasoning_for(
                MessagesRequest.model_validate(
                    {
                        "model": _MODEL,
                        "messages": [{"role": "user", "content": "x"}],
                        "thinking": {"type": "enabled", "budget_tokens": 1024},
                    }
                )
            ),
            {"thinking_budget": 1024},
        ),
    ],
)
def test_encodes_reasoning_through_extra_body(
    poe_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    expected_extra_body: dict | None,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "Summarize the diff."}],
        }
    )

    body = poe_provider._chat._build_request_body(request, reasoning=reasoning)

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["model"] == _MODEL
    extra = body.get("extra_body") or {}
    if expected_extra_body is None:
        assert extra == {}
    else:
        assert extra == expected_extra_body
    # Poe ignores top-level reasoning_effort, so it must never be sent there.
    assert "reasoning_effort" not in body
    assert "reasoning" not in body
    assert "thinking" not in body


def test_normalizes_n_to_one_like_groq(
    poe_provider: OpenAIChatProvider,
) -> None:
    """Poe documents n as "must be exactly 1"."""

    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "x"}],
        }
    )

    with patch(
        "free_claude_code.providers.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "x"}],
            "n": 4,
        }
        body = poe_provider._chat._build_request_body(
            request,
            reasoning=REASONING_DEFAULT,
        )

    assert body["n"] == 1


def test_replays_thinking_in_reasoning_content_field(
    poe_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [
                {"role": "user", "content": "Solve it."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Work through it."},
                        {"type": "text", "text": "The answer is 42."},
                    ],
                },
                {"role": "user", "content": "Continue."},
            ],
        }
    )

    body = poe_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": "Work through it.",
    }


def test_allows_custom_bot_parameters_through_extra_body(
    poe_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"aspect": "1280x720"},
        }
    )

    body = poe_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["extra_body"]["aspect"] == "1280x720"


def test_rejects_extra_body_overriding_canonical_fields(
    poe_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"max_tokens": 10},
        }
    )

    with pytest.raises(Exception, match="canonical"):
        poe_provider._chat._build_request_body(
            request,
            reasoning=reasoning_for(request),
        )


def _poe_model_payload() -> dict:
    """Model shapes follow the live /models response."""
    return {
        "object": "list",
        "data": [
            {
                "id": _MODEL,
                "supported_endpoints": ["/v1/chat/completions", "/v1/responses"],
                "architecture": {"input_modalities": ["text", "image"]},
                "reasoning": {"supports_reasoning_effort": True},
                "context_window": {
                    "context_length": 400000,
                    "max_output_tokens": 128000,
                },
            },
            {
                "id": "claude-sonnet-4.6",
                "supported_endpoints": ["/v1/chat/completions"],
                "architecture": {"input_modalities": ["text", "image"]},
                "reasoning": {"supports_reasoning_effort": False},
                "context_window": {
                    "context_length": 983040,
                    "max_output_tokens": 64000,
                },
            },
            {
                "id": "nano-banana-pro",
                "supported_endpoints": ["/v1/chat/completions"],
                "architecture": {"input_modalities": ["text", "image"]},
                "reasoning": {"supports_reasoning_effort": False},
                "context_window": {
                    "context_length": 65536,
                    "max_output_tokens": 8192,
                },
            },
            # Non-chat endpoints are filtered out by required_path_values.
            {
                "id": "gpt-image-1.5",
                "supported_endpoints": ["/v1/responses"],
                "architecture": {"input_modalities": ["text", "image"]},
                "context_window": {"context_length": 400000},
            },
            # video/audio modalities are outside FCC's text/image enum and
            # must be filtered, not crash parsing.
            {
                "id": "gemini-3.1-pro",
                "supported_endpoints": ["/v1/chat/completions"],
                "architecture": {
                    "input_modalities": ["text", "image", "video", "audio"]
                },
                "reasoning": {"supports_reasoning_effort": True},
                "context_window": {
                    "context_length": 1048576,
                    "max_output_tokens": 65536,
                },
            },
        ],
    }


@pytest.mark.asyncio
async def test_lists_models_from_models_endpoint(
    poe_provider: OpenAIChatProvider,
) -> None:
    poe_provider._client.get = AsyncMock(return_value=_poe_model_payload())

    model_infos = await poe_provider.list_model_infos()

    by_id = {info.model_id: info for info in model_infos}
    assert set(by_id) == {
        _MODEL,
        "claude-sonnet-4.6",
        "nano-banana-pro",
        "gemini-3.1-pro",
    }
    gpt = by_id[_MODEL]
    assert gpt.supports_thinking is True
    assert gpt.input_modalities == frozenset(
        {ModelInputModality.TEXT, ModelInputModality.IMAGE}
    )
    assert gpt.context_window_tokens == 400000
    assert gpt.max_output_tokens == 128000
    sonnet = by_id["claude-sonnet-4.6"]
    assert sonnet.supports_thinking is False
    gemini = by_id["gemini-3.1-pro"]
    # video/audio entries outside the FCC enum are dropped, text/image kept.
    assert gemini.input_modalities == frozenset(
        {ModelInputModality.TEXT, ModelInputModality.IMAGE}
    )
    poe_provider._client.get.assert_awaited_once()
    call = poe_provider._client.get.await_args
    assert call is not None
    assert call.args == ("/models",)


@pytest.mark.asyncio
async def test_rejects_empty_model_ids(poe_provider: OpenAIChatProvider) -> None:
    poe_provider._client.get = AsyncMock(return_value={"data": [{"id": ""}]})

    with pytest.raises(ModelListResponseError, match="include id"):
        await poe_provider.list_model_infos()


@pytest.mark.asyncio
async def test_rejects_payload_without_chat_models(
    poe_provider: OpenAIChatProvider,
) -> None:
    poe_provider._client.get = AsyncMock(
        return_value={
            "data": [
                {
                    "id": "gpt-image-1.5",
                    "supported_endpoints": ["/v1/responses"],
                }
            ]
        }
    )

    with pytest.raises(ModelListResponseError, match="model ids"):
        await poe_provider.list_model_infos()
