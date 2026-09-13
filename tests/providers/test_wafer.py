"""Tests for the Wafer OpenAI-chat provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import WAFER_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest, Tool
from free_claude_code.providers.openai_chat import (
    OpenAIChatProvider,
)
from tests.providers.support import (
    REASONING_OFF,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)


@pytest.fixture
def wafer_config():
    return make_provider_config(
        api_key="test-wafer-key",
        base_url=WAFER_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def wafer_provider(wafer_config):
    return profiled_provider(
        "wafer",
        wafer_config,
        admission=immediate_admission(),
    )


def test_default_base_url():
    assert WAFER_DEFAULT_BASE == "https://pass.wafer.ai/v1"


def test_init_uses_openai_chat_provider(wafer_provider):
    assert isinstance(wafer_provider, OpenAIChatProvider)
    assert wafer_provider._api_key == "test-wafer-key"
    assert wafer_provider._base_url == WAFER_DEFAULT_BASE
    assert wafer_provider._provider_name == "WAFER"


def test_build_request_body_openai_shape_and_defaults(wafer_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "DeepSeek-V4-Pro",
            "messages": [Message(role="user", content="Hello")],
            "tools": [
                Tool(
                    name="echo",
                    description="Echo input",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }
    )

    body = wafer_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == "DeepSeek-V4-Pro"
    assert body["messages"][0] == {"role": "user", "content": "Hello"}
    assert body["tools"][0]["function"]["name"] == "echo"
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_honors_effective_no_thinking(wafer_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "DeepSeek-V4-Pro",
            "messages": [{"role": "user", "content": "Explore the codebase."}],
        }
    )

    body = wafer_provider._chat._build_request_body(request, reasoning=REASONING_OFF)

    assert body["reasoning_effort"] == "none"


def test_build_request_body_preserves_request_disabled_thinking(wafer_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "DeepSeek-V4-Pro",
            "messages": [{"role": "user", "content": "Explore the codebase."}],
            "thinking": {"type": "disabled"},
        }
    )

    body = wafer_provider._chat._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["reasoning_effort"] == "none"


def test_build_request_body_uses_resolved_policy_without_inspecting_model(
    wafer_provider,
):
    request = MessagesRequest.model_validate(
        {
            "model": "DeepSeek-V4-Pro",
            "messages": [{"role": "user", "content": "Explore the codebase."}],
        }
    )

    body = wafer_provider._chat._build_request_body(request, reasoning=REASONING_OFF)

    assert body["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_lists_models_from_openai_models_endpoint(wafer_provider):
    wafer_provider._client.models.list = AsyncMock(
        return_value=MagicMock(
            data=[MagicMock(id="DeepSeek-V4-Pro"), MagicMock(id="MiniMax-M2.7")]
        )
    )

    assert await wafer_provider.list_model_infos() == frozenset(
        {ProviderModelInfo("DeepSeek-V4-Pro"), ProviderModelInfo("MiniMax-M2.7")}
    )

    wafer_provider._client.models.list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(wafer_provider):
    wafer_provider._client = MagicMock()
    wafer_provider._client.close = AsyncMock()

    await wafer_provider.cleanup()

    wafer_provider._client.close.assert_awaited_once()
