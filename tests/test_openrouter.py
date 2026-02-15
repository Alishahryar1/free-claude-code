"""Tests for OpenRouter provider."""

import pytest
from unittest.mock import AsyncMock, patch

from api.models.anthropic import MessagesRequest, Message
from providers.openrouter import OpenRouterProvider
from providers.base import ProviderConfig
from config.nim import NimSettings


@pytest.fixture
def provider_config():
    return ProviderConfig(
        api_key="test_openrouter_key",
        base_url="https://openrouter.ai/api/v1",
        rate_limit=20,
        rate_window=60,
        nim_settings=NimSettings(),
    )


@pytest.fixture
def openrouter_provider(provider_config):
    with patch("providers.openrouter.client.GlobalRateLimiter") as mock:
        mock.get_instance.return_value.execute_with_retry = AsyncMock()
        provider = OpenRouterProvider(provider_config)
        return provider


@pytest.mark.asyncio
async def test_init(provider_config):
    """Test provider initialization."""
    with patch("providers.openrouter.client.AsyncOpenAI") as mock_openai:
        provider = OpenRouterProvider(provider_config)
        assert provider._api_key == "test_openrouter_key"
        assert provider._base_url == "https://openrouter.ai/api/v1"
        mock_openai.assert_called_once_with(
            api_key="test_openrouter_key",
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
            timeout=300.0,
        )


@pytest.mark.asyncio
async def test_build_request_body(openrouter_provider):
    """Test request body conversion."""
    req = MessagesRequest(
        model="google/gemini-2.0-flash-001",
        messages=[Message(role="user", content="Hi")],
        max_tokens=100,
    )
    body = openrouter_provider._build_request_body(req)
    assert body["model"] == "google/gemini-2.0-flash-001"
    assert len(body["messages"]) >= 1
    assert body["max_tokens"] == 100
