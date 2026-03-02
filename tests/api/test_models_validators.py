from unittest.mock import patch

import pytest

from api.models.anthropic import Message, MessagesRequest, TokenCountRequest
from config.settings import Settings
@pytest.fixture
def mock_settings():
    settings = Settings()
    settings.model = "nvidia_nim/target-model-from-settings"
    return settings
def test_messages_request_map_model_claude_to_default(mock_settings):
    with patch("api.models.anthropic.get_settings", return_value=mock_settings):
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        assert request.model == "target-model-from-settings"
        assert request.original_model == "claude-3-opus"
def test_messages_request_map_model_with_provider_prefix(mock_settings):
    with patch("api.models.anthropic.get_settings", return_value=mock_settings):
        request = MessagesRequest(
            model="anthropic/claude-3-haiku",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        assert request.model == "target-model-from-settings"
def test_token_count_request_model_validation(mock_settings):
    with patch("api.models.anthropic.get_settings", return_value=mock_settings):
        request = TokenCountRequest(
            model="claude-3-sonnet", messages=[Message(role="user", content="hello")]
        )

        assert request.model == "target-model-from-settings"
def test_messages_request_model_mapping_logs(mock_settings):
    with (
        patch("api.models.anthropic.get_settings", return_value=mock_settings),
        patch("api.models.anthropic.logger.debug") as mock_log,
    ):
        MessagesRequest(
            model="claude-2.1",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        mock_log.assert_called()
        args = mock_log.call_args[0][0]
        assert "MODEL MAPPING" in args
        assert "claude-2.1" in args
        assert "target-model-from-settings" in args
def test_messages_request_resolved_provider_model_default(mock_settings):
    """resolved_provider_model is set to the full model string."""
    with patch("api.models.anthropic.get_settings", return_value=mock_settings):
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
        assert (
            request.resolved_provider_model == "nvidia_nim/target-model-from-settings"
        )
def test_messages_request_tier_aware_opus_override():
    """Opus model routes to MODEL_OPUS when set."""
    settings = Settings()
    settings.model = "nvidia_nim/fallback-model"
    settings.model_opus = "open_router/deepseek/deepseek-r1"

    with patch("api.models.anthropic.get_settings", return_value=settings):
        request = MessagesRequest(
            model="claude-opus-4-20250514",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
        assert request.model == "deepseek/deepseek-r1"
        assert request.resolved_provider_model == "open_router/deepseek/deepseek-r1"
        assert request.original_model == "claude-opus-4-20250514"
def test_messages_request_tier_aware_haiku_override():
    """Haiku model routes to MODEL_HAIKU when set."""
    settings = Settings()
    settings.model = "nvidia_nim/fallback-model"
    settings.model_haiku = "lmstudio/qwen2.5-7b"

    with patch("api.models.anthropic.get_settings", return_value=settings):
        request = MessagesRequest(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
        assert request.model == "qwen2.5-7b"
        assert request.resolved_provider_model == "lmstudio/qwen2.5-7b"
def test_messages_request_tier_aware_sonnet_override():
    """Sonnet model routes to MODEL_SONNET when set."""
    settings = Settings()
    settings.model = "nvidia_nim/fallback-model"
    settings.model_sonnet = "nvidia_nim/meta/llama-3.3-70b-instruct"

    with patch("api.models.anthropic.get_settings", return_value=settings):
        request = MessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
        assert request.model == "meta/llama-3.3-70b-instruct"
        assert (
            request.resolved_provider_model == "nvidia_nim/meta/llama-3.3-70b-instruct"
        )
def test_messages_request_tier_fallback_when_not_set():
    """When tier override is None, falls back to MODEL."""
    settings = Settings()
    settings.model = "nvidia_nim/fallback-model"
    # model_opus is None

    with patch("api.models.anthropic.get_settings", return_value=settings):
        request = MessagesRequest(
            model="claude-opus-4-20250514",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
        assert request.model == "fallback-model"
        assert request.resolved_provider_model == "nvidia_nim/fallback-model"
def test_token_count_request_tier_aware():
    """TokenCountRequest also uses tier-aware resolution."""
    settings = Settings()
    settings.model = "nvidia_nim/fallback-model"
    settings.model_haiku = "lmstudio/qwen2.5-7b"

    with patch("api.models.anthropic.get_settings", return_value=settings):
        request = TokenCountRequest(
            model="claude-3-haiku-20240307",
            messages=[Message(role="user", content="hello")],
        )
        assert request.model == "qwen2.5-7b"
