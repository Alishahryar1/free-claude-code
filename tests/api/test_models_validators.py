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


def test_messages_request_map_model_non_claude_unchanged(mock_settings):
    with patch("api.models.anthropic.get_settings", return_value=mock_settings):
        request = MessagesRequest(
            model="gpt-4",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        # normalize_model_name returns original if not Claude
        assert request.model == "gpt-4"


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
    """Test that model mapping is logged."""
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


def test_messages_request_no_mapping_logs():
    """Test that non-mapped models are also logged."""
    settings = Settings()
    settings.model = "nvidia_nim/default-model"
    settings.haiku_model = None
    settings.sonnet_model = None
    settings.opus_model = None

    with (
        patch("api.models.anthropic.get_settings", return_value=settings),
        patch("api.models.anthropic.logger.debug") as mock_log,
    ):
        MessagesRequest(
            model="gpt-4",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        mock_log.assert_called()
        args = mock_log.call_args[0][0]
        assert "MODEL:" in args
        assert "gpt-4" in args
        assert "(no mapping)" in args


def test_messages_request_claude_no_specific_mapping_logs():
    """Test that Claude models without specific mapping are logged."""
    settings = Settings()
    settings.model = "nvidia_nim/default-model"
    settings.haiku_model = None
    settings.sonnet_model = None
    settings.opus_model = None

    with (
        patch("api.models.anthropic.get_settings", return_value=settings),
        patch("api.models.anthropic.logger.debug") as mock_log,
    ):
        MessagesRequest(
            model="claude-3-haiku",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        mock_log.assert_called()
        args = mock_log.call_args[0][0]
        # Should show mapping to default
        assert "MODEL MAPPING" in args
        assert "claude-3-haiku" in args
        assert "default-model" in args


def test_messages_request_haiku_mapping_with_haiku_model_set():
    """Test Haiku models map to haiku_model when set."""
    settings = Settings()
    settings.model = "nvidia_nim/default-model"
    settings.haiku_model = "open_router/google/gemma-3-4b"
    settings.sonnet_model = None
    settings.opus_model = None

    with patch("api.models.anthropic.get_settings", return_value=settings):
        request = MessagesRequest(
            model="claude-3-haiku",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        assert request.model == "open_router/google/gemma-3-4b"
        assert request.original_model == "claude-3-haiku"


def test_messages_request_sonnet_mapping_with_sonnet_model_set():
    """Test Sonnet models map to sonnet_model when set."""
    settings = Settings()
    settings.model = "nvidia_nim/default-model"
    settings.haiku_model = None
    settings.sonnet_model = "open_router/stepfun/step-3.5-flash:free"
    settings.opus_model = None

    with patch("api.models.anthropic.get_settings", return_value=settings):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        assert request.model == "open_router/stepfun/step-3.5-flash:free"


def test_messages_request_opus_mapping_with_opus_model_set():
    """Test Opus models map to opus_model when set."""
    settings = Settings()
    settings.model = "nvidia_nim/default-model"
    settings.haiku_model = None
    settings.sonnet_model = None
    settings.opus_model = "nvidia_nim/meta/llama3-70b-instruct"

    with patch("api.models.anthropic.get_settings", return_value=settings):
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        assert request.model == "nvidia_nim/meta/llama3-70b-instruct"


def test_messages_request_all_three_mappings_set():
    """Test with all three Claude model mappings set."""
    settings = Settings()
    settings.model = "nvidia_nim/default-model"
    settings.haiku_model = "open_router/haiku-model"
    settings.sonnet_model = "open_router/sonnet-model"
    settings.opus_model = "open_router/opus-model"

    with patch("api.models.anthropic.get_settings", return_value=settings):
        # Haiku
        request_haiku = MessagesRequest(
            model="claude-3-haiku",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
        assert request_haiku.model == "open_router/haiku-model"

        # Sonnet
        request_sonnet = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
        assert request_sonnet.model == "open_router/sonnet-model"

        # Opus
        request_opus = MessagesRequest(
            model="claude-3-opus",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
        assert request_opus.model == "open_router/opus-model"

        # Other Claude models fall back to default
        request_other = MessagesRequest(
            model="claude-2.1",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
        assert request_other.model == "default-model"
