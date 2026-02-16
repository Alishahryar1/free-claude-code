"""Tests for messaging platform factory."""

from unittest.mock import patch, MagicMock

from messaging.factory import create_messaging_platform


class TestCreateMessagingPlatform:
    """Tests for create_messaging_platform factory function."""

    def test_telegram_with_token(self):
        """Create Telegram platform when bot_token is provided."""
        mock_platform = MagicMock()
        with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
            with patch(
                "messaging.telegram.TelegramPlatform", return_value=mock_platform
            ):
                result = create_messaging_platform(
                    "telegram",
                    bot_token="test_token",
                    allowed_user_id="12345",
                )

        assert result is mock_platform

    def test_telegram_without_token(self):
        """Return None when no bot_token for Telegram."""
        result = create_messaging_platform("telegram")
        assert result is None

    def test_telegram_empty_token(self):
        """Return None when bot_token is empty string."""
        result = create_messaging_platform("telegram", bot_token="")
        assert result is None

    def test_unknown_platform(self):
        """Return None for unknown platform types."""
        result = create_messaging_platform("discord")
        assert result is None

    def test_unknown_platform_with_kwargs(self):
        """Return None for unknown platform even with kwargs."""
        result = create_messaging_platform("slack", bot_token="token")
        assert result is None
