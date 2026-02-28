from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from messaging.platforms.telegram import TelegramPlatform


@pytest.fixture
def telegram_platform():
    with patch("messaging.platforms.telegram.TELEGRAM_AVAILABLE", True):
        platform = TelegramPlatform(bot_token="test_token", allowed_user_id="12345")
        return platform


def test_telegram_platform_init_no_token():
    with patch.dict("os.environ", {}, clear=True):
        platform = TelegramPlatform(bot_token=None)
        assert platform.bot_token is None


@pytest.mark.asyncio
async def test_telegram_platform_start_success(telegram_platform):
    with patch("telegram.ext.Application.builder") as mock_builder:
        mock_app = MagicMock()
        mock_app.initialize = AsyncMock()
        mock_app.start = AsyncMock()
        mock_app.updater.start_polling = AsyncMock()

        mock_builder.return_value.token.return_value.request.return_value.build.return_value = mock_app

        # Mock MessagingRateLimiter
        with patch("messaging.limiter.MessagingRateLimiter.get_instance", AsyncMock()):
            await telegram_platform.start()

            assert telegram_platform._connected is True
            mock_app.initialize.assert_called_once()
            mock_app.start.assert_called_once()


@pytest.mark.asyncio
async def test_telegram_platform_send_message_success(telegram_platform):
    mock_bot = AsyncMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 999
    mock_bot.send_message.return_value = mock_msg

    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    msg_id = await telegram_platform.send_message("chat_1", "hello")

    assert msg_id == "999"
    mock_bot.send_message.assert_called_once_with(
        chat_id="chat_1",
        text="hello",
        reply_to_message_id=None,
        parse_mode="MarkdownV2",
    )


@pytest.mark.asyncio
async def test_telegram_platform_edit_message_success(telegram_platform):
    mock_bot = AsyncMock()
    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    await telegram_platform.edit_message("chat_1", "999", "new text")

    mock_bot.edit_message_text.assert_called_once_with(
        chat_id="chat_1", message_id=999, text="new text", parse_mode="MarkdownV2"
    )


@pytest.mark.asyncio
async def test_telegram_platform_queue_send_message(telegram_platform):
    mock_limiter = AsyncMock()
    telegram_platform._limiter = mock_limiter

    await telegram_platform.queue_send_message("chat_1", "hello", fire_and_forget=False)

    mock_limiter.enqueue.assert_called_once()


@pytest.mark.asyncio
async def test_on_telegram_message_authorized(telegram_platform):
    handler = AsyncMock()
    telegram_platform.on_message(handler)

    mock_update = MagicMock()
    mock_update.message.text = "hello"
    mock_update.message.message_id = 1
    mock_update.effective_user.id = 12345
    mock_update.effective_chat.id = 6789
    mock_update.message.reply_to_message = None

    await telegram_platform._on_telegram_message(mock_update, MagicMock())

    handler.assert_called_once()
    incoming = handler.call_args[0][0]
    assert incoming.text == "hello"
    assert incoming.user_id == "12345"


@pytest.mark.asyncio
async def test_on_telegram_message_unauthorized(telegram_platform):
    handler = AsyncMock()
    telegram_platform.on_message(handler)

    mock_update = MagicMock()
    mock_update.message.text = "hello"
    mock_update.effective_user.id = 99999  # Unauthorized

    await telegram_platform._on_telegram_message(mock_update, MagicMock())

    handler.assert_not_called()
