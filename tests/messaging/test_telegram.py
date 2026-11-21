from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import TelegramError

from free_claude_code.messaging.models import IncomingMessage
from free_claude_code.messaging.platforms.telegram import TelegramRuntime
from free_claude_code.messaging.voice import Transcriber


def _mock_application_builder(mock_app: MagicMock) -> MagicMock:
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = mock_app
    return builder


def _limiter_mock() -> MagicMock:
    limiter = MagicMock()
    limiter.start = MagicMock()
    limiter.shutdown = AsyncMock()
    return limiter


def _telegram_runtime(
    bot_token: str | None = None,
    allowed_user_id: str | None = None,
    *,
    telegram_proxy_url: str | None = None,
    limiter: MagicMock | None = None,
    transcriber: Transcriber | None = None,
    log_raw_messaging_content: bool = False,
    log_api_error_tracebacks: bool = False,
) -> TelegramRuntime:
    return TelegramRuntime(
        bot_token,
        allowed_user_id,
        limiter=limiter or _limiter_mock(),
        transcriber=transcriber,
        telegram_proxy_url=telegram_proxy_url,
        log_raw_messaging_content=log_raw_messaging_content,
        log_api_error_tracebacks=log_api_error_tracebacks,
    )


@pytest.fixture
def telegram_platform():
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="test_token", allowed_user_id="12345")
        return platform


def test_telegram_platform_init_no_token() -> None:
    with patch.dict("os.environ", {}, clear=True):
        platform = _telegram_runtime(bot_token=None)
        assert platform.bot_token is None


@pytest.mark.asyncio
async def test_telegram_platform_start_success(
    monkeypatch: pytest.MonkeyPatch, telegram_platform: TelegramRuntime
) -> None:
    mock_send_message = AsyncMock()
    monkeypatch.setattr(telegram_platform.outbound, "send_message", mock_send_message)
    with patch("telegram.ext.Application.builder") as mock_builder:
        mock_app = MagicMock()
        mock_app.initialize = AsyncMock()
        mock_app.start = AsyncMock()
        mock_app.updater.start_polling = AsyncMock()

        mock_builder.return_value = _mock_application_builder(mock_app)

        await telegram_platform.start()

        assert telegram_platform._connected is True
        mock_app.initialize.assert_called_once()
        mock_app.start.assert_called_once()
        start = telegram_platform._limiter.start
        assert isinstance(start, MagicMock)
        start.assert_called_once_with()
        mock_send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_platform_start_with_proxy() -> None:
    limiter = _limiter_mock()
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(
            bot_token="test_token",
            allowed_user_id="12345",
            telegram_proxy_url="socks5://127.0.0.1:1080",
            limiter=limiter,
        )

    with (
        patch("telegram.ext.Application.builder") as mock_builder,
        patch(
            "free_claude_code.messaging.platforms.telegram.HTTPXRequest"
        ) as request_cls,
    ):
        mock_app = MagicMock()
        mock_app.initialize = AsyncMock()
        mock_app.start = AsyncMock()
        mock_app.updater.start_polling = AsyncMock()

        builder = _mock_application_builder(mock_app)
        mock_builder.return_value = builder
        request = MagicMock()
        update_request = MagicMock()
        request_cls.side_effect = [request, update_request]

        await platform.start()

        assert request_cls.call_count == 2
        request_cls.assert_any_call(
            connection_pool_size=8,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy="socks5://127.0.0.1:1080",
        )
        builder.request.assert_called_once_with(request)
        builder.get_updates_request.assert_called_once_with(update_request)
        assert platform._connected is True
        limiter.start.assert_called_once_with()


@pytest.mark.asyncio
async def test_telegram_platform_send_message_success(
    telegram_platform: TelegramRuntime,
) -> None:
    mock_bot = AsyncMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 999
    mock_bot.send_message.return_value = mock_msg

    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    msg_id = await telegram_platform.outbound.send_message("chat_1", "hello")

    assert msg_id == "999"
    mock_bot.send_message.assert_called_once_with(
        chat_id="chat_1",
        text="hello",
        reply_to_message_id=None,
        parse_mode="MarkdownV2",
    )


@pytest.mark.asyncio
async def test_telegram_platform_edit_message_success(
    telegram_platform: TelegramRuntime,
) -> None:
    mock_bot = AsyncMock()
    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    await telegram_platform.outbound.edit_message("chat_1", "999", "new text")

    mock_bot.edit_message_text.assert_called_once_with(
        chat_id="chat_1", message_id=999, text="new text", parse_mode="MarkdownV2"
    )


@pytest.mark.asyncio
async def test_telegram_platform_delete_messages_uses_batch_api(
    telegram_platform: TelegramRuntime,
) -> None:
    mock_bot = AsyncMock()
    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    await telegram_platform.outbound.delete_messages("chat_1", ["1", "2", "bad"])

    mock_bot.delete_messages.assert_awaited_once_with(
        chat_id="chat_1",
        message_ids=[1, 2],
    )
    mock_bot.delete_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_platform_delete_messages_chunks_batch_api(
    telegram_platform: TelegramRuntime,
) -> None:
    mock_bot = AsyncMock()
    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    await telegram_platform.outbound.delete_messages(
        "chat_1",
        [str(i) for i in range(105)],
    )

    assert mock_bot.delete_messages.await_count == 2
    assert mock_bot.delete_messages.await_args_list[0].kwargs["message_ids"] == list(
        range(100)
    )
    assert mock_bot.delete_messages.await_args_list[1].kwargs["message_ids"] == list(
        range(100, 105)
    )


@pytest.mark.asyncio
async def test_telegram_platform_delete_messages_falls_back_without_batch(
    telegram_platform: TelegramRuntime,
) -> None:
    class BotWithoutBatch:
        def __init__(self) -> None:
            self.delete_message = AsyncMock()

    bot = BotWithoutBatch()
    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = bot

    await telegram_platform.outbound.delete_messages("chat_1", ["1", "2"])

    assert bot.delete_message.await_args_list[0].kwargs == {
        "chat_id": "chat_1",
        "message_id": 1,
    }
    assert bot.delete_message.await_args_list[1].kwargs == {
        "chat_id": "chat_1",
        "message_id": 2,
    }


@pytest.mark.asyncio
async def test_telegram_platform_delete_messages_falls_back_after_batch_failure(
    telegram_platform: TelegramRuntime,
) -> None:
    mock_bot = AsyncMock()
    mock_bot.delete_messages.side_effect = RuntimeError("bulk failed")
    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    await telegram_platform.outbound.delete_messages("chat_1", ["1", "2"])

    mock_bot.delete_messages.assert_awaited_once_with(
        chat_id="chat_1",
        message_ids=[1, 2],
    )
    assert mock_bot.delete_message.await_args_list[0].kwargs == {
        "chat_id": "chat_1",
        "message_id": 1,
    }
    assert mock_bot.delete_message.await_args_list[1].kwargs == {
        "chat_id": "chat_1",
        "message_id": 2,
    }


@pytest.mark.asyncio
async def test_telegram_platform_delete_messages_falls_back_after_swallowed_error(
    telegram_platform: TelegramRuntime,
) -> None:
    mock_bot = AsyncMock()
    mock_bot.delete_messages.side_effect = TelegramError("message can't be deleted")
    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    await telegram_platform.outbound.delete_messages("chat_1", ["1", "2"])

    mock_bot.delete_messages.assert_awaited_once_with(
        chat_id="chat_1",
        message_ids=[1, 2],
    )
    assert mock_bot.delete_message.await_args_list[0].kwargs == {
        "chat_id": "chat_1",
        "message_id": 1,
    }
    assert mock_bot.delete_message.await_args_list[1].kwargs == {
        "chat_id": "chat_1",
        "message_id": 2,
    }


@pytest.mark.asyncio
async def test_telegram_platform_single_delete_still_swallows_known_error(
    telegram_platform: TelegramRuntime,
) -> None:
    mock_bot = AsyncMock()
    mock_bot.delete_message.side_effect = TelegramError("message can't be deleted")
    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    await telegram_platform.outbound.delete_message("chat_1", "1")

    mock_bot.delete_message.assert_awaited_once_with(
        chat_id="chat_1",
        message_id=1,
    )


@pytest.mark.asyncio
async def test_telegram_platform_queue_send_message(
    monkeypatch: pytest.MonkeyPatch, telegram_platform: TelegramRuntime
) -> None:
    mock_limiter = telegram_platform._limiter
    mock_enqueue = AsyncMock(return_value="message")
    monkeypatch.setattr(mock_limiter, "enqueue", mock_enqueue)

    await telegram_platform.outbound.queue_send_message(
        "chat_1", "hello", fire_and_forget=False
    )

    mock_enqueue.assert_called_once()


@pytest.mark.asyncio
async def test_on_telegram_message_authorized(
    telegram_platform: TelegramRuntime,
) -> None:
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
    assert handler.call_args is not None
    incoming: object = handler.call_args.args[0]
    assert isinstance(incoming, IncomingMessage)
    assert incoming.text == "hello"


@pytest.mark.asyncio
async def test_on_telegram_message_unauthorized(
    telegram_platform: TelegramRuntime,
) -> None:
    handler = AsyncMock()
    telegram_platform.on_message(handler)

    mock_update = MagicMock()
    mock_update.message.text = "hello"
    mock_update.effective_user.id = 99999  # Unauthorized

    await telegram_platform._on_telegram_message(mock_update, MagicMock())

    handler.assert_not_called()
