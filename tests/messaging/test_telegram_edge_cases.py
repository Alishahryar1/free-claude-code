import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import NetworkError, RetryAfter, TelegramError


def _limiter_mock() -> MagicMock:
    limiter = MagicMock()
    limiter.start = MagicMock()
    limiter.shutdown = AsyncMock()
    return limiter


def _telegram_runtime(*args, limiter=None, transcriber=None, **kwargs):
    from free_claude_code.messaging.platforms.telegram import TelegramRuntime

    return TelegramRuntime(
        *args,
        limiter=limiter or _limiter_mock(),
        transcriber=transcriber,
        **kwargs,
    )


def test_telegram_platform_init_raises_when_dependency_missing():
    with (
        patch(
            "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", False
        ),
        pytest.raises(ImportError),
    ):
        _telegram_runtime(bot_token="x")


@pytest.mark.asyncio
async def test_telegram_platform_start_requires_token():
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True),
    ):
        platform = _telegram_runtime(bot_token=None)
        with pytest.raises(ValueError):
            await platform.start()


@pytest.mark.asyncio
async def test_telegram_platform_quiesce_and_close_without_application():
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")
        platform._application = None
        platform._connected = True
        await platform.quiesce()
        await platform.close()
        assert platform.is_connected is False
        platform._limiter.shutdown.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_telegram_close_cleans_up_partially_initialized_application():
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")
        platform._application = MagicMock()
        platform._application.running = False
        platform._application.updater.running = False
        platform._application.updater.stop = AsyncMock()
        platform._application.stop = AsyncMock()
        platform._application.shutdown = AsyncMock()
        platform.outbound.close = AsyncMock()

        await platform.quiesce()
        await platform.close()

        platform._application.updater.stop.assert_not_awaited()
        platform._application.stop.assert_not_awaited()
        platform.outbound.close.assert_awaited_once_with()
        platform._limiter.shutdown.assert_awaited_once_with()
        platform._application.shutdown.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_telegram_two_phase_lifecycle_drains_before_delivery_close():
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")
        order: list[str] = []
        platform._application = MagicMock()
        platform._application.running = True
        platform._application.updater.running = True
        platform._application.updater.stop = AsyncMock(
            side_effect=lambda: order.append("updater.stop")
        )
        platform._application.stop = AsyncMock(
            side_effect=lambda: order.append("application.stop")
        )
        platform.outbound.close = AsyncMock(
            side_effect=lambda: order.append("outbound.close")
        )
        platform._limiter.shutdown = AsyncMock(
            side_effect=lambda: order.append("limiter.shutdown")
        )
        platform._application.shutdown = AsyncMock(
            side_effect=lambda: order.append("application.shutdown")
        )

        await platform.quiesce()
        assert order == ["updater.stop", "application.stop"]

        await platform.close()

        assert order == [
            "updater.stop",
            "application.stop",
            "outbound.close",
            "limiter.shutdown",
            "application.shutdown",
        ]
        assert platform.is_connected is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_step",
    ["updater.stop", "application.stop"],
)
async def test_telegram_quiesce_attempts_all_steps_after_failure(failing_step):
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")
        order: list[str] = []

        async def record(step: str) -> None:
            order.append(step)
            if step == failing_step:
                raise RuntimeError(step)

        def action(step: str):
            async def run() -> None:
                await record(step)

            return run

        platform._application = MagicMock()
        platform._application.running = True
        platform._application.updater.running = True
        platform._application.updater.stop = AsyncMock(
            side_effect=action("updater.stop")
        )
        platform._application.stop = AsyncMock(side_effect=action("application.stop"))
        platform.outbound.close = AsyncMock(side_effect=action("outbound.close"))
        platform._limiter.shutdown = AsyncMock(side_effect=action("limiter.shutdown"))
        platform._application.shutdown = AsyncMock(
            side_effect=action("application.shutdown")
        )

        with pytest.raises(RuntimeError, match=failing_step):
            await platform.quiesce()

        assert order == ["updater.stop", "application.stop"]
        assert platform.is_connected is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_step",
    ["outbound.close", "limiter.shutdown", "application.shutdown"],
)
async def test_telegram_close_attempts_all_steps_after_failure(failing_step):
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")
        order: list[str] = []

        async def record(step: str) -> None:
            order.append(step)
            if step == failing_step:
                raise RuntimeError(step)

        def action(step: str):
            async def run() -> None:
                await record(step)

            return run

        platform._application = MagicMock()
        platform._application.shutdown = AsyncMock(
            side_effect=action("application.shutdown")
        )
        platform.outbound.close = AsyncMock(side_effect=action("outbound.close"))
        platform._limiter.shutdown = AsyncMock(side_effect=action("limiter.shutdown"))

        with pytest.raises(RuntimeError, match=failing_step):
            await platform.close()

        assert order == [
            "outbound.close",
            "limiter.shutdown",
            "application.shutdown",
        ]


@pytest.mark.asyncio
async def test_with_retry_returns_none_when_message_not_modified_network_error():
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")

        async def _f():
            raise NetworkError("Message is not modified")

        assert await platform.outbound._with_retry(_f) is None


@pytest.mark.asyncio
async def test_with_retry_retries_network_error_then_succeeds(monkeypatch):
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        calls = {"n": 0}

        async def _f():
            calls["n"] += 1
            if calls["n"] == 1:
                raise NetworkError("temporary")
            return "ok"

        assert await platform.outbound._with_retry(_f) == "ok"
        assert calls["n"] == 2


@pytest.mark.asyncio
async def test_with_retry_honors_retry_after_timedelta(monkeypatch):
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        calls = {"n": 0}

        async def _f():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RetryAfter(retry_after=timedelta(seconds=0.01))
            return "ok"

        assert await platform.outbound._with_retry(_f) == "ok"
        assert calls["n"] == 2


@pytest.mark.asyncio
async def test_with_retry_drops_parse_mode_on_markdown_entity_error():
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")

        calls = []

        async def _f(parse_mode=None):
            calls.append(parse_mode)
            if len(calls) == 1:
                raise TelegramError("Can't parse entities: bad markdown")
            return "ok"

        assert await platform.outbound._with_retry(_f, parse_mode="MarkdownV2") == "ok"
        assert calls == ["MarkdownV2", None]


@pytest.mark.asyncio
async def test_with_retry_can_raise_known_message_errors_for_bulk_fallback():
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")

        async def _f():
            raise TelegramError("message can't be deleted")

        with pytest.raises(TelegramError):
            await platform.outbound._with_retry(
                _f,
                suppress_known_message_errors=False,
            )


@pytest.mark.asyncio
async def test_queue_send_message_uses_required_limiter():
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")
        platform._application = MagicMock()
        mock_msg = MagicMock()
        mock_msg.message_id = 1
        platform._application.bot = AsyncMock()
        platform._application.bot.send_message = AsyncMock(return_value=mock_msg)

        async def enqueue(operation, dedup_key=None):
            return await operation()

        platform._limiter.enqueue = AsyncMock(side_effect=enqueue)
        assert (
            await platform.outbound.queue_send_message("c", "t", fire_and_forget=False)
            == "1"
        )
        platform._limiter.enqueue.assert_awaited_once()
        platform._application.bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_queue_edit_message_uses_required_limiter():
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")
        platform._application = MagicMock()
        platform._application.bot = AsyncMock()
        platform._application.bot.edit_message_text = AsyncMock()

        async def enqueue(operation, dedup_key=None):
            return await operation()

        platform._limiter.enqueue = AsyncMock(side_effect=enqueue)
        await platform.outbound.queue_edit_message("c", "1", "t", fire_and_forget=False)
        platform._limiter.enqueue.assert_awaited_once()
        platform._application.bot.edit_message_text.assert_awaited_once()


def test_fire_and_forget_non_coroutine_uses_ensure_future(monkeypatch):
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")

        ef = MagicMock()
        monkeypatch.setattr(asyncio, "ensure_future", ef)

        platform.outbound.fire_and_forget(MagicMock())
        ef.assert_called_once()


@pytest.mark.asyncio
async def test_on_start_command_replies_and_forwards():
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")
        with patch.object(
            platform, "_on_telegram_message", new_callable=AsyncMock
        ) as mock_msg:
            update = MagicMock()
            update.message.reply_text = AsyncMock()

            await platform._on_start_command(update, MagicMock())
            update.message.reply_text.assert_awaited_once()
            mock_msg.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_telegram_message_handler_error_sends_error_message():
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t", allowed_user_id="123")
        with patch.object(
            platform.outbound, "send_message", new_callable=AsyncMock
        ) as mock_send:

            async def _boom(_incoming):
                raise RuntimeError("bad")

            platform.on_message(_boom)

            update = MagicMock()
            update.message.text = "hello"
            update.message.message_id = 7
            update.message.reply_to_message = None
            update.effective_user.id = 123
            update.effective_chat.id = 456

            await platform._on_telegram_message(update, MagicMock())
            mock_send.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_start_retries_on_network_error(monkeypatch):
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="token", allowed_user_id=None)

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        with patch("telegram.ext.Application.builder") as mock_builder:
            mock_app = MagicMock()
            mock_app.initialize = AsyncMock(side_effect=[NetworkError("no"), None])
            mock_app.start = AsyncMock()
            mock_app.updater = None

            mock_builder.return_value.token.return_value.request.return_value.build.return_value = mock_app

            await platform.start()
            assert platform.is_connected is True
            assert mock_app.initialize.await_count == 2
            mock_app.start.assert_awaited_once_with()
            platform._limiter.start.assert_called_once_with()


@pytest.mark.asyncio
async def test_telegram_polling_retry_does_not_restart_running_application(
    monkeypatch,
):
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="token", allowed_user_id=None)

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        with patch("telegram.ext.Application.builder") as mock_builder:
            mock_app = MagicMock()
            mock_app.initialize = AsyncMock()
            mock_app.start = AsyncMock()
            mock_app.updater.start_polling = AsyncMock(
                side_effect=[NetworkError("temporary polling failure"), None]
            )
            mock_builder.return_value.token.return_value.request.return_value.build.return_value = mock_app

            await platform.start()

            assert platform.is_connected is True
            mock_app.initialize.assert_awaited_once_with()
            mock_app.start.assert_awaited_once_with()
            assert mock_app.updater.start_polling.await_count == 2
            platform._limiter.start.assert_called_once_with()


@pytest.mark.asyncio
async def test_edit_message_with_text_exceeding_4096_raises():
    """edit_message with text > 4096 raises TelegramError (BadRequest)."""
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")
        platform._application = MagicMock()
        platform._application.bot = AsyncMock()
        platform._application.bot.edit_message_text = AsyncMock(
            side_effect=TelegramError("Bad Request: message is too long")
        )

        with pytest.raises(TelegramError):
            await platform.outbound.edit_message("c", "1", "x" * 5000)


@pytest.mark.asyncio
async def test_edit_message_empty_string():
    """edit_message with empty string - Telegram accepts (no-op edit)."""
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")
        platform._application = MagicMock()
        platform._application.bot = AsyncMock()
        platform._application.bot.edit_message_text = AsyncMock()

        await platform.outbound.edit_message("c", "1", "")
        platform._application.bot.edit_message_text.assert_awaited_once_with(
            chat_id="c", message_id=1, text="", parse_mode="MarkdownV2"
        )


@pytest.mark.asyncio
async def test_send_message_empty_string():
    """send_message with empty string - Telegram may reject; we pass through."""
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")
        platform._application = MagicMock()
        mock_msg = MagicMock()
        mock_msg.message_id = 1
        platform._application.bot = AsyncMock()
        platform._application.bot.send_message = AsyncMock(return_value=mock_msg)

        msg_id = await platform.outbound.send_message("c", "")
        assert msg_id == "1"
        platform._application.bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_telegram_message_non_text_update_ignored():
    """Update with message.photo but no text returns early without calling handler."""
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t", allowed_user_id="123")
        handler = AsyncMock()
        platform.on_message(handler)

        update = MagicMock()
        update.message.text = None
        update.message.photo = [MagicMock()]
        update.message.message_id = 7
        update.message.reply_to_message = None
        update.effective_user.id = 123
        update.effective_chat.id = 456

        await platform._on_telegram_message(update, MagicMock())
        handler.assert_not_called()


@pytest.mark.asyncio
async def test_with_retry_message_not_found_returns_none():
    """'message to edit not found' returns None without retry."""
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = _telegram_runtime(bot_token="t")

        async def _f():
            raise TelegramError("message to edit not found")

        result = await platform.outbound._with_retry(_f)
        assert result is None
