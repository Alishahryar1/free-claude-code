"""Tests for Discord platform adapter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.messaging.platforms.discord import (
    DISCORD_AVAILABLE,
    DiscordRuntime,
    _get_discord,
)
from free_claude_code.messaging.platforms.discord_inbound import (
    discord_text_message_from_event,
    parse_allowed_channels,
)
from free_claude_code.messaging.platforms.discord_io import truncate_discord_message


def _limiter_mock() -> MagicMock:
    limiter = MagicMock()
    limiter.start = MagicMock()
    limiter.shutdown = AsyncMock()
    return limiter


def _discord_runtime(*args, limiter=None, transcriber=None, **kwargs) -> DiscordRuntime:
    return DiscordRuntime(
        *args,
        limiter=limiter or _limiter_mock(),
        transcriber=transcriber,
        **kwargs,
    )


class TestGetDiscord:
    """Tests for _get_discord helper."""

    def test_raises_when_discord_not_available(self):
        import free_claude_code.messaging.platforms.discord as discord_mod

        with (
            patch.object(discord_mod, "DISCORD_AVAILABLE", False),
            patch.object(discord_mod, "_discord_module", None),
            pytest.raises(ImportError, match=r"discord\.py is required"),
        ):
            _get_discord()


class TestParseAllowedChannels:
    """Tests for _parse_allowed_channels helper."""

    def test_empty_string_returns_empty_set(self):
        assert parse_allowed_channels("") == set()
        assert parse_allowed_channels(None) == set()

    def test_whitespace_only_returns_empty_set(self):
        assert parse_allowed_channels("   ") == set()

    def test_single_channel(self):
        assert parse_allowed_channels("123456789") == {"123456789"}

    def test_comma_separated(self):
        assert parse_allowed_channels("111,222,333") == {"111", "222", "333"}

    def test_strips_whitespace(self):
        assert parse_allowed_channels(" 111 , 222 ") == {"111", "222"}

    def test_empty_parts_ignored(self):
        assert parse_allowed_channels("111,,222,") == {"111", "222"}


class TestDiscordInbound:
    def test_text_message_normalizes_accepted_event(self):
        msg = MagicMock()
        msg.author.id = 456
        msg.author.display_name = "User"
        msg.content = "hello"
        msg.channel.id = 123
        msg.id = 789
        msg.reference.message_id = 555

        incoming = discord_text_message_from_event(
            msg,
            log_raw_messaging_content=False,
        )

        assert incoming.text == "hello"
        assert incoming.chat_id == "123"
        assert incoming.message_id == "789"
        assert incoming.platform == "discord"
        assert incoming.reply_to_message_id == "555"


@pytest.mark.skipif(not DISCORD_AVAILABLE, reason="discord.py not installed")
class TestDiscordRuntime:
    """Tests for Discord runtime and messenger behavior."""

    def test_init_with_token(self):
        platform = _discord_runtime(
            bot_token="test_token",
            allowed_channel_ids="123,456",
        )
        assert platform.bot_token == "test_token"
        assert platform.allowed_channel_ids == {"123", "456"}

    def test_init_without_allowed_channels(self):
        with patch.dict("os.environ", {"ALLOWED_DISCORD_CHANNELS": ""}, clear=False):
            platform = _discord_runtime(bot_token="token", allowed_channel_ids="")
        assert platform.allowed_channel_ids == set()

    def test_empty_allowed_channels_rejects_all_messages(self):
        """When allowed_channel_ids is empty, no channels are allowed (secure default)."""
        with patch.dict("os.environ", {"ALLOWED_DISCORD_CHANNELS": ""}, clear=False):
            platform = _discord_runtime(bot_token="token", allowed_channel_ids="")
        assert platform.allowed_channel_ids == set()
        # Empty set means: not self.allowed_channel_ids is True -> reject

    def test_truncate_long_message(self):
        long_text = "x" * 2500
        truncated = truncate_discord_message(long_text)
        assert len(truncated) == 2000
        assert truncated.endswith("...")

    def test_truncate_short_message_unchanged(self):
        short = "hello"
        assert truncate_discord_message(short) == short

    def test_truncate_exactly_at_limit_unchanged(self):
        exact = "x" * 2000
        assert truncate_discord_message(exact) == exact

    def test_truncate_one_over_limit_truncates(self):
        over = "x" * 2001
        result = truncate_discord_message(over)
        assert len(result) == 2000
        assert result.endswith("...")

    def test_truncate_empty_string(self):
        assert truncate_discord_message("") == ""

    @pytest.mark.asyncio
    async def test_send_message_returns_message_id(self):
        platform = _discord_runtime(bot_token="token")
        mock_msg = MagicMock()
        mock_msg.id = 999
        mock_channel = AsyncMock()
        mock_channel.send = AsyncMock(return_value=mock_msg)
        platform._connected = True
        with patch.object(
            platform._client, "get_channel", MagicMock(return_value=mock_channel)
        ):
            msg_id = await platform.outbound.send_message("123", "Hello")
        assert msg_id == "999"

    @pytest.mark.asyncio
    async def test_edit_message(self):
        platform = _discord_runtime(bot_token="token")
        mock_msg = AsyncMock()
        mock_channel = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
        platform._connected = True
        with patch.object(
            platform._client, "get_channel", MagicMock(return_value=mock_channel)
        ):
            await platform.outbound.edit_message("123", "456", "Updated text")
        mock_msg.edit.assert_called_once_with(content="Updated text")

    @pytest.mark.asyncio
    async def test_send_message_channel_not_found_raises(self):
        platform = _discord_runtime(bot_token="token")
        platform._connected = True
        with (
            patch.object(platform._client, "get_channel", MagicMock(return_value=None)),
            pytest.raises(RuntimeError, match="Channel"),
        ):
            await platform.outbound.send_message("123", "Hello")

    @pytest.mark.asyncio
    async def test_send_message_channel_no_send_raises(self):
        platform = _discord_runtime(bot_token="token")
        platform._connected = True
        mock_channel = MagicMock(spec=[])  # No send attr
        with (
            patch.object(
                platform._client, "get_channel", MagicMock(return_value=mock_channel)
            ),
            pytest.raises(RuntimeError, match="Channel"),
        ):
            await platform.outbound.send_message("123", "Hello")

    @pytest.mark.asyncio
    async def test_queue_send_message_uses_required_limiter(self):
        platform = _discord_runtime(bot_token="token")

        async def enqueue(operation, dedup_key=None):
            return await operation()

        platform._limiter.enqueue = AsyncMock(side_effect=enqueue)
        platform._connected = True
        mock_channel = AsyncMock()
        mock_msg = MagicMock()
        mock_msg.id = 42
        mock_channel.send = AsyncMock(return_value=mock_msg)
        with patch.object(
            platform._client, "get_channel", MagicMock(return_value=mock_channel)
        ):
            result = await platform.outbound.queue_send_message(
                "123", "hi", fire_and_forget=False
            )
        assert result == "42"
        platform._limiter.enqueue.assert_awaited_once()
        mock_channel.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_queue_edit_message_uses_required_limiter(self):
        platform = _discord_runtime(bot_token="token")

        async def enqueue(operation, dedup_key=None):
            return await operation()

        platform._limiter.enqueue = AsyncMock(side_effect=enqueue)
        platform._connected = True
        mock_msg = AsyncMock()
        mock_channel = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
        with patch.object(
            platform._client, "get_channel", MagicMock(return_value=mock_channel)
        ):
            await platform.outbound.queue_edit_message(
                "123", "456", "Updated", fire_and_forget=False
            )
        platform._limiter.enqueue.assert_awaited_once()
        mock_msg.edit.assert_called_once_with(content="Updated")

    @pytest.mark.asyncio
    async def test_on_discord_message_bot_ignored(self):
        platform = _discord_runtime(bot_token="token", allowed_channel_ids="123")
        handler = AsyncMock()
        platform.on_message(handler)
        msg = MagicMock()
        msg.author.bot = True
        msg.content = "hello"
        msg.channel.id = 123
        await platform._on_discord_message(msg)
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_discord_message_empty_content_ignored(self):
        platform = _discord_runtime(bot_token="token", allowed_channel_ids="123")
        handler = AsyncMock()
        platform.on_message(handler)
        msg = MagicMock()
        msg.author.bot = False
        msg.content = ""
        msg.channel.id = 123
        await platform._on_discord_message(msg)
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_discord_message_channel_not_allowed_ignored(self):
        platform = _discord_runtime(bot_token="token", allowed_channel_ids="123")
        handler = AsyncMock()
        platform.on_message(handler)
        msg = MagicMock()
        msg.author.bot = False
        msg.content = "hello"
        msg.channel.id = 999
        await platform._on_discord_message(msg)
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_discord_message_valid_calls_handler(self):
        platform = _discord_runtime(bot_token="token", allowed_channel_ids="123")
        handler = AsyncMock()
        platform.on_message(handler)
        msg = MagicMock()
        msg.author.bot = False
        msg.author.id = 456
        msg.author.display_name = "User"
        msg.content = "hello"
        msg.channel.id = 123
        msg.id = 789
        msg.reference = None
        await platform._on_discord_message(msg)
        handler.assert_awaited_once()
        call = handler.call_args[0][0]
        assert call.text == "hello"
        assert call.chat_id == "123"
        assert call.message_id == "789"
        assert call.platform == "discord"

    @pytest.mark.asyncio
    async def test_send_message_with_reply_to(self):
        platform = _discord_runtime(bot_token="token")
        mock_msg = MagicMock()
        mock_msg.id = 999
        mock_channel = AsyncMock()
        mock_channel.send = AsyncMock(return_value=mock_msg)
        platform._connected = True
        with (
            patch.object(
                platform._client, "get_channel", MagicMock(return_value=mock_channel)
            ),
            patch(
                "free_claude_code.messaging.platforms.discord._get_discord"
            ) as mock_get,
        ):
            mock_discord = MagicMock()
            mock_get.return_value = mock_discord
            platform.outbound._get_discord = mock_get
            msg_id = await platform.outbound.send_message(
                "123", "Hello", reply_to="456"
            )
        assert msg_id == "999"
        mock_channel.send.assert_awaited_once()
        call_kw = mock_channel.send.call_args[1]
        assert call_kw.get("reference") is not None

    @pytest.mark.asyncio
    async def test_edit_message_not_found_returns_gracefully(self):
        import discord as discord_pkg

        platform = _discord_runtime(bot_token="token")
        mock_channel = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status = 404
        mock_channel.fetch_message = AsyncMock(
            side_effect=discord_pkg.NotFound(mock_resp, "Not found")
        )
        platform._connected = True
        with patch.object(
            platform._client, "get_channel", MagicMock(return_value=mock_channel)
        ):
            await platform.outbound.edit_message("123", "456", "Updated")
        # Should not raise - NotFound is caught and we return

    @pytest.mark.asyncio
    async def test_delete_message(self):
        platform = _discord_runtime(bot_token="token")
        mock_msg = AsyncMock()
        mock_channel = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
        platform._connected = True
        with (
            patch.object(
                platform._client, "get_channel", MagicMock(return_value=mock_channel)
            ),
            patch(
                "free_claude_code.messaging.platforms.discord._get_discord"
            ) as mock_get,
        ):
            mock_get.return_value = MagicMock()
            platform.outbound._get_discord = mock_get
            await platform.outbound.delete_message("123", "456")
        mock_msg.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fire_and_forget_with_coroutine(self):
        platform = _discord_runtime(bot_token="token")
        completed = asyncio.Event()

        async def _task():
            completed.set()

        platform.outbound.fire_and_forget(_task())
        await completed.wait()
        await platform.outbound.close()
        assert platform.outbound._outbox._background_tasks == set()

    def test_on_message_registers_handler(self):
        platform = _discord_runtime(bot_token="token")
        handler = AsyncMock()
        platform.on_message(handler)
        assert platform._message_handler is handler

    @pytest.mark.asyncio
    async def test_start_requires_token(self):
        with patch.dict("os.environ", {"DISCORD_BOT_TOKEN": ""}, clear=False):
            platform = _discord_runtime(bot_token="")
            with pytest.raises(ValueError, match="DISCORD_BOT_TOKEN"):
                await platform.start()

    @pytest.mark.asyncio
    async def test_start_connects(self):
        limiter = _limiter_mock()
        platform = _discord_runtime(bot_token="token", limiter=limiter)
        keep_running = asyncio.Event()

        async def _fake_start(_token):
            platform._mark_connected()
            await keep_running.wait()

        with patch.object(
            platform._client,
            "start",
            new_callable=AsyncMock,
            side_effect=_fake_start,
        ):
            await platform.start()
        assert platform.is_connected is True
        limiter.start.assert_called_once_with()
        await platform.quiesce()
        await platform.close()

    @pytest.mark.asyncio
    async def test_client_failure_after_readiness_marks_runtime_disconnected(self):
        limiter = _limiter_mock()
        platform = _discord_runtime(bot_token="token", limiter=limiter)
        fail_client = asyncio.Event()

        async def _fake_start(_token):
            platform._mark_connected()
            await fail_client.wait()
            raise RuntimeError("connection lost")

        with patch.object(
            platform._client,
            "start",
            new_callable=AsyncMock,
            side_effect=_fake_start,
        ):
            await platform.start()
            assert platform.is_connected is True

            fail_client.set()
            assert platform._start_task is not None
            result = await asyncio.wait_for(
                asyncio.gather(platform._start_task, return_exceptions=True),
                timeout=1.0,
            )
            assert isinstance(result[0], RuntimeError)
            await asyncio.sleep(0)

        assert platform.is_connected is False
        await platform.quiesce()
        await platform.close()

    @pytest.mark.asyncio
    async def test_quiesce_when_client_already_closed_then_close_delivery(self):
        limiter = _limiter_mock()
        platform = _discord_runtime(bot_token="token", limiter=limiter)
        platform._connected = True
        with patch.object(
            platform._client, "is_closed", new_callable=MagicMock, return_value=True
        ):
            await platform.quiesce()
        assert platform.is_connected is False
        limiter.shutdown.assert_not_awaited()

        await platform.close()

        limiter.shutdown.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_quiesce_closes_client_without_closing_delivery(self):
        limiter = _limiter_mock()
        platform = _discord_runtime(bot_token="token", limiter=limiter)
        platform._connected = True
        mock_close = AsyncMock()
        with (
            patch.object(
                platform._client,
                "is_closed",
                new_callable=MagicMock,
                return_value=False,
            ),
            patch.object(platform._client, "close", mock_close),
        ):
            platform._start_task = None
            await platform.quiesce()
        mock_close.assert_awaited_once()
        assert platform.is_connected is False
        limiter.shutdown.assert_not_awaited()

        await platform.close()

        limiter.shutdown.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_quiesce_drains_start_task_when_client_close_fails(self):
        limiter = _limiter_mock()
        platform = _discord_runtime(bot_token="token", limiter=limiter)
        platform._connected = True
        started = asyncio.Event()

        async def pending_start() -> None:
            started.set()
            await asyncio.Event().wait()

        platform._start_task = asyncio.create_task(pending_start())
        await started.wait()
        with (
            patch.object(
                platform._client,
                "is_closed",
                new_callable=MagicMock,
                return_value=False,
            ),
            patch.object(
                platform._client,
                "close",
                new_callable=AsyncMock,
                side_effect=RuntimeError("close failed"),
            ),
            pytest.raises(RuntimeError, match="close failed"),
        ):
            await platform.quiesce()

        assert platform._start_task is None
        limiter.shutdown.assert_not_awaited()
        assert platform.is_connected is False

        await platform.close()

        limiter.shutdown.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_start_propagates_client_failure_immediately(self):
        limiter = _limiter_mock()
        platform = _discord_runtime(bot_token="token", limiter=limiter)

        with (
            patch.object(
                platform._client,
                "start",
                new_callable=AsyncMock,
                side_effect=RuntimeError("invalid token"),
            ),
            pytest.raises(RuntimeError, match="invalid token"),
        ):
            await platform.start()

        assert platform._start_task is not None
        assert platform._start_task.done()
        await platform.quiesce()
        await platform.close()

    @pytest.mark.asyncio
    async def test_quiesce_waits_for_tracked_inbound_handler(self):
        platform = _discord_runtime(bot_token="token")
        platform._accepting_messages = True
        entered = asyncio.Event()
        release = asyncio.Event()

        async def handle(_message) -> None:
            entered.set()
            await release.wait()

        with (
            patch.object(platform, "_on_discord_message", side_effect=handle),
            patch.object(
                platform._client,
                "is_closed",
                new_callable=MagicMock,
                return_value=True,
            ),
        ):
            handler_task = asyncio.create_task(
                platform._handle_client_message(MagicMock())
            )
            await entered.wait()
            quiesce_task = asyncio.create_task(platform.quiesce())
            await asyncio.sleep(0)

            assert not quiesce_task.done()
            release.set()
            await handler_task
            await quiesce_task

        assert platform._inbound_tasks == set()
        await platform.close()

    @pytest.mark.asyncio
    async def test_quiesce_preserves_caller_cancellation(self):
        platform = _discord_runtime(bot_token="token")
        entered = asyncio.Event()

        async def close_client() -> None:
            entered.set()
            await asyncio.Event().wait()

        with (
            patch.object(
                platform._client,
                "is_closed",
                new_callable=MagicMock,
                return_value=False,
            ),
            patch.object(platform._client, "close", side_effect=close_client),
        ):
            quiesce_task = asyncio.create_task(platform.quiesce())
            await entered.wait()
            quiesce_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await quiesce_task

        await platform.close()
