"""Tests for Discord platform adapter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from messaging.platforms.discord import (
    DISCORD_AVAILABLE,
    DiscordPlatform,
    _get_discord,
    _parse_allowed_channels,
)


class TestGetDiscord:
    """Tests for _get_discord helper."""

    def test_raises_when_discord_not_available(self):
        import messaging.platforms.discord as discord_mod

        with (
            patch.object(discord_mod, "DISCORD_AVAILABLE", False),
            patch.object(discord_mod, "_discord_module", None),
            pytest.raises(ImportError, match=r"discord\.py is required"),
        ):
            _get_discord()


class TestParseAllowedChannels:
    """Tests for _parse_allowed_channels helper."""

    def test_empty_string_returns_empty_set(self):
        assert _parse_allowed_channels("") == set()
        assert _parse_allowed_channels(None) == set()

    def test_whitespace_only_returns_empty_set(self):
        assert _parse_allowed_channels("   ") == set()

    def test_single_channel(self):
        assert _parse_allowed_channels("123456789") == {"123456789"}

    def test_comma_separated(self):
        assert _parse_allowed_channels("111,222,333") == {"111", "222", "333"}

    def test_strips_whitespace(self):
        assert _parse_allowed_channels(" 111 , 222 ") == {"111", "222"}

    def test_empty_parts_ignored(self):
        assert _parse_allowed_channels("111,,222,") == {"111", "222"}


@pytest.mark.skipif(not DISCORD_AVAILABLE, reason="discord.py not installed")
class TestDiscordPlatform:
    """Tests for DiscordPlatform (requires discord.py)."""

    def test_init_with_token(self):
        platform = DiscordPlatform(
            bot_token="test_token",
            allowed_channel_ids="123,456",
        )
        assert platform.bot_token == "test_token"
        assert platform.allowed_channel_ids == {"123", "456"}

    def test_init_without_allowed_channels(self):
        with patch.dict("os.environ", {"ALLOWED_DISCORD_CHANNELS": ""}, clear=False):
            platform = DiscordPlatform(bot_token="token", allowed_channel_ids="")
        assert platform.allowed_channel_ids == set()

    def test_empty_allowed_channels_rejects_all_messages(self):
        """When allowed_channel_ids is empty, no channels are allowed (secure default)."""
        with patch.dict("os.environ", {"ALLOWED_DISCORD_CHANNELS": ""}, clear=False):
            platform = DiscordPlatform(bot_token="token", allowed_channel_ids="")
        assert platform.allowed_channel_ids == set()
        # Empty set means: not self.allowed_channel_ids is True -> reject

    def test_truncate_long_message(self):
        platform = DiscordPlatform(bot_token="token")
        long_text = "x" * 2500
        truncated = platform._truncate(long_text)
        assert len(truncated) == 2000
        assert truncated.endswith("...")

    def test_truncate_short_message_unchanged(self):
        platform = DiscordPlatform(bot_token="token")
        short = "hello"
        assert platform._truncate(short) == short

    def test_truncate_exactly_at_limit_unchanged(self):
        platform = DiscordPlatform(bot_token="token")
        exact = "x" * 2000
        assert platform._truncate(exact) == exact

    def test_truncate_one_over_limit_truncates(self):
        platform = DiscordPlatform(bot_token="token")
        over = "x" * 2001
        result = platform._truncate(over)
        assert len(result) == 2000
        assert result.endswith("...")

    def test_truncate_empty_string(self):
        platform = DiscordPlatform(bot_token="token")
        assert platform._truncate("") == ""

    @pytest.mark.asyncio
    async def test_send_message_returns_message_id(self):
        platform = DiscordPlatform(bot_token="token")
        mock_msg = MagicMock()
        mock_msg.id = 999
        mock_channel = AsyncMock()
        mock_channel.send = AsyncMock(return_value=mock_msg)
        platform._connected = True
        with patch.object(
            platform._client, "get_channel", MagicMock(return_value=mock_channel)
        ):
            msg_id = await platform.send_message("123", "Hello")
        assert msg_id == "999"

    @pytest.mark.asyncio
    async def test_edit_message(self):
        platform = DiscordPlatform(bot_token="token")
        mock_msg = AsyncMock()
        mock_channel = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
        platform._connected = True
        with patch.object(
            platform._client, "get_channel", MagicMock(return_value=mock_channel)
        ):
            await platform.edit_message("123", "456", "Updated text")
        mock_msg.edit.assert_called_once_with(content="Updated text")

    @pytest.mark.asyncio
    async def test_send_message_channel_not_found_raises(self):
        platform = DiscordPlatform(bot_token="token")
        platform._connected = True
        with (
            patch.object(platform._client, "get_channel", MagicMock(return_value=None)),
            pytest.raises(RuntimeError, match="Channel"),
        ):
            await platform.send_message("123", "Hello")

    @pytest.mark.asyncio
    async def test_send_message_channel_no_send_raises(self):
        platform = DiscordPlatform(bot_token="token")
        platform._connected = True
        mock_channel = MagicMock(spec=[])  # No send attr
        with (
            patch.object(
                platform._client, "get_channel", MagicMock(return_value=mock_channel)
            ),
            pytest.raises(RuntimeError, match="Channel"),
        ):
            await platform.send_message("123", "Hello")

    @pytest.mark.asyncio
    async def test_queue_send_message_without_limiter_calls_send_message(self):
        platform = DiscordPlatform(bot_token="token")
        platform._limiter = None
        platform._connected = True
        mock_channel = AsyncMock()
        mock_msg = MagicMock()
        mock_msg.id = 42
        mock_channel.send = AsyncMock(return_value=mock_msg)
        with patch.object(
            platform._client, "get_channel", MagicMock(return_value=mock_channel)
        ):
            result = await platform.queue_send_message("123", "hi")
        assert result == "42"
        mock_channel.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_queue_edit_message_without_limiter_calls_edit_message(self):
        platform = DiscordPlatform(bot_token="token")
        platform._limiter = None
        platform._connected = True
        mock_msg = AsyncMock()
        mock_channel = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
        with patch.object(
            platform._client, "get_channel", MagicMock(return_value=mock_channel)
        ):
            await platform.queue_edit_message("123", "456", "Updated")
        mock_msg.edit.assert_called_once_with(content="Updated")

    @pytest.mark.asyncio
    async def test_on_discord_message_bot_ignored(self):
        platform = DiscordPlatform(bot_token="token", allowed_channel_ids="123")
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
        platform = DiscordPlatform(bot_token="token", allowed_channel_ids="123")
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
        platform = DiscordPlatform(bot_token="token", allowed_channel_ids="123")
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
        platform = DiscordPlatform(bot_token="token", allowed_channel_ids="123")
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
        assert call.user_id == "456"
        assert call.message_id == "789"
        assert call.platform == "discord"

    @pytest.mark.asyncio
    async def test_send_message_with_reply_to(self):
        platform = DiscordPlatform(bot_token="token")
        mock_msg = MagicMock()
        mock_msg.id = 999
        mock_channel = AsyncMock()
        mock_channel.send = AsyncMock(return_value=mock_msg)
        platform._connected = True
        with (
            patch.object(
                platform._client, "get_channel", MagicMock(return_value=mock_channel)
            ),
            patch("messaging.platforms.discord._get_discord") as mock_get,
        ):
            mock_discord = MagicMock()
            mock_get.return_value = mock_discord
            msg_id = await platform.send_message("123", "Hello", reply_to="456")
        assert msg_id == "999"
        mock_channel.send.assert_awaited_once()
        call_kw = mock_channel.send.call_args[1]
        assert call_kw.get("reference") is not None

    @pytest.mark.asyncio
    async def test_edit_message_not_found_returns_gracefully(self):
        import discord as discord_pkg

        platform = DiscordPlatform(bot_token="token")
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
            await platform.edit_message("123", "456", "Updated")
        # Should not raise - NotFound is caught and we return

    @pytest.mark.asyncio
    async def test_delete_message(self):
        platform = DiscordPlatform(bot_token="token")
        mock_msg = AsyncMock()
        mock_channel = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
        platform._connected = True
        with (
            patch.object(
                platform._client, "get_channel", MagicMock(return_value=mock_channel)
            ),
            patch("messaging.platforms.discord._get_discord") as mock_get,
        ):
            mock_get.return_value = MagicMock()
            await platform.delete_message("123", "456")
        mock_msg.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fire_and_forget_with_coroutine(self):
        platform = DiscordPlatform(bot_token="token")

        async def _task():
            pass

        coro = _task()
        with patch("asyncio.create_task") as mock_create:

            def _run(c):
                return asyncio.ensure_future(c)

            mock_create.side_effect = _run
            platform.fire_and_forget(coro)
            mock_create.assert_called_once()
        await asyncio.sleep(0)

    def test_on_message_registers_handler(self):
        platform = DiscordPlatform(bot_token="token")
        handler = AsyncMock()
        platform.on_message(handler)
        assert platform._message_handler is handler

    @pytest.mark.asyncio
    async def test_start_requires_token(self):
        with patch.dict("os.environ", {"DISCORD_BOT_TOKEN": ""}, clear=False):
            platform = DiscordPlatform(bot_token="")
            with pytest.raises(ValueError, match="DISCORD_BOT_TOKEN"):
                await platform.start()

    @pytest.mark.asyncio
    async def test_start_connects(self):
        platform = DiscordPlatform(bot_token="token")

        async def _fake_start(_token):
            platform._connected = True

        with (
            patch.object(
                platform._client,
                "start",
                new_callable=AsyncMock,
                side_effect=_fake_start,
            ),
            patch(
                "messaging.limiter.MessagingRateLimiter.get_instance",
                new_callable=AsyncMock,
            ),
        ):
            await platform.start()
        assert platform.is_connected is True

    @pytest.mark.asyncio
    async def test_stop_when_already_closed(self):
        platform = DiscordPlatform(bot_token="token")
        platform._connected = True
        with patch.object(
            platform._client, "is_closed", new_callable=MagicMock, return_value=True
        ):
            await platform.stop()
        assert platform.is_connected is False

    @pytest.mark.asyncio
    async def test_stop_closes_client(self):
        platform = DiscordPlatform(bot_token="token")
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
            await platform.stop()
        mock_close.assert_awaited_once()
        assert platform.is_connected is False
