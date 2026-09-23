"""Tests for voice note handling in Telegram and Discord platforms."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.messaging.platforms.discord import (
    DISCORD_AVAILABLE,
    DiscordRuntime,
)
from free_claude_code.messaging.platforms.discord_inbound import get_audio_attachment
from free_claude_code.messaging.platforms.telegram import TelegramRuntime


@pytest.fixture
def telegram_platform():
    transcriber = MagicMock()
    transcriber.transcribe = AsyncMock(return_value="Hello from voice")
    transcriber.close = AsyncMock()
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        platform = TelegramRuntime(
            bot_token="test_token",
            allowed_user_id="12345",
            limiter=MagicMock(),
            transcriber=transcriber,
        )
    return platform, transcriber


@pytest.mark.asyncio
async def test_telegram_voice_disabled_sends_reply():
    """When voice_note_enabled is False, reply with disabled message."""
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        telegram_platform = TelegramRuntime(
            bot_token="test_token",
            allowed_user_id="12345",
            limiter=MagicMock(),
            transcriber=None,
        )
    mock_update = MagicMock()
    mock_update.message.voice = MagicMock(file_id="f1", mime_type="audio/ogg")
    mock_update.effective_user.id = 12345
    mock_update.effective_chat.id = 6789
    mock_update.message.reply_text = AsyncMock()

    await telegram_platform._on_telegram_voice(mock_update, MagicMock())

    mock_update.message.reply_text.assert_called_once_with("Voice notes are disabled.")


@pytest.mark.asyncio
async def test_telegram_voice_unauthorized_ignored(telegram_platform):
    """Voice from unauthorized user is ignored (no reply)."""
    platform, transcriber = telegram_platform
    mock_update = MagicMock()
    mock_update.message.voice = MagicMock(file_id="f1", mime_type="audio/ogg")
    mock_update.effective_user.id = 99999  # Not 12345
    mock_update.message.reply_text = AsyncMock()

    await platform._on_telegram_voice(mock_update, MagicMock())

    mock_update.message.reply_text.assert_not_called()
    transcriber.transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_voice_success_invokes_handler(telegram_platform):
    """Successful transcription invokes message handler with transcribed text."""
    platform, transcriber = telegram_platform
    handler = AsyncMock()
    platform.on_message(handler)

    mock_update = MagicMock()
    mock_voice = MagicMock(file_id="f1", mime_type="audio/ogg")
    mock_update.message.voice = mock_voice
    mock_update.message.message_id = 42
    mock_update.message.reply_to_message = None
    mock_update.effective_user.id = 12345
    mock_update.effective_chat.id = 6789
    mock_update.message.reply_text = AsyncMock()

    mock_file = AsyncMock()
    mock_context = MagicMock()
    mock_context.bot.get_file = AsyncMock(return_value=mock_file)

    async def fake_download(custom_path=None):
        if custom_path:
            Path(custom_path).write_bytes(b"fake ogg")

    mock_file.download_to_drive = fake_download

    mock_queue_send = AsyncMock(return_value="999")
    with patch.object(
        platform.outbound,
        "queue_send_message",
        mock_queue_send,
    ):
        await platform._on_telegram_voice(mock_update, mock_context)

    mock_queue_send.assert_called_once()
    call_args, call_kw = mock_queue_send.call_args
    assert "Transcribing voice note" in call_args[1]
    assert call_kw["reply_to"] == "42"
    assert call_kw["fire_and_forget"] is False
    transcriber.transcribe.assert_awaited_once()

    handler.assert_called_once()
    incoming = handler.call_args[0][0]
    assert incoming.text == "Hello from voice"
    assert incoming.chat_id == "6789"
    assert incoming.platform == "telegram"
    assert incoming.status_message_id == "999"


@pytest.mark.asyncio
async def test_telegram_bulk_voice_cancellation_delegates(telegram_platform) -> None:
    platform, _transcriber = telegram_platform
    platform._voice_flow.cancel_all_pending_voices = AsyncMock(return_value=())

    assert await platform.cancel_all_pending_voices() == ()

    platform._voice_flow.cancel_all_pending_voices.assert_awaited_once()


@pytest.mark.skipif(not DISCORD_AVAILABLE, reason="discord.py not installed")
class TestDiscordGetAudioAttachment:
    """Tests for _get_audio_attachment helper."""

    def test_returns_none_when_no_attachments(self):
        msg = MagicMock()
        msg.attachments = []
        assert get_audio_attachment(msg) is None

    def test_returns_none_when_no_audio_attachments(self):
        msg = MagicMock()
        att = MagicMock()
        att.content_type = "image/png"
        att.filename = "pic.png"
        msg.attachments = [att]
        assert get_audio_attachment(msg) is None

    def test_returns_attachment_by_content_type(self):
        msg = MagicMock()
        att = MagicMock()
        att.content_type = "audio/ogg"
        att.filename = "voice.ogg"
        msg.attachments = [att]
        assert get_audio_attachment(msg) is att

    def test_returns_attachment_by_extension(self):
        msg = MagicMock()
        att = MagicMock()
        att.content_type = "application/octet-stream"
        att.filename = "voice.ogg"
        msg.attachments = [att]
        assert get_audio_attachment(msg) is att


@pytest.mark.skipif(not DISCORD_AVAILABLE, reason="discord.py not installed")
@pytest.mark.asyncio
async def test_discord_voice_disabled_sends_reply():
    """When voice_note_enabled is False, reply with disabled message."""
    platform = DiscordRuntime(
        bot_token="token",
        allowed_channel_ids="123",
        limiter=MagicMock(),
        transcriber=None,
    )
    platform._message_handler = None

    mock_message = MagicMock()
    mock_message.author.bot = False
    mock_message.content = None
    mock_message.channel.id = 123
    mock_message.reply = AsyncMock()

    mock_att = MagicMock()
    mock_att.content_type = "audio/ogg"
    mock_att.filename = "voice.ogg"
    mock_message.attachments = [mock_att]

    await platform._on_discord_message(mock_message)

    mock_message.reply.assert_called_once_with("Voice notes are disabled.")


@pytest.mark.skipif(not DISCORD_AVAILABLE, reason="discord.py not installed")
@pytest.mark.asyncio
async def test_discord_bulk_voice_cancellation_delegates() -> None:
    platform = DiscordRuntime(
        bot_token="token",
        allowed_channel_ids="123",
        limiter=MagicMock(),
        transcriber=None,
    )
    platform._voice_flow.cancel_all_pending_voices = AsyncMock(return_value=())

    assert await platform.cancel_all_pending_voices() == ()

    platform._voice_flow.cancel_all_pending_voices.assert_awaited_once()
