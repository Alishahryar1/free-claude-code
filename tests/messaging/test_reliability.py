from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import NetworkError, RetryAfter, TelegramError

from messaging.platforms.telegram import TelegramPlatform


@pytest.fixture
def telegram_platform():
    with patch("messaging.platforms.telegram.TELEGRAM_AVAILABLE", True):
        platform = TelegramPlatform(bot_token="test_token", allowed_user_id="12345")
        return platform


@pytest.mark.asyncio
async def test_telegram_retry_on_network_error(telegram_platform):
    mock_bot = AsyncMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 999

    # Fail twice, then succeed
    mock_bot.send_message.side_effect = [
        NetworkError("Connection failed"),
        NetworkError("Connection failed"),
        mock_msg,
    ]

    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    # We need to patch asyncio.sleep to speed up the test
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        msg_id = await telegram_platform.send_message("chat_1", "hello")

        assert msg_id == "999"
        assert mock_bot.send_message.call_count == 3
        assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_telegram_retry_on_retry_after(telegram_platform):
    mock_bot = AsyncMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 1000

    # Fail with RetryAfter, then succeed
    mock_bot.send_message.side_effect = [RetryAfter(retry_after=5), mock_msg]

    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        msg_id = await telegram_platform.send_message("chat_1", "hello")

        assert msg_id == "1000"
        assert mock_bot.send_message.call_count == 2
        mock_sleep.assert_called_with(5)


@pytest.mark.asyncio
async def test_telegram_no_retry_on_bad_request(telegram_platform):
    mock_bot = AsyncMock()

    # Fail with generic TelegramError (should not retry unless specific conditions met)
    mock_bot.send_message.side_effect = TelegramError("Bad Request: some error")

    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    with pytest.raises(TelegramError):
        await telegram_platform.send_message("chat_1", "hello")

    assert mock_bot.send_message.call_count == 1


def test_handler_build_message_hardening():
    # Formatting hardening now lives in TranscriptBuffer rendering.
    from messaging.rendering.telegram_markdown import (
        escape_md_v2,
        escape_md_v2_code,
        mdv2_bold,
        mdv2_code_inline,
        render_markdown_to_mdv2,
    )
    from messaging.transcript import RenderCtx, TranscriptBuffer

    ctx = RenderCtx(
        bold=mdv2_bold,
        code_inline=mdv2_code_inline,
        escape_code=escape_md_v2_code,
        escape_text=escape_md_v2,
        render_markdown=render_markdown_to_mdv2,
    )

    # Case 1: Empty transcript + no status => empty string.
    t = TranscriptBuffer()
    msg = t.render(ctx, limit_chars=3900, status=None)
    assert msg == ""

    # Case 2: Truncation with code block closing and status preserved.
    t.apply({"type": "thinking_chunk", "text": ("thought " * 200)})
    t.apply({"type": "text_chunk", "text": ("This is a very long message. " * 300)})

    msg = t.render(ctx, limit_chars=3900, status="Finishing...")

    assert len(msg) <= 4096
    assert "Finishing..." in msg
    if "```" in msg:
        assert msg.count("```") % 2 == 0


def test_render_output_never_exceeds_4096():
    """Transcript render with various status lengths never exceeds Telegram 4096 limit."""
    from messaging.rendering.telegram_markdown import (
        escape_md_v2,
        escape_md_v2_code,
        mdv2_bold,
        mdv2_code_inline,
        render_markdown_to_mdv2,
    )
    from messaging.transcript import RenderCtx, TranscriptBuffer

    ctx = RenderCtx(
        bold=mdv2_bold,
        code_inline=mdv2_code_inline,
        escape_code=escape_md_v2_code,
        escape_text=escape_md_v2,
        render_markdown=render_markdown_to_mdv2,
    )

    t = TranscriptBuffer()
    t.apply({"type": "thinking_chunk", "text": "x" * 500})
    t.apply({"type": "text_chunk", "text": "y" * 3500})

    for status in [None, "Done", "✅ *Complete*", "A" * 100]:
        msg = t.render(ctx, limit_chars=3900, status=status)
        assert len(msg) <= 4096, f"status={status!r} produced len={len(msg)}"
