"""Discord inbound event normalization."""

from typing import Any

from loguru import logger

from ..models import IncomingMessage
from ..rendering.discord_markdown import format_status_discord
from .voice_flow import (
    VoiceNoteRequest,
    audio_suffix_from_metadata,
    is_audio_metadata,
)


def parse_allowed_channels(raw: str | None) -> set[str]:
    """Parse comma-separated Discord channel IDs."""
    if not raw or not raw.strip():
        return set()
    return {s.strip() for s in raw.split(",") if s.strip()}


def get_audio_attachment(message: Any) -> Any | None:
    """Return the first audio attachment from a Discord message."""
    for att in message.attachments:
        if is_audio_metadata(att.filename, att.content_type):
            return att
    return None


def discord_text_message_from_event(
    message: Any,
    *,
    log_raw_messaging_content: bool,
) -> IncomingMessage:
    """Normalize a Discord message into an incoming text message."""
    channel_id = str(message.channel.id)
    message_id = str(message.id)
    reply_to = (
        str(message.reference.message_id)
        if message.reference and message.reference.message_id
        else None
    )
    raw_content = message.content or ""
    if log_raw_messaging_content:
        text_preview = raw_content[:80]
        if len(raw_content) > 80:
            text_preview += "..."
        logger.info(
            "DISCORD_MSG: chat_id={} message_id={} reply_to={} text_preview={!r}",
            channel_id,
            message_id,
            reply_to,
            text_preview,
        )
    else:
        logger.info(
            "DISCORD_MSG: chat_id={} message_id={} reply_to={} text_len={}",
            channel_id,
            message_id,
            reply_to,
            len(raw_content),
        )

    return IncomingMessage(
        text=message.content,
        chat_id=channel_id,
        user_id=str(message.author.id),
        message_id=message_id,
        platform="discord",
        reply_to_message_id=reply_to,
        username=message.author.display_name,
        raw_event=message,
    )


def discord_voice_request_from_event(
    message: Any,
    attachment: Any,
    channel_id: str,
) -> VoiceNoteRequest:
    """Normalize a Discord voice/audio attachment into a voice-note request."""
    message_id = str(message.id)
    reply_to = (
        str(message.reference.message_id)
        if message.reference and message.reference.message_id
        else None
    )

    async def _download_to(tmp_path) -> None:
        await attachment.save(str(tmp_path))

    async def _reply_text(text: str) -> None:
        await message.reply(text)

    return VoiceNoteRequest(
        platform="discord",
        chat_id=channel_id,
        user_id=str(message.author.id),
        message_id=message_id,
        raw_event=message,
        content_type=attachment.content_type or "audio/ogg",
        temp_suffix=audio_suffix_from_metadata(
            filename=attachment.filename,
            content_type=attachment.content_type,
        ),
        status_text=format_status_discord("Transcribing voice note..."),
        reply_to_message_id=reply_to,
        username=message.author.display_name,
        download_to=_download_to,
        reply_text=_reply_text,
    )
