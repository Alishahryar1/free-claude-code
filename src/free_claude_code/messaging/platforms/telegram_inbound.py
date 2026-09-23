"""Telegram inbound event normalization."""

from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes

from ..models import IncomingMessage
from ..rendering.telegram_markdown import format_status
from .voice_flow import VoiceNoteRequest, audio_suffix_from_metadata


def telegram_text_message_from_update(
    update: Update,
    *,
    allowed_user_id: str | None,
    log_raw_messaging_content: bool,
) -> IncomingMessage | None:
    """Normalize a Telegram text update into an incoming message."""
    if (
        not update.message
        or not update.message.text
        or not update.effective_user
        or not update.effective_chat
    ):
        return None

    user_id = str(update.effective_user.id)
    chat_id = str(update.effective_chat.id)
    if allowed_user_id and user_id != str(allowed_user_id).strip():
        logger.warning("Unauthorized access attempt from {}", user_id)
        return None

    message = update.message
    message_id = str(message.message_id)
    reply_to = (
        str(message.reply_to_message.message_id) if message.reply_to_message else None
    )
    thread_id = (
        str(message.message_thread_id)
        if getattr(message, "message_thread_id", None) is not None
        else None
    )
    raw_text = message.text or ""
    if log_raw_messaging_content:
        text_preview = raw_text[:80]
        if len(raw_text) > 80:
            text_preview += "..."
        logger.info(
            "TELEGRAM_MSG: chat_id={} message_id={} reply_to={} text_preview={!r}",
            chat_id,
            message_id,
            reply_to,
            text_preview,
        )
    else:
        logger.info(
            "TELEGRAM_MSG: chat_id={} message_id={} reply_to={} text_len={}",
            chat_id,
            message_id,
            reply_to,
            len(raw_text),
        )

    return IncomingMessage(
        text=raw_text,
        chat_id=chat_id,
        user_id=user_id,
        message_id=message_id,
        platform="telegram",
        reply_to_message_id=reply_to,
        message_thread_id=thread_id,
        raw_event=update,
    )


def telegram_voice_request_from_update(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    allowed_user_id: str | None,
) -> VoiceNoteRequest | None:
    """Normalize a Telegram voice update into a voice-note request."""
    message = update.message
    effective_user = update.effective_user
    effective_chat = update.effective_chat
    if (
        message is None
        or message.voice is None
        or effective_user is None
        or effective_chat is None
    ):
        return None

    user_id = str(effective_user.id)
    if allowed_user_id and user_id != str(allowed_user_id).strip():
        logger.warning("Unauthorized voice access attempt from {}", user_id)
        return None

    voice = message.voice
    chat_id = str(effective_chat.id)
    message_id = str(message.message_id)
    thread_id = (
        str(message.message_thread_id)
        if getattr(message, "message_thread_id", None) is not None
        else None
    )
    reply_to = (
        str(message.reply_to_message.message_id) if message.reply_to_message else None
    )

    async def _download_to(tmp_path) -> None:
        tg_file = await context.bot.get_file(voice.file_id)
        await tg_file.download_to_drive(custom_path=str(tmp_path))

    async def _reply_text(text: str) -> None:
        await message.reply_text(text)

    return VoiceNoteRequest(
        platform="telegram",
        chat_id=chat_id,
        user_id=user_id,
        message_id=message_id,
        raw_event=update,
        content_type=voice.mime_type or "audio/ogg",
        temp_suffix=audio_suffix_from_metadata(content_type=voice.mime_type),
        status_text=format_status("⏳", "Transcribing voice note..."),
        status_parse_mode="MarkdownV2",
        message_thread_id=thread_id,
        reply_to_message_id=reply_to,
        username=None,
        download_to=_download_to,
        reply_text=_reply_text,
    )
