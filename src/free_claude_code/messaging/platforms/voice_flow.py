"""Shared voice-note flow for messaging platform adapters."""

import asyncio
import contextlib
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from free_claude_code.core.diagnostics import format_user_error_preview

from ..models import IncomingMessage, MessageScope
from ..voice import (
    PendingVoiceClaim,
    PendingVoiceRegistry,
    Transcriber,
    VoiceCancellationResult,
    VoiceHandoffOutcome,
)

AUDIO_EXTENSIONS = (".ogg", ".mp4", ".mp3", ".wav", ".m4a")
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024
VOICE_DISABLED_MESSAGE = "Voice notes are disabled."
VOICE_TRANSCRIPTION_ERROR_MESSAGE = (
    "Could not transcribe voice note. Please try again or send text."
)

MessageHandler = Callable[[IncomingMessage], Awaitable[None]]
QueueSend = Callable[..., Awaitable[str | None]]
QueueDeleteMany = Callable[..., Awaitable[None]]


@dataclass(frozen=True)
class VoiceNoteRequest:
    """Platform-normalized voice-note input."""

    platform: str
    chat_id: str
    user_id: str
    message_id: str
    raw_event: Any
    content_type: str
    temp_suffix: str
    status_text: str
    download_to: Callable[[Path], Awaitable[None]]
    reply_text: Callable[[str], Awaitable[None]]
    reply_to_message_id: str | None = None
    status_parse_mode: str | None = None
    message_thread_id: str | None = None
    username: str | None = None

    @property
    def scope(self) -> MessageScope:
        return MessageScope(platform=self.platform, chat_id=self.chat_id)


def is_audio_metadata(filename: str | None, content_type: str | None) -> bool:
    """Return whether attachment metadata describes an audio file."""
    normalized_content_type = (content_type or "").lower()
    normalized_filename = (filename or "").lower()
    return normalized_content_type.startswith("audio/") or any(
        normalized_filename.endswith(extension) for extension in AUDIO_EXTENSIONS
    )


def audio_suffix_from_metadata(
    *,
    filename: str | None = None,
    content_type: str | None = None,
    default: str = ".ogg",
) -> str:
    """Choose a temp-file suffix from platform attachment metadata."""
    normalized_filename = (filename or "").lower()
    normalized_content_type = (content_type or "").lower()

    if "m4a" in normalized_content_type:
        return ".m4a"
    if "mp4" in normalized_content_type:
        if normalized_filename.endswith(".m4a"):
            return ".m4a"
        return ".mp4"
    if "mpeg" in normalized_content_type or "mp3" in normalized_content_type:
        return ".mp3"
    if "wav" in normalized_content_type:
        return ".wav"

    for extension in AUDIO_EXTENSIONS:
        if normalized_filename.endswith(extension):
            return extension
    return default


class VoiceNoteFlow:
    """Own common voice transcription state and control flow."""

    def __init__(
        self,
        *,
        transcriber: Transcriber | None,
        log_raw_messaging_content: bool,
        log_api_error_tracebacks: bool,
    ) -> None:
        self._transcriber = transcriber
        self._log_raw_messaging_content = log_raw_messaging_content
        self._log_api_error_tracebacks = log_api_error_tracebacks
        self._pending_voice = PendingVoiceRegistry()

    @property
    def is_enabled(self) -> bool:
        """Return whether voice-note handling is enabled."""
        return self._transcriber is not None

    async def reply_if_disabled(
        self, reply_text: Callable[[str], Awaitable[None]]
    ) -> bool:
        """Reply with the disabled message when voice-note handling is disabled."""
        if self.is_enabled:
            return False
        await reply_text(VOICE_DISABLED_MESSAGE)
        return True

    async def cancel_pending_voice(
        self, scope: MessageScope, reply_id: str
    ) -> VoiceCancellationResult | None:
        """Cancel a pending voice transcription."""
        return await self._pending_voice.cancel(scope, reply_id)

    async def cancel_all_pending_voices(
        self,
    ) -> tuple[VoiceCancellationResult, ...]:
        """Cancel every pending voice transcription and published handoff."""
        return await self._pending_voice.cancel_all()

    async def cancel_pending_voices_in_scope(
        self,
        scope: MessageScope,
    ) -> tuple[VoiceCancellationResult, ...]:
        """Cancel pending voice transcriptions belonging to one chat."""
        return await self._pending_voice.cancel_scope(scope)

    async def handle(
        self,
        request: VoiceNoteRequest,
        *,
        message_handler: MessageHandler | None,
        queue_send_message: QueueSend,
        queue_delete_messages: QueueDeleteMany,
    ) -> bool:
        """Transcribe a voice note and hand the resulting turn to messaging."""
        if await self.reply_if_disabled(request.reply_text):
            return True

        if message_handler is None:
            return False

        claim = await self._pending_voice.reserve(
            request.scope,
            request.message_id,
        )
        if claim is None:
            return True

        status_msg_id: str | None = None
        status_bound = False
        tmp_path: Path | None = None
        handed_off = False

        try:
            delivered_status_id = await queue_send_message(
                request.chat_id,
                request.status_text,
                reply_to=request.message_id,
                parse_mode=request.status_parse_mode,
                fire_and_forget=False,
                message_thread_id=request.message_thread_id,
            )
            if delivered_status_id is None:
                raise RuntimeError("Voice status delivery returned no message ID.")
            status_msg_id = str(delivered_status_id)
            status_bound = await self._pending_voice.bind_status(claim, status_msg_id)
            if not status_bound:
                _, cancellation = await self._finish_failed_pending_voice(
                    request,
                    claim,
                    status_msg_id,
                    queue_delete_messages,
                    status_bound=False,
                    handed_off=False,
                )
                if cancellation is not None:
                    raise cancellation from None
                return True

            with tempfile.NamedTemporaryFile(
                suffix=request.temp_suffix, delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)

            await request.download_to(tmp_path)
            _validate_audio_file(tmp_path)

            transcriber = self._transcriber
            if transcriber is None:
                raise RuntimeError("Voice transcription is not configured.")
            transcribed = await transcriber.transcribe(tmp_path)

            incoming = IncomingMessage(
                text=transcribed,
                chat_id=request.chat_id,
                user_id=request.user_id,
                message_id=request.message_id,
                platform=request.platform,
                reply_to_message_id=request.reply_to_message_id,
                message_thread_id=request.message_thread_id,
                username=request.username,
                raw_event=request.raw_event,
                status_message_id=status_msg_id,
            )
            self._log_transcription(request, transcribed)

            async def handle_incoming() -> None:
                nonlocal handed_off
                handed_off = True
                await message_handler(incoming)

            handoff_outcome = await self._pending_voice.handoff(
                claim,
                handle_incoming,
            )
            if handoff_outcome is VoiceHandoffOutcome.REJECTED:
                _, cancellation = await self._finish_failed_pending_voice(
                    request,
                    claim,
                    status_msg_id,
                    queue_delete_messages,
                    status_bound=status_bound,
                    handed_off=False,
                )
                if cancellation is not None:
                    raise cancellation from None
                return True
            return True
        except asyncio.CancelledError:
            await self._finish_failed_pending_voice(
                request,
                claim,
                status_msg_id,
                queue_delete_messages,
                status_bound=status_bound,
                handed_off=handed_off,
            )
            raise
        except (ValueError, ImportError) as e:
            discarded, cancellation = await self._finish_failed_pending_voice(
                request,
                claim,
                status_msg_id,
                queue_delete_messages,
                status_bound=status_bound,
                handed_off=handed_off,
            )
            if cancellation is not None:
                raise cancellation from None
            if not handed_off and not discarded:
                return True
            await request.reply_text(format_user_error_preview(e))
            return True
        except Exception as e:
            discarded, cancellation = await self._finish_failed_pending_voice(
                request,
                claim,
                status_msg_id,
                queue_delete_messages,
                status_bound=status_bound,
                handed_off=handed_off,
            )
            if cancellation is not None:
                raise cancellation from None
            if not handed_off and not discarded:
                return True
            if self._log_api_error_tracebacks:
                logger.error("Voice transcription failed: {}", e)
            else:
                logger.error(
                    "Voice transcription failed: exc_type={}",
                    type(e).__name__,
                )
            await request.reply_text(VOICE_TRANSCRIPTION_ERROR_MESSAGE)
            return True
        except BaseException:
            await self._finish_failed_pending_voice(
                request,
                claim,
                status_msg_id,
                queue_delete_messages,
                status_bound=status_bound,
                handed_off=handed_off,
            )
            raise
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    tmp_path.unlink(missing_ok=True)

    async def _finish_failed_pending_voice(
        self,
        request: VoiceNoteRequest,
        claim: PendingVoiceClaim,
        status_msg_id: str | None,
        queue_delete_messages: QueueDeleteMany,
        *,
        status_bound: bool,
        handed_off: bool,
    ) -> tuple[bool, asyncio.CancelledError | None]:
        """Finish owned cleanup before propagating any caller cancellation."""
        cleanup_task = asyncio.create_task(
            self._clear_failed_pending_voice(
                request,
                claim,
                status_msg_id,
                queue_delete_messages,
                status_bound=status_bound,
                handed_off=handed_off,
            ),
            name=f"voice-cleanup-{claim.claim_id}",
        )
        cancellation: asyncio.CancelledError | None = None
        current = asyncio.current_task()
        while True:
            cancelling_before = current.cancelling() if current is not None else 0
            try:
                return await asyncio.shield(cleanup_task), cancellation
            except asyncio.CancelledError as error:
                if current is None or current.cancelling() <= cancelling_before:
                    raise
                cancellation = cancellation or error

    async def _clear_failed_pending_voice(
        self,
        request: VoiceNoteRequest,
        claim: PendingVoiceClaim,
        status_msg_id: str | None,
        queue_delete_messages: QueueDeleteMany,
        *,
        status_bound: bool,
        handed_off: bool,
    ) -> bool:
        discarded = await self._pending_voice.discard(claim)
        if (
            not handed_off
            and status_msg_id is not None
            and (discarded or not status_bound)
        ):
            with contextlib.suppress(Exception):
                await queue_delete_messages(request.chat_id, [status_msg_id])
        return discarded

    def _log_transcription(self, request: VoiceNoteRequest, transcribed: str) -> None:
        label = request.platform.upper()
        if self._log_raw_messaging_content:
            logger.info(
                "{}_VOICE: chat_id={} message_id={} transcribed={!r}",
                label,
                request.chat_id,
                request.message_id,
                (transcribed[:80] + "..." if len(transcribed) > 80 else transcribed),
            )
        else:
            logger.info(
                "{}_VOICE: chat_id={} message_id={} transcribed_len={}",
                label,
                request.chat_id,
                request.message_id,
                len(transcribed),
            )


def _validate_audio_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    size = file_path.stat().st_size
    if size > MAX_AUDIO_SIZE_BYTES:
        raise ValueError(
            f"Audio file too large ({size} bytes). Max {MAX_AUDIO_SIZE_BYTES} bytes."
        )
