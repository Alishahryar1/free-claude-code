"""Telegram messaging runtime."""

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable

# Opt-in to future behavior for python-telegram-bot (retry_after as timedelta).
os.environ["PTB_TIMEDELTA"] = "1"

from loguru import logger

from free_claude_code.core.diagnostics import format_user_error_preview

from ..limiter import MessagingRateLimiter
from ..models import IncomingMessage, MessageScope
from ..rendering.telegram_markdown import escape_md_v2
from ..voice import Transcriber, VoiceCancellationResult
from .ports import InboundMessageHandler
from .telegram_inbound import (
    telegram_text_message_from_update,
    telegram_voice_request_from_update,
)
from .telegram_io import TelegramMessenger
from .voice_flow import VoiceNoteFlow

try:
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    from telegram.request import HTTPXRequest

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class TelegramRuntime:
    """Owns Telegram SDK lifecycle and inbound event handoff."""

    name = "telegram"

    def __init__(
        self,
        bot_token: str | None = None,
        allowed_user_id: str | None = None,
        *,
        telegram_proxy_url: str | None = None,
        limiter: MessagingRateLimiter,
        transcriber: Transcriber | None,
        log_raw_messaging_content: bool = False,
        log_api_error_tracebacks: bool = False,
    ) -> None:
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is required. Install with: pip install python-telegram-bot"
            )

        self.bot_token = bot_token
        self.allowed_user_id = allowed_user_id
        self.telegram_proxy_url = telegram_proxy_url
        if not self.bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN not set")

        self._application: Application | None = None
        self._message_handler: InboundMessageHandler | None = None
        self._connected = False
        self._limiter = limiter
        self.outbound = TelegramMessenger(
            get_application=lambda: self._application,
            limiter=limiter,
        )
        self._voice_flow = VoiceNoteFlow(
            transcriber=transcriber,
            log_raw_messaging_content=log_raw_messaging_content,
            log_api_error_tracebacks=log_api_error_tracebacks,
        )
        self._log_raw_messaging_content = log_raw_messaging_content
        self._log_api_error_tracebacks = log_api_error_tracebacks

    async def cancel_pending_voice(
        self, scope: MessageScope, reply_id: str
    ) -> VoiceCancellationResult | None:
        """Cancel a pending voice transcription."""
        return await self._voice_flow.cancel_pending_voice(scope, reply_id)

    async def cancel_all_pending_voices(
        self,
    ) -> tuple[VoiceCancellationResult, ...]:
        """Cancel every pending voice transcription and handoff."""
        return await self._voice_flow.cancel_all_pending_voices()

    async def cancel_pending_voices_in_scope(
        self,
        scope: MessageScope,
    ) -> tuple[VoiceCancellationResult, ...]:
        """Cancel pending voice transcriptions belonging to one chat."""
        return await self._voice_flow.cancel_pending_voices_in_scope(scope)

    async def start(self) -> None:
        """Initialize and connect to Telegram."""
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        if self.telegram_proxy_url:
            request = HTTPXRequest(
                connection_pool_size=8,
                connect_timeout=30.0,
                read_timeout=30.0,
                proxy=self.telegram_proxy_url,
            )
            update_request = HTTPXRequest(
                connection_pool_size=8,
                connect_timeout=30.0,
                read_timeout=30.0,
                proxy=self.telegram_proxy_url,
            )
            builder = (
                Application.builder()
                .token(self.bot_token)
                .request(request)
                .get_updates_request(update_request)
            )
        else:
            request = HTTPXRequest(
                connection_pool_size=8, connect_timeout=30.0, read_timeout=30.0
            )
            builder = Application.builder().token(self.bot_token).request(request)
        application = builder.build()
        self._application = application

        application.add_handler(
            MessageHandler(filters.TEXT & (~filters.COMMAND), self._on_telegram_message)
        )
        application.add_handler(CommandHandler("start", self._on_start_command))
        application.add_handler(
            MessageHandler(filters.COMMAND, self._on_telegram_message)
        )
        application.add_handler(MessageHandler(filters.VOICE, self._on_telegram_voice))

        await self._retry_connection_step(
            application.initialize,
            step="initialization",
        )
        await application.start()
        self._limiter.start()
        updater = application.updater
        if updater is not None:
            await self._retry_connection_step(
                lambda: updater.start_polling(drop_pending_updates=False),
                step="polling",
            )
        self._connected = True

        logger.info("Telegram platform started (Bot API)")

    async def _retry_connection_step(
        self,
        operation: Callable[[], Awaitable[object]],
        *,
        step: str,
    ) -> None:
        """Retry one independently repeatable Telegram connection step."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await operation()
                return
            except Exception as exc:
                if attempt == max_attempts:
                    logger.error(
                        "Telegram {} failed after {} attempts",
                        step,
                        max_attempts,
                    )
                    raise
                wait_time = 2 * attempt
                if self._log_api_error_tracebacks:
                    logger.warning(
                        "Telegram {} failed (attempt {}/{}): {}. Retrying in {}s...",
                        step,
                        attempt,
                        max_attempts,
                        exc,
                        wait_time,
                    )
                else:
                    logger.warning(
                        "Telegram {} failed (attempt {}/{}): exc_type={}. Retrying in {}s...",
                        step,
                        attempt,
                        max_attempts,
                        type(exc).__name__,
                        wait_time,
                    )
                await asyncio.sleep(wait_time)

    async def quiesce(self) -> None:
        """Stop Telegram ingress after draining active SDK handlers."""
        application = self._application
        updater = application.updater if application is not None else None
        try:
            if updater is not None and updater.running:
                await updater.stop()
        finally:
            try:
                if application is not None and application.running:
                    await application.stop()
            finally:
                self._connected = False

    async def close(self) -> None:
        """Close Telegram delivery and initialized SDK resources."""
        application = self._application
        try:
            await self.outbound.close()
        finally:
            try:
                await self._limiter.shutdown()
            finally:
                try:
                    if application is not None:
                        await application.shutdown()
                finally:
                    logger.info("Telegram platform closed")

    def on_message(self, handler: Callable[[IncomingMessage], Awaitable[None]]) -> None:
        """Register the workflow callback for inbound messages."""
        self._message_handler = handler

    @property
    def is_connected(self) -> bool:
        """Return whether Telegram startup completed."""
        return self._connected

    async def _on_start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.message:
            await update.message.reply_text("👋 Hello! I am the Claude Code Proxy Bot.")
        await self._on_telegram_message(update, context)

    async def _on_telegram_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        incoming = telegram_text_message_from_update(
            update,
            allowed_user_id=self.allowed_user_id,
            log_raw_messaging_content=self._log_raw_messaging_content,
        )
        if incoming is None or self._message_handler is None:
            return

        try:
            await self._message_handler(incoming)
        except Exception as e:
            if self._log_api_error_tracebacks:
                logger.error("Error handling message: {}", e)
            else:
                logger.error("Error handling message: exc_type={}", type(e).__name__)
            with contextlib.suppress(Exception):
                await self.outbound.send_message(
                    incoming.chat_id,
                    f"❌ *{escape_md_v2('Error:')}* {escape_md_v2(format_user_error_preview(e))}",
                    reply_to=incoming.message_id,
                    message_thread_id=incoming.message_thread_id,
                    parse_mode="MarkdownV2",
                )

    async def _on_telegram_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.message

        async def _reply_text(text: str) -> None:
            if message is not None:
                await message.reply_text(text)

        if await self._voice_flow.reply_if_disabled(_reply_text):
            return

        request = telegram_voice_request_from_update(
            update,
            context,
            allowed_user_id=self.allowed_user_id,
        )
        if request is None:
            return

        await self._voice_flow.handle(
            request,
            message_handler=self._message_handler,
            queue_send_message=self.outbound.queue_send_message,
            queue_delete_messages=self.outbound.queue_delete_messages,
        )
