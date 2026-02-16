"""
Telegram Platform Adapter

Implements MessagingPlatform for Telegram using python-telegram-bot.
"""

import asyncio
import os

# Opt-in to future behavior for python-telegram-bot (retry_after as timedelta)
# This must be set BEFORE importing telegram.error
os.environ["PTB_TIMEDELTA"] = "1"

from typing import Callable, Awaitable, Optional, Any

from loguru import logger

from .base import MessagingPlatform
from .models import IncomingMessage
from .telegram_markdown import escape_md_v2


# Optional import - python-telegram-bot may not be installed
try:
    from telegram import Update, Bot
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        ContextTypes,
        filters,
    )
    from telegram.error import TelegramError, RetryAfter, NetworkError
    from telegram.request import HTTPXRequest

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class TelegramPlatform(MessagingPlatform):
    """
    Telegram messaging platform adapter.

    Uses python-telegram-bot (BoT API) for Telegram access.
    Requires a Bot Token from @BotFather.
    """

    name = "telegram"

    def __init__(
        self,
        bot_token: Optional[str] = None,
        allowed_user_id: Optional[str] = None,
    ):
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is required. Install with: pip install python-telegram-bot"
            )

        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.allowed_user_id = allowed_user_id or os.getenv("ALLOWED_TELEGRAM_USER_ID")

        if not self.bot_token:
            # We don't raise here to allow instantiation for testing/conditional logic,
            # but start() will fail.
            logger.warning("TELEGRAM_BOT_TOKEN not set")

        self._application: Optional[Application] = None
        self._message_handler: Optional[
            Callable[[IncomingMessage], Awaitable[None]]
        ] = None
        self._connected = False
        self._limiter: Optional[Any] = None  # Will be MessagingRateLimiter

    async def start(self) -> None:
        """Initialize and connect to Telegram."""
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        # Configure request with longer timeouts
        request = HTTPXRequest(
            connection_pool_size=8, connect_timeout=30.0, read_timeout=30.0
        )

        # Build Application
        builder = Application.builder().token(self.bot_token).request(request)
        self._application = builder.build()

        # Register Internal Handlers
        # We catch ALL text messages and commands to forward them
        self._application.add_handler(
            MessageHandler(filters.TEXT & (~filters.COMMAND), self._on_telegram_message)
        )
        self._application.add_handler(CommandHandler("start", self._on_start_command))
        # Catch-all for other commands if needed, or let them fall through
        self._application.add_handler(
            MessageHandler(filters.COMMAND, self._on_telegram_message)
        )

        # Initialize internal components with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self._application.initialize()
                await self._application.start()

                # Start polling (non-blocking way for integration)
                if self._application.updater:
                    await self._application.updater.start_polling(
                        drop_pending_updates=False
                    )

                self._connected = True
                break
            except (NetworkError, Exception) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    logger.warning(
                        f"Connection failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect after {max_retries} attempts")
                    raise

        # Initialize rate limiter
        from .limiter import MessagingRateLimiter

        self._limiter = await MessagingRateLimiter.get_instance()

        # Send startup notification
        try:
            target = self.allowed_user_id
            if target:
                startup_text = (
                    f"🚀 *{escape_md_v2('Claude Code Proxy is online!')}* "
                    f"{escape_md_v2('(Bot API)')}"
                )
                await self.send_message(
                    target,
                    startup_text,
                )
        except Exception as e:
            logger.warning(f"Could not send startup message: {e}")

        logger.info("Telegram platform started (Bot API)")

    async def stop(self) -> None:
        """Stop the bot."""
        if self._application and self._application.updater:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()

        self._connected = False
        logger.info("Telegram platform stopped")

    async def _with_retry(
        self, func: Callable[..., Awaitable[Any]], *args, **kwargs
    ) -> Any:
        """Helper to execute a function with exponential backoff on network errors."""
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except (NetworkError, asyncio.TimeoutError) as e:
                if "Message is not modified" in str(e):
                    return None
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"Telegram API network error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Telegram API failed after {max_retries} attempts: {e}"
                    )
                    raise
            except RetryAfter as e:
                # Telegram explicitly tells us to wait (PTB_TIMEDELTA: retry_after is timedelta)
                from datetime import timedelta

                retry_after = e.retry_after
                if isinstance(retry_after, timedelta):
                    wait_secs = retry_after.total_seconds()
                else:
                    wait_secs = float(retry_after)

                logger.warning(f"Rate limited by Telegram, waiting {wait_secs}s...")
                await asyncio.sleep(wait_secs)
                # We don't increment attempt here, as this is a specific instruction
                return await func(*args, **kwargs)
            except TelegramError as e:
                # Non-network Telegram errors
                err_lower = str(e).lower()
                if "message is not modified" in err_lower:
                    return None
                # Best-effort no-op cases (common during chat cleanup / /clear).
                if any(
                    x in err_lower
                    for x in [
                        "message to edit not found",
                        "message to delete not found",
                        "message can't be deleted",
                        "message can't be edited",
                        "not enough rights to delete",
                    ]
                ):
                    return None
                if "Can't parse entities" in str(e) and kwargs.get("parse_mode"):
                    logger.warning("Markdown failed, retrying without parse_mode")
                    kwargs["parse_mode"] = None
                    return await func(*args, **kwargs)
                raise

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
        parse_mode: Optional[str] = "MarkdownV2",
    ) -> str:
        """Send a message to a chat."""
        app = self._application
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        async def _do_send(parse_mode=parse_mode):
            bot = app.bot
            msg = await bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=int(reply_to) if reply_to else None,
                parse_mode=parse_mode,
            )
            return str(msg.message_id)

        return await self._with_retry(_do_send, parse_mode=parse_mode)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: Optional[str] = "MarkdownV2",
    ) -> None:
        """Edit an existing message."""
        app = self._application
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        async def _do_edit(parse_mode=parse_mode):
            bot = app.bot
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(message_id),
                text=text,
                parse_mode=parse_mode,
            )

        await self._with_retry(_do_edit, parse_mode=parse_mode)

    async def delete_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete a message from a chat."""
        app = self._application
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        async def _do_delete():
            bot = app.bot
            await bot.delete_message(chat_id=chat_id, message_id=int(message_id))

        await self._with_retry(_do_delete)

    async def delete_messages(self, chat_id: str, message_ids: list[str]) -> None:
        """Delete multiple messages (best-effort)."""
        if not message_ids:
            return
        app = self._application
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        # PTB supports bulk deletion via delete_messages; fall back to per-message.
        bot = app.bot
        if hasattr(bot, "delete_messages"):

            async def _do_bulk():
                mids = []
                for mid in message_ids:
                    try:
                        mids.append(int(mid))
                    except Exception:
                        continue
                if not mids:
                    return None
                # delete_messages accepts a sequence of ints (up to 100).
                await bot.delete_messages(chat_id=chat_id, message_ids=mids)

            await self._with_retry(_do_bulk)
            return

        for mid in message_ids:
            await self.delete_message(chat_id, mid)

    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
        parse_mode: Optional[str] = "MarkdownV2",
        fire_and_forget: bool = True,
    ) -> Optional[str]:
        """Enqueue a message to be sent (using limiter)."""
        # Note: Bot API handles limits better, but we still use our limiter for nice queuing
        if not self._limiter:
            return await self.send_message(chat_id, text, reply_to, parse_mode)

        async def _send():
            return await self.send_message(chat_id, text, reply_to, parse_mode)

        if fire_and_forget:
            self._limiter.fire_and_forget(_send)
            return None
        else:
            return await self._limiter.enqueue(_send)

    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: Optional[str] = "MarkdownV2",
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a message edit."""
        if not self._limiter:
            return await self.edit_message(chat_id, message_id, text, parse_mode)

        async def _edit():
            return await self.edit_message(chat_id, message_id, text, parse_mode)

        dedup_key = f"edit:{chat_id}:{message_id}"
        if fire_and_forget:
            self._limiter.fire_and_forget(_edit, dedup_key=dedup_key)
        else:
            await self._limiter.enqueue(_edit, dedup_key=dedup_key)

    async def queue_delete_message(
        self,
        chat_id: str,
        message_id: str,
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a message delete."""
        if not self._limiter:
            return await self.delete_message(chat_id, message_id)

        async def _delete():
            return await self.delete_message(chat_id, message_id)

        dedup_key = f"del:{chat_id}:{message_id}"
        if fire_and_forget:
            self._limiter.fire_and_forget(_delete, dedup_key=dedup_key)
        else:
            await self._limiter.enqueue(_delete, dedup_key=dedup_key)

    async def queue_delete_messages(
        self,
        chat_id: str,
        message_ids: list[str],
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a bulk delete (if supported) or a sequence of deletes."""
        if not message_ids:
            return

        if not self._limiter:
            return await self.delete_messages(chat_id, message_ids)

        async def _bulk():
            return await self.delete_messages(chat_id, message_ids)

        # Dedup by the chunk content; okay to be coarse here.
        dedup_key = f"del_bulk:{chat_id}:{hash(tuple(message_ids))}"
        if fire_and_forget:
            self._limiter.fire_and_forget(_bulk, dedup_key=dedup_key)
        else:
            await self._limiter.enqueue(_bulk, dedup_key=dedup_key)

    def fire_and_forget(self, task: Awaitable[Any]) -> None:
        """Execute a coroutine without awaiting it."""
        if asyncio.iscoroutine(task):
            asyncio.create_task(task)
        else:
            asyncio.ensure_future(task)

    def on_message(
        self,
        handler: Callable[[IncomingMessage], Awaitable[None]],
    ) -> None:
        """Register a message handler callback."""
        self._message_handler = handler

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    async def _on_start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        if update.message:
            await update.message.reply_text("👋 Hello! I am the Claude Code Proxy Bot.")
        # We can also treat this as a message if we want it to trigger something
        await self._on_telegram_message(update, context)

    async def _on_telegram_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming updates."""
        if (
            not update.message
            or not update.message.text
            or not update.effective_user
            or not update.effective_chat
        ):
            return

        user_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)

        # Security check
        if self.allowed_user_id:
            if user_id != str(self.allowed_user_id).strip():
                logger.warning(f"Unauthorized access attempt from {user_id}")
                return

        message_id = str(update.message.message_id)
        reply_to = (
            str(update.message.reply_to_message.message_id)
            if update.message.reply_to_message
            else None
        )
        text_preview = (update.message.text or "")[:80]
        if len(update.message.text or "") > 80:
            text_preview += "..."
        logger.info(
            "TELEGRAM_MSG: chat_id=%s message_id=%s reply_to=%s text_preview=%r",
            chat_id,
            message_id,
            reply_to,
            text_preview,
        )

        if not self._message_handler:
            return

        incoming = IncomingMessage(
            text=update.message.text,
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id,
            platform="telegram",
            reply_to_message_id=reply_to,
            raw_event=update,
        )

        try:
            await self._message_handler(incoming)
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            try:
                await self.send_message(
                    chat_id,
                    f"❌ *{escape_md_v2('Error:')}* {escape_md_v2(str(e)[:200])}",
                    reply_to=incoming.message_id,
                    parse_mode="MarkdownV2",
                )
            except Exception:
                pass
