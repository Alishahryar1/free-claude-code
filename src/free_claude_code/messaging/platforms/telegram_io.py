"""Telegram outbound delivery."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from loguru import logger

from ..limiter import MessagingRateLimiter
from .outbox import PlatformOutbox

TELEGRAM_DELETE_MESSAGES_BATCH_SIZE = 100

TelegramNetworkError: type[BaseException]
TelegramRetryAfter: type[BaseException]
TelegramBaseError: type[BaseException]
try:
    from telegram.error import (
        NetworkError as _TelegramNetworkError,
    )
    from telegram.error import (
        RetryAfter as _TelegramRetryAfter,
    )
    from telegram.error import (
        TelegramError as _TelegramBaseError,
    )

    TelegramNetworkError = _TelegramNetworkError
    TelegramRetryAfter = _TelegramRetryAfter
    TelegramBaseError = _TelegramBaseError
except ImportError:
    TelegramNetworkError = TimeoutError
    TelegramRetryAfter = TimeoutError
    TelegramBaseError = Exception

ApplicationGetter = Callable[[], Any | None]


class TelegramMessenger:
    """Owns Telegram sends, edits, deletes, and queued delivery."""

    def __init__(
        self,
        *,
        get_application: ApplicationGetter,
        limiter: MessagingRateLimiter,
    ) -> None:
        self._get_application = get_application
        self._outbox = PlatformOutbox(
            limiter=limiter,
            send=self.send_message,
            edit=self.edit_message,
            delete_many=self.delete_messages,
        )

    async def _with_retry(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        suppress_known_message_errors: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Execute a Telegram API call with the platform retry policy."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except (TimeoutError, TelegramNetworkError) as e:
                if "Message is not modified" in str(e):
                    if suppress_known_message_errors:
                        return None
                    raise
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        "Telegram API network error (attempt {}/{}): {}. "
                        "Retrying in {}s...",
                        attempt + 1,
                        max_retries,
                        e,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "Telegram API failed after {} attempts: {}",
                        max_retries,
                        e,
                    )
                    raise
            except TelegramRetryAfter as e:
                retry_after = getattr(e, "retry_after", 0)
                wait_secs = (
                    retry_after.total_seconds()
                    if isinstance(retry_after, timedelta)
                    else float(retry_after)
                )
                logger.warning("Rate limited by Telegram, waiting {}s...", wait_secs)
                await asyncio.sleep(wait_secs)
                return await func(*args, **kwargs)
            except TelegramBaseError as e:
                err_lower = str(e).lower()
                if "message is not modified" in err_lower:
                    if suppress_known_message_errors:
                        return None
                    raise
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
                    if suppress_known_message_errors:
                        return None
                    raise
                if "Can't parse entities" in str(e) and kwargs.get("parse_mode"):
                    logger.warning("Markdown failed, retrying without parse_mode")
                    kwargs["parse_mode"] = None
                    return await func(*args, **kwargs)
                raise
        return None

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = "MarkdownV2",
        message_thread_id: str | None = None,
    ) -> str:
        """Send a Telegram message immediately."""
        app = self._get_application()
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        async def _do_send(parse_mode: str | None = parse_mode) -> str:
            kwargs: dict[str, Any] = {
                "chat_id": chat_id,
                "text": text,
                "reply_to_message_id": int(reply_to) if reply_to else None,
                "parse_mode": parse_mode,
            }
            if message_thread_id is not None:
                kwargs["message_thread_id"] = int(message_thread_id)
            msg = await app.bot.send_message(**kwargs)
            return str(msg.message_id)

        return await self._with_retry(_do_send, parse_mode=parse_mode)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = "MarkdownV2",
    ) -> None:
        """Edit a Telegram message immediately."""
        app = self._get_application()
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        async def _do_edit(parse_mode: str | None = parse_mode) -> None:
            await app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(message_id),
                text=text,
                parse_mode=parse_mode,
            )

        await self._with_retry(_do_edit, parse_mode=parse_mode)

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        """Delete a Telegram message immediately."""
        app = self._get_application()
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        async def _do_delete() -> None:
            await app.bot.delete_message(chat_id=chat_id, message_id=int(message_id))

        await self._with_retry(_do_delete)

    async def delete_messages(self, chat_id: str, message_ids: list[str]) -> None:
        """Delete multiple Telegram messages best-effort."""
        if not message_ids:
            return
        app = self._get_application()
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        bot = app.bot
        mids: list[int] = []
        for mid in message_ids:
            try:
                mids.append(int(mid))
            except Exception:
                continue
        if not mids:
            return

        if hasattr(bot, "delete_messages"):
            for start in range(0, len(mids), TELEGRAM_DELETE_MESSAGES_BATCH_SIZE):
                chunk = mids[start : start + TELEGRAM_DELETE_MESSAGES_BATCH_SIZE]
                chunk_snapshot = tuple(chunk)

                async def _do_bulk(ids: tuple[int, ...] = chunk_snapshot) -> None:
                    await bot.delete_messages(chat_id=chat_id, message_ids=list(ids))

                try:
                    await self._with_retry(
                        _do_bulk,
                        suppress_known_message_errors=False,
                    )
                except Exception as e:
                    logger.debug(
                        "Telegram bulk delete failed for chat {}: {}; falling back",
                        chat_id,
                        type(e).__name__,
                    )
                    for mid in chunk:
                        await self.delete_message(chat_id, str(mid))
            return

        for mid in mids:
            await self.delete_message(chat_id, str(mid))

    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = "MarkdownV2",
        fire_and_forget: bool = True,
        message_thread_id: str | None = None,
    ) -> str | None:
        """Queue a Telegram send."""
        return await self._outbox.queue_send_message(
            chat_id,
            text,
            reply_to,
            parse_mode,
            fire_and_forget,
            message_thread_id,
        )

    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = "MarkdownV2",
        fire_and_forget: bool = True,
    ) -> None:
        """Queue a Telegram edit."""
        await self._outbox.queue_edit_message(
            chat_id,
            message_id,
            text,
            parse_mode,
            fire_and_forget,
        )

    async def queue_delete_messages(
        self,
        chat_id: str,
        message_ids: list[str],
        fire_and_forget: bool = True,
    ) -> None:
        """Queue a Telegram bulk delete."""
        await self._outbox.queue_delete_messages(
            chat_id,
            message_ids,
            fire_and_forget,
        )

    def fire_and_forget(self, task: Awaitable[Any]) -> None:
        """Execute a coroutine without awaiting it."""
        self._outbox.fire_and_forget(task)

    async def close(self) -> None:
        """Cancel outstanding outbound work."""
        await self._outbox.close()
