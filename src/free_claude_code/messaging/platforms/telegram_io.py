"""Telegram outbound delivery."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from loguru import logger

from ..delivery_failures import (
    MAX_ATTEMPTS,
    CallState,
    DeliveryFailure,
    FailureKind,
    Ignore,
    PlatformRateLimited,
    Raise,
    RateLimited,
    Retry,
    RetryPlain,
    decide,
)
from ..limiter import MessagingRateLimiter
from .outbox import PlatformOutbox

TELEGRAM_DELETE_MESSAGES_BATCH_SIZE = 100


class _NoBadRequest(Exception):
    """Stands in for BadRequest when python-telegram-bot is not installed."""


TelegramNetworkError: type[BaseException]
TelegramBadRequest: type[BaseException]
TelegramRetryAfter: type[BaseException]
TelegramBaseError: type[BaseException]
try:
    from telegram.error import (
        BadRequest as _TelegramBadRequest,
    )
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
    TelegramBadRequest = _TelegramBadRequest
    TelegramRetryAfter = _TelegramRetryAfter
    TelegramBaseError = _TelegramBaseError
except ImportError:
    TelegramNetworkError = TimeoutError
    TelegramBadRequest = _NoBadRequest
    TelegramRetryAfter = TimeoutError
    TelegramBaseError = Exception

ApplicationGetter = Callable[[], Any | None]

_GONE_MESSAGE_ERRORS = (
    "message to edit not found",
    "message to delete not found",
    "message can't be deleted",
    "message can't be edited",
    "not enough rights to delete",
)


def _failure(error: BaseException) -> DeliveryFailure:
    """Read a failed Telegram call as a platform-neutral delivery failure."""
    text = str(error).lower()
    if isinstance(error, TelegramRetryAfter):
        return DeliveryFailure(
            FailureKind.RATE_LIMITED, _seconds(getattr(error, "retry_after", 0))
        )
    if "message is not modified" in text:
        return DeliveryFailure(FailureKind.UNMODIFIED)
    # BadRequest subclasses NetworkError: a rejection, not a lost call.
    if not isinstance(error, TelegramBadRequest) and isinstance(
        error, (TimeoutError, TelegramNetworkError)
    ):
        return DeliveryFailure(FailureKind.NETWORK)
    if any(phrase in text for phrase in _GONE_MESSAGE_ERRORS):
        return DeliveryFailure(FailureKind.GONE)
    if "can't parse entities" in text:
        return DeliveryFailure(FailureKind.UNPARSEABLE)
    return DeliveryFailure(FailureKind.REJECTED)


def _seconds(retry_after: float | timedelta) -> float:
    if isinstance(retry_after, timedelta):
        return retry_after.total_seconds()
    return float(retry_after)


def _log_retry(
    failure: DeliveryFailure, attempt: int, error: BaseException, delay: float
) -> None:
    if failure.kind is FailureKind.RATE_LIMITED:
        logger.warning("Rate limited by Telegram, waiting {}s...", delay)
        return
    logger.warning(
        "Telegram API network error (attempt {}/{}): {}. Retrying in {}s...",
        attempt,
        MAX_ATTEMPTS,
        error,
        delay,
    )


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
        """Execute a Telegram API call with the platform retry policy.

        Network errors and rate limits share one budget of three calls; a
        plain-text resend after unparseable Markdown does not use it.
        """
        attempts = 1
        while True:
            try:
                return await func(*args, **kwargs)
            except (TimeoutError, TelegramBaseError) as e:
                failure = _failure(e)
                state = CallState(
                    attempts=attempts,
                    formatted=bool(kwargs.get("parse_mode")),
                    suppress_known=suppress_known_message_errors,
                )
                match decide(failure, state):
                    case Ignore():
                        return None
                    case RetryPlain():
                        logger.warning("Markdown failed, retrying without parse_mode")
                        kwargs["parse_mode"] = None
                    case Retry(delay=delay):
                        _log_retry(failure, attempts, e, delay)
                        attempts += 1
                        await asyncio.sleep(delay)
                    case RateLimited(retry_after=retry_after):
                        raise PlatformRateLimited(retry_after) from e
                    case Raise():
                        if failure.kind is FailureKind.NETWORK:
                            logger.error(
                                "Telegram API failed after {} attempts: {}",
                                attempts,
                                e,
                            )
                        raise

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
