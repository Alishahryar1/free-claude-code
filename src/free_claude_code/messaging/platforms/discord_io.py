"""Discord outbound delivery."""

import asyncio
import secrets
from collections.abc import Awaitable, Callable
from typing import Any, cast

import aiohttp
from loguru import logger

from ..delivery_failures import (
    MAX_ATTEMPTS,
    CallState,
    DeliveryFailure,
    FailureKind,
    Retry,
    decide,
)
from ..limiter import MessagingRateLimiter
from .outbox import PlatformOutbox

DISCORD_MESSAGE_LIMIT = 2000

ClientGetter = Callable[[], Any]
DiscordGetter = Callable[[], Any]


def truncate_discord_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> str:
    """Return text that fits Discord's message limit."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _failure(error: BaseException) -> DeliveryFailure:
    """Read a failed Discord call as a platform-neutral delivery failure.

    discord.py already retries rate limits, server errors and the macOS and
    Windows connection resets, so only connection failures it lets through
    count as lost calls.
    """
    if isinstance(error, aiohttp.ClientConnectionError | TimeoutError):
        return DeliveryFailure(FailureKind.NETWORK)
    return DeliveryFailure(FailureKind.REJECTED)


class DiscordMessenger:
    """Owns Discord sends, edits, deletes, and queued delivery."""

    def __init__(
        self,
        *,
        get_client: ClientGetter,
        get_discord: DiscordGetter,
        limiter: MessagingRateLimiter,
    ) -> None:
        self._get_client = get_client
        self._get_discord = get_discord
        self._outbox = PlatformOutbox(
            limiter=limiter,
            send=self.send_message,
            edit=self.edit_message,
            delete_many=self.delete_messages,
        )

    async def _with_retry[T](self, func: Callable[[], Awaitable[T]]) -> T:
        """Run one Discord operation, retrying calls lost on the network."""
        attempts = 1
        while True:
            try:
                return await func()
            except Exception as e:
                failure = _failure(e)
                state = CallState(
                    attempts=attempts, formatted=False, suppress_known=True
                )
                match decide(failure, state):
                    case Retry(delay=delay):
                        logger.warning(
                            "Discord API network error (attempt {}/{}): {}. Retrying in {}s...",
                            attempts,
                            MAX_ATTEMPTS,
                            type(e).__name__,
                            delay,
                        )
                        attempts += 1
                        await asyncio.sleep(delay)
                    case _:
                        if failure.kind is FailureKind.NETWORK:
                            logger.error(
                                "Discord API failed after {} attempts: {}",
                                attempts,
                                type(e).__name__,
                            )
                        raise

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: str | None = None,
    ) -> str:
        """Send a Discord message immediately."""
        client = self._get_client()
        channel = client.get_channel(int(chat_id))
        if not channel or not hasattr(channel, "send"):
            raise RuntimeError(f"Channel {chat_id} not found")

        text = truncate_discord_message(text)
        channel = cast(Any, channel)
        # Discord returns the existing message when a nonce is reused, so a
        # retried send whose response was lost does not post twice.
        nonce = secrets.token_hex(12)

        if reply_to:
            discord = self._get_discord()
            ref = discord.MessageReference(
                message_id=int(reply_to),
                channel_id=int(chat_id),
            )

            async def _send() -> Any:
                return await channel.send(content=text, reference=ref, nonce=nonce)
        else:

            async def _send() -> Any:
                return await channel.send(content=text, nonce=nonce)

        msg = await self._with_retry(_send)
        return str(msg.id)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
    ) -> None:
        """Edit a Discord message immediately."""
        client = self._get_client()
        channel = client.get_channel(int(chat_id))
        if not channel or not hasattr(channel, "fetch_message"):
            raise RuntimeError(f"Channel {chat_id} not found")

        discord = self._get_discord()
        channel = cast(Any, channel)

        async def _edit() -> None:
            try:
                msg = await channel.fetch_message(int(message_id))
            except discord.NotFound:
                return
            await msg.edit(content=truncate_discord_message(text))

        await self._with_retry(_edit)

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        """Delete a Discord message immediately."""
        client = self._get_client()
        channel = client.get_channel(int(chat_id))
        if not channel or not hasattr(channel, "fetch_message"):
            return

        discord = self._get_discord()
        channel = cast(Any, channel)

        async def _delete() -> None:
            try:
                msg = await channel.fetch_message(int(message_id))
                await msg.delete()
            except discord.NotFound, discord.Forbidden:
                pass

        await self._with_retry(_delete)

    async def delete_messages(self, chat_id: str, message_ids: list[str]) -> None:
        """Delete multiple Discord messages best-effort."""
        for mid in message_ids:
            await self.delete_message(chat_id, mid)

    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
        message_thread_id: str | None = None,
    ) -> str | None:
        """Queue a Discord send."""
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
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
    ) -> None:
        """Queue a Discord edit."""
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
        """Queue a Discord bulk delete."""
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
