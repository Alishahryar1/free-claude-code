"""Shared queued delivery helper for messaging platforms."""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from typing import Any, cast

from loguru import logger

from ..limiter import MessagingRateLimiter

SendOperation = Callable[
    [str, str, str | None, str | None, str | None],
    Awaitable[str],
]
EditOperation = Callable[[str, str, str, str | None], Awaitable[None]]
DeleteManyOperation = Callable[[str, list[str]], Awaitable[None]]


class PlatformOutbox:
    """Own queueing, deduplication, and fire-and-forget delivery policy."""

    def __init__(
        self,
        *,
        limiter: MessagingRateLimiter,
        send: SendOperation,
        edit: EditOperation,
        delete_many: DeleteManyOperation,
    ) -> None:
        self._limiter = limiter
        self._send = send
        self._edit = edit
        self._delete_many = delete_many
        self._background_tasks: set[asyncio.Future[Any]] = set()
        self._closed = False

    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
        message_thread_id: str | None = None,
    ) -> str | None:
        """Queue or immediately send a platform message."""
        self._require_open()

        async def _send() -> str:
            return await self._send(
                chat_id,
                text,
                reply_to,
                parse_mode,
                message_thread_id,
            )

        if fire_and_forget:
            self._limiter.fire_and_forget(_send)
            return None
        return cast(str | None, await self._limiter.enqueue(_send))

    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
    ) -> None:
        """Queue or immediately edit a platform message."""
        self._require_open()

        async def _edit() -> None:
            await self._edit(chat_id, message_id, text, parse_mode)

        dedup_key = f"edit:{chat_id}:{message_id}"
        if fire_and_forget:
            self._limiter.fire_and_forget(_edit, dedup_key=dedup_key)
        else:
            await self._limiter.enqueue(_edit, dedup_key=dedup_key)

    async def queue_delete_messages(
        self,
        chat_id: str,
        message_ids: list[str],
        fire_and_forget: bool = True,
    ) -> None:
        """Queue or immediately bulk-delete platform messages."""
        self._require_open()
        ids_snapshot = tuple(str(message_id) for message_id in message_ids)
        if not ids_snapshot:
            return

        async def _delete_many() -> None:
            await self._delete_many(chat_id, list(ids_snapshot))

        digest = hashlib.sha256("\x1f".join(ids_snapshot).encode()).hexdigest()[:16]
        dedup_key = f"del_bulk:{chat_id}:{digest}"
        if fire_and_forget:
            self._limiter.fire_and_forget(_delete_many, dedup_key=dedup_key)
        else:
            await self._limiter.enqueue(_delete_many, dedup_key=dedup_key)

    def fire_and_forget(self, task: Awaitable[Any]) -> None:
        """Run and retain arbitrary outbound work until completion or shutdown."""
        future = asyncio.ensure_future(task)
        if self._closed:
            future.cancel()
            raise RuntimeError("Platform outbox is closed.")
        self._background_tasks.add(future)
        future.add_done_callback(self._complete_background_task)

    async def close(self) -> None:
        """Cancel and await arbitrary outbound work owned by this outbox."""
        if not self._closed:
            self._closed = True
        tasks = tuple(self._background_tasks)
        for task in tasks:
            task.cancel()
        try:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._background_tasks.difference_update(
                task for task in tasks if task.done()
            )

    def _complete_background_task(self, task: asyncio.Future[Any]) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Outbound background task failed: exc_type={}",
                type(error).__name__,
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Platform outbox is closed.")
