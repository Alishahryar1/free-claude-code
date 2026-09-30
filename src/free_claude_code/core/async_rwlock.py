"""Concurrent readers and exclusive writers with preference for queued writes."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from .async_tasks import wait_owned


class AsyncReadWriteLock:
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._writer and not self._waiting_writers
            )
            self._readers += 1
        try:
            yield
        finally:
            await wait_owned(asyncio.create_task(self._release(writer=False)))

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        async with self._condition:
            self._waiting_writers += 1
            try:
                await self._condition.wait_for(
                    lambda: not self._writer and not self._readers
                )
                self._writer = True
            finally:
                self._waiting_writers -= 1
                self._condition.notify_all()
        try:
            yield
        finally:
            await wait_owned(asyncio.create_task(self._release(writer=True)))

    async def _release(self, *, writer: bool) -> None:
        async with self._condition:
            if writer:
                self._writer = False
            else:
                self._readers -= 1
            self._condition.notify_all()
