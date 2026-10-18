"""Wait for owned work in tests without imposing it on catalog consumers."""

import asyncio
from contextlib import asynccontextmanager

from free_claude_code.core.async_rwlock import AsyncReadWriteLock
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


async def finish_discovery(manager: ProviderRuntimeManager) -> None:
    await manager._start_pass(manager._current)


async def finish_publication(manager: ProviderRuntimeManager) -> None:
    task = manager._publication_task
    if task is not None:
        await task


def pause_next_write(monkeypatch, lock: AsyncReadWriteLock):
    """Pause one operation before ownership, allowing a later action to overtake it."""
    entered, release = asyncio.Event(), asyncio.Event()
    original = lock.write
    first = True

    @asynccontextmanager
    async def held():
        nonlocal first
        if first:
            first = False
            entered.set()
            await release.wait()
        async with original():
            yield

    monkeypatch.setattr(lock, "write", held)
    return entered, release
