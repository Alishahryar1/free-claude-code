import asyncio

import pytest

from free_claude_code.core.async_rwlock import AsyncReadWriteLock


@pytest.mark.asyncio
async def test_readers_overlap_and_writer_waits():
    lock = AsyncReadWriteLock()
    entered, release = asyncio.Event(), asyncio.Event()

    async def reader():
        async with lock.read():
            entered.set()
            await release.wait()

    task = asyncio.create_task(reader())
    try:
        await entered.wait()
        async with lock.read():
            writer_entered = asyncio.Event()

            async def writer():
                async with lock.write():
                    writer_entered.set()

            writer_task = asyncio.create_task(writer())
            await asyncio.sleep(0)
            assert not writer_entered.is_set()
        assert not writer_entered.is_set()
        release.set()
        await asyncio.wait_for(asyncio.gather(task, writer_task), 5)
        assert writer_entered.is_set()
    finally:
        release.set()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_writer", [False, True])
async def test_queued_writer_precedes_later_reader(cancel_writer):
    lock = AsyncReadWriteLock()
    queued, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    order = []

    async def writer():
        queued.set()
        async with lock.write():
            order.append("writer")
            entered.set()
            await release.wait()

    async def reader():
        async with lock.read():
            order.append("reader")

    async with lock.read():
        writer_task = asyncio.create_task(writer())
        await queued.wait()
        reader_task = asyncio.create_task(reader())
        await asyncio.sleep(0)
        assert order == []
        if cancel_writer:
            writer_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await writer_task
            await asyncio.wait_for(reader_task, 5)
            assert order == ["reader"]
    if not cancel_writer:
        await asyncio.wait_for(entered.wait(), 5)
        assert order == ["writer"]
        release.set()
        await asyncio.wait_for(asyncio.gather(writer_task, reader_task), 5)
        assert order == ["writer", "reader"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["read", "write"])
async def test_exception_releases_access(mode):
    lock = AsyncReadWriteLock()
    with pytest.raises(ValueError):
        async with getattr(lock, mode)():
            raise ValueError("failed operation")
    async with lock.write():
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["read", "write"])
async def test_cancelled_queued_access_does_not_strand_waiters(mode):
    lock = AsyncReadWriteLock()
    queued = asyncio.Event()

    async def operation():
        queued.set()
        async with getattr(lock, mode)():
            pytest.fail("Cancelled waiter entered")

    async with lock.write():
        task = asyncio.create_task(operation())
        await queued.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    async with lock.write():
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["read", "write"])
async def test_repeated_cancellation_drains_contended_release(mode):
    lock = AsyncReadWriteLock()
    entered, finish = asyncio.Event(), asyncio.Event()

    async def operation():
        async with getattr(lock, mode)():
            entered.set()
            await finish.wait()

    task = asyncio.create_task(operation())
    await entered.wait()
    # Hold admission bookkeeping so releasing access must await cleanup.
    async with lock._condition:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    async with lock.write():
        pass
