"""Device login owns pipe drainage even when cancellation races process spawn."""

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock

import pytest

from free_claude_code.providers.github_copilot import login


class Pipe(asyncio.StreamReader):
    def __init__(self) -> None:
        super().__init__()
        self.drained = asyncio.Event()
        self.feed_data(b"child output")

    async def read(self, n: int = -1) -> bytes:
        data = await super().read(n)
        if not data:
            self.drained.set()
        return data


class Child:
    """Mirror subprocess wait's dependency on completion of both pipe transports."""

    def __init__(self) -> None:
        self.stdout = Pipe()
        self.stderr = Pipe()
        self.returncode: int | None = None
        self.wait_entered = asyncio.Event()

    def kill(self) -> None:
        self.returncode = -1
        self.stdout.feed_eof()
        self.stderr.feed_eof()

    async def wait(self) -> int:
        self.wait_entered.set()
        await self.stdout.drained.wait()
        await self.stderr.drained.wait()
        assert self.returncode is not None
        return self.returncode


@pytest.mark.asyncio
async def test_cancel_during_spawn_drains_pipes_before_waiting_for_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = Child()
    spawned, return_child = asyncio.Event(), asyncio.Event()

    async def spawn(*_args: object, **_kwargs: object) -> Child:
        spawned.set()
        await return_child.wait()
        return child

    async def discard(pipe: Pipe) -> None:
        while await pipe.read(1024):
            pass

    monkeypatch.setattr(login, "verified_cli_path", AsyncMock(return_value="inert-cli"))
    monkeypatch.setattr(login.asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(login.device_login(lambda _challenge: None))
    await spawned.wait()
    task.cancel()
    return_child.set()
    await child.wait_entered.wait()
    try:
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done, "login cancellation abandoned unread child pipes"
        with pytest.raises(asyncio.CancelledError):
            await task
        assert child.stdout.drained.is_set() and child.stderr.drained.is_set()
    finally:
        # Make the red run terminate cleanly too; this does not satisfy the assertion.
        if not task.done():
            await asyncio.gather(discard(child.stdout), discard(child.stderr))
        with suppress(asyncio.CancelledError):
            await task
