"""Explicit barriers for Code browser tests, on the application's own loop."""

import asyncio
from collections.abc import Coroutine
from unittest.mock import patch

from free_claude_code.application.code_sessions import CodeService
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from tests.code_sessions_support import FakeHarness


class CodeControl:
    def __init__(self, directory):
        self.harness = FakeHarness()
        self.folder_picker = FolderPickerControl()
        self.service = CodeService(
            SQLiteCodeStore(directory / "code.db", directory / "code.lock"),
            self.harness,
        )
        self.loop: asyncio.AbstractEventLoop | None = None

    def run[T](self, work: Coroutine[object, object, T]) -> T:
        assert self.loop is not None
        return asyncio.run_coroutine_threadsafe(work, self.loop).result(timeout=10)

    def connection(self):
        self.run(self.harness.started.wait())
        return self.harness.connections[-1]

    async def hold_send(self):
        self.admission = asyncio.Event()
        original = self.service._store.admit_run

        async def admit(session, run, item, expected_revision):
            await self.admission.wait()
            return await original(session, run, item, expected_revision)

        self.admission_patch = patch.object(self.service._store, "admit_run", admit)
        self.admission_patch.start()

    async def release_send(self):
        self.admission.set()
        self.admission_patch.stop()


class FolderPickerControl:
    """Substitute the native dialog while retaining its real runtime owner."""

    def __init__(self):
        self.calls = asyncio.Queue()

    async def select(self, initial_path, stop):
        result = asyncio.get_running_loop().create_future()
        closed = asyncio.Event()
        self.calls.put_nowait((initial_path, result, closed))
        stopped = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait((result, stopped), return_when=asyncio.FIRST_COMPLETED)
            return None if stop.is_set() else result.result()
        finally:
            stopped.cancel()
            await asyncio.gather(stopped, return_exceptions=True)
            closed.set()

    async def finish(self, call, path):
        call[1].set_result(path)
        await call[2].wait()

    async def fail(self, call):
        call[1].set_exception(RuntimeError("No desktop available"))
        await call[2].wait()
