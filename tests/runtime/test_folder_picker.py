import asyncio
import json
import os
import subprocess
import sys

import pytest

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.runtime.folder_picker import NativeFolderPicker


class DialogProcess:
    """A real child, with barriers around the return from subprocess creation."""

    def __init__(self, script):
        self.script = script
        self.started = asyncio.Event()
        self.return_spawn = asyncio.Event()
        self.return_spawn.set()
        self.process: asyncio.subprocess.Process

    async def spawn(self, _initial_path):
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            "print('ready', flush=True)\n" + self.script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            start_new_session=sys.platform != "win32",
        )
        assert self.process.stdout is not None
        assert (await self.process.stdout.readline()).rstrip(b"\r\n") == b"ready"
        self.started.set()
        await self.return_spawn.wait()
        return self.process


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_selection_and_cancellation_return_after_child_exit(
    monkeypatch, tmp_path, cancelled
):
    path = None if cancelled else str(tmp_path / "project café with spaces")
    child = DialogProcess(f"print({json.dumps({'path': path})!r})")
    picker = NativeFolderPicker()
    monkeypatch.setattr(picker, "_spawn", child.spawn)
    assert await picker.pick_folder(None) == path
    assert child.process.returncode == 0
    # Completion makes the next independent click available.
    assert await picker.pick_folder(None) == path
    await picker.close()


@pytest.mark.asyncio
async def test_selection_failure_racing_shutdown_is_reported_to_request(monkeypatch):
    picker = NativeFolderPicker()
    started = asyncio.Event()

    async def spawn(_initial):
        started.set()
        await picker._stop.wait()
        raise FileNotFoundError("Native picker disappeared while shutdown started")

    monkeypatch.setattr(picker, "_spawn", spawn)
    request = asyncio.create_task(picker.pick_folder(None))
    await started.wait()
    try:
        await picker.close()
    finally:
        with pytest.raises(
            ApplicationUnavailableError, match="Enter the path manually"
        ):
            await request


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ['{"path": 42}', "{}", "not json", '{"path":""}'])
async def test_invalid_child_result_gives_manual_entry_guidance(monkeypatch, output):
    child = DialogProcess(f"print({output!r})")
    picker = NativeFolderPicker()
    monkeypatch.setattr(picker, "_spawn", child.spawn)
    with pytest.raises(ApplicationUnavailableError, match="Enter the path manually"):
        await picker.pick_folder(None)
    assert child.process.returncode == 0
    await picker.close()


@pytest.mark.asyncio
async def test_two_clicks_and_disconnect_during_spawn_keep_one_owned_child(monkeypatch):
    child = DialogProcess("import threading\nthreading.Event().wait()")
    child.return_spawn.clear()
    picker = NativeFolderPicker()
    monkeypatch.setattr(picker, "_spawn", child.spawn)
    request = asyncio.create_task(picker.pick_folder(None))
    try:
        await child.started.wait()
        request.cancel()
        with pytest.raises(ApplicationUnavailableError, match="already open"):
            await picker.pick_folder(None)
        assert not request.done()
        request.cancel()
        child.return_spawn.set()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert child.process.returncode is not None
    finally:
        child.return_spawn.set()
        await picker.close()


@pytest.mark.asyncio
async def test_shutdown_closes_picker_and_rejects_new_clicks(monkeypatch):
    child = DialogProcess("import threading\nthreading.Event().wait()")
    picker = NativeFolderPicker()
    monkeypatch.setattr(picker, "_spawn", child.spawn)
    request = asyncio.create_task(picker.pick_folder(None))
    await child.started.wait()
    picker.begin_shutdown()
    with pytest.raises(ApplicationUnavailableError, match="shutting down"):
        await picker.pick_folder(None)
    await picker.close()
    assert await request is None
    assert child.process.returncode is not None
    await picker.close()


@pytest.mark.asyncio
async def test_interrupted_close_still_waits_for_child_cleanup(monkeypatch):
    child = DialogProcess("import threading\nthreading.Event().wait()")
    child.return_spawn.clear()
    picker = NativeFolderPicker()
    monkeypatch.setattr(picker, "_spawn", child.spawn)
    request = asyncio.create_task(picker.pick_folder(None))
    await child.started.wait()
    closing = asyncio.create_task(picker.close())
    # The explicit stop signal acknowledges that close has started.
    await picker._stop.wait()
    closing.cancel()
    child.return_spawn.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert await request is None
    assert child.process.returncode is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process group ownership")
@pytest.mark.asyncio
async def test_cancellation_reaps_a_helper_waiting_on_its_child(monkeypatch):
    child = DialogProcess(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', 'import threading; threading.Event().wait()'])"
    )
    picker = NativeFolderPicker()
    monkeypatch.setattr(picker, "_spawn", child.spawn)
    request = asyncio.create_task(picker.pick_folder(None))
    await child.started.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    # communicate() also waits for pipes inherited by descendants to close.
    assert child.process.returncode is not None
    await picker.close()
