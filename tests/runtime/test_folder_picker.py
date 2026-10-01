import asyncio
import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.runtime.folder_picker import NativeFolderPicker


async def _retained_handle_probe(directory, cancel):
    root = Path(directory)
    ready = root / "descendant.pid"
    child_script = (
        "import os, pathlib, threading; "
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); "
        "threading.Event().wait()"
    )
    selection = str(root)
    payload = json.dumps({"path": selection}) if sys.platform == "win32" else selection
    script = (
        "import pathlib, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_script!r}], start_new_session=True)\n"
        f"while not pathlib.Path({str(ready)!r}).exists(): time.sleep(0.01)\n"
        + ("time.sleep(60)\n" if cancel else f"print({payload!r}, flush=True)\n")
    )
    picker = NativeFolderPicker()
    process = None

    async def spawn(
        _initial, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    ):
        nonlocal process
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            script,
            stdout=stdout,
            stderr=stderr,
            start_new_session=sys.platform != "win32",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        (root / "helper.pid").write_text(str(process.pid))
        return process

    patcher = pytest.MonkeyPatch()
    patcher.setattr(picker, "_spawn", spawn)
    request = asyncio.create_task(picker.pick_folder(None))
    try:
        async with asyncio.timeout(5):
            while not ready.exists() or not ready.read_text():
                await asyncio.sleep(0.01)
        if cancel:
            request.cancel()
        done, _ = await asyncio.wait({request}, timeout=1)
        assert done, "Picker is waiting for output handles retained by a descendant"
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await request
        else:
            assert await request == selection
        assert process is not None and process.returncode is not None
    finally:
        if ready.exists():
            os.kill(int(ready.read_text()), signal.SIGTERM)
            ready.unlink()
        await picker.close()
        await asyncio.gather(request, return_exceptions=True)
        (root / "helper.pid").unlink(missing_ok=True)
        patcher.undo()


@pytest.mark.parametrize("cancel", [False, True])
def test_picker_does_not_wait_for_inherited_output_handles(tmp_path, cancel):
    script = (
        "import asyncio, runpy, sys; "
        "probe = runpy.run_path(sys.argv[1])['_retained_handle_probe']; "
        "asyncio.run(probe(sys.argv[2], sys.argv[3] == 'True'))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, __file__, str(tmp_path), str(cancel)],
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        for path in tmp_path.glob("*.pid"):
            with suppress(ProcessLookupError):
                os.kill(int(path.read_text()), signal.SIGTERM)
    assert result.returncode == 0, result.stdout + result.stderr


class DialogProcess:
    """A real child, with barriers around the return from subprocess creation."""

    def __init__(self, script):
        self.script = script
        self.started = asyncio.Event()
        self.return_spawn = asyncio.Event()
        self.return_spawn.set()
        self.process: asyncio.subprocess.Process
        self.captures = []

    async def spawn(self, _initial_path, stdout, stderr):
        self.captures.extend((stdout, stderr))
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            self.script,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            start_new_session=sys.platform != "win32",
        )
        self.started.set()
        await self.return_spawn.wait()
        return self.process


async def _exec_identity_probe(directory, after_exec):
    root = Path(directory)
    helper_ready, exec_ready, proceed = (
        root / name for name in ("helper.pid", "exec.pid", "proceed")
    )
    command = root / "native-picker"
    command.write_text(
        f"#!{sys.executable}\nimport os, pathlib, threading\n"
        f"pathlib.Path({str(exec_ready)!r}).write_text(str(os.getpid()))\n"
        "threading.Event().wait()\n",
        encoding="utf-8",
    )
    command.chmod(0o700)
    child = DialogProcess(
        "import os, pathlib, time\n"
        "from free_claude_code.runtime import native_folder_dialog as native\n"
        f"pathlib.Path({str(helper_ready)!r}).write_text(str(os.getpid()))\n"
        f"while not pathlib.Path({str(proceed)!r}).exists(): time.sleep(0.01)\n"
        f"native.shutil.which = lambda _: {str(command)!r}\n"
        "native._linux(None)\n"
    )
    picker = NativeFolderPicker()
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(picker, "_spawn", child.spawn)
        request = asyncio.create_task(picker.pick_folder(None))
        try:
            async with asyncio.timeout(5):
                while not helper_ready.exists() or not helper_ready.read_text():
                    await asyncio.sleep(0.01)
                if after_exec:
                    proceed.touch()
                    while not exec_ready.exists() or not exec_ready.read_text():
                        await asyncio.sleep(0.01)
                    assert int(exec_ready.read_text()) == child.process.pid
                assert int(helper_ready.read_text()) == child.process.pid
            request.cancel()
            done, _ = await asyncio.wait({request}, timeout=1)
            assert done, "Picker cancellation did not reap its owned process"
            with pytest.raises(asyncio.CancelledError):
                await request
            assert child.process.returncode is not None
        finally:
            await picker.close()
            await asyncio.gather(request, return_exceptions=True)
            helper_ready.unlink(missing_ok=True)
            exec_ready.unlink(missing_ok=True)


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec replacement")
@pytest.mark.parametrize("after_exec", [False, True])
def test_posix_exec_preserves_owned_pid_and_cancellation(tmp_path, after_exec):
    script = (
        "import asyncio, runpy, sys; "
        "probe = runpy.run_path(sys.argv[1])['_exec_identity_probe']; "
        "asyncio.run(probe(sys.argv[2], sys.argv[3] == 'True'))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, __file__, str(tmp_path), str(after_exec)],
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        if (pid_file := tmp_path / "helper.pid").exists():
            with suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_selection_and_cancellation_return_after_child_exit(
    monkeypatch, tmp_path, cancelled
):
    path = None if cancelled else str(tmp_path / "project café with spaces")
    if sys.platform.startswith("linux") and cancelled:
        script = "import sys; sys.exit(1)"
    else:
        output = json.dumps({"path": path}) if sys.platform == "win32" else path or ""
        script = f"print({output!r})"
    child = DialogProcess(script)
    picker = NativeFolderPicker()
    monkeypatch.setattr(picker, "_spawn", child.spawn)
    assert await picker.pick_folder(None) == path
    assert child.process.returncode == (
        1 if sys.platform.startswith("linux") and cancelled else 0
    )
    for capture in child.captures:
        with pytest.raises(OSError):
            os.fstat(capture)
    # Completion makes the next independent click available.
    assert await picker.pick_folder(None) == path
    await picker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("spawn_fails", [False, True])
async def test_spawn_and_result_failures_close_captures_before_unregister(
    monkeypatch, spawn_fails
):
    from free_claude_code.runtime import folder_picker

    picker = NativeFolderPicker()
    child = DialogProcess("print('invalid result')")
    captures = []
    registrations = []

    async def spawn(initial, stdout, stderr):
        captures.extend((stdout, stderr))
        if spawn_fails:
            raise FileNotFoundError("Native executable disappeared")
        return await child.spawn(initial, stdout, stderr)

    def unregister(pid):
        assert child.process.returncode is not None
        for capture in captures:
            with pytest.raises(OSError):
                os.fstat(capture)
        registrations.remove(pid)

    monkeypatch.setattr(picker, "_spawn", spawn)
    monkeypatch.setattr(folder_picker, "register_pid", registrations.append)
    monkeypatch.setattr(folder_picker, "unregister_pid", unregister)
    with pytest.raises(ApplicationUnavailableError):
        await picker.pick_folder(None)
    assert not registrations
    for capture in captures:
        with pytest.raises(OSError):
            os.fstat(capture)
    assert picker._active is None
    await picker.close()


@pytest.mark.asyncio
async def test_selection_failure_racing_shutdown_is_reported_to_request(monkeypatch):
    picker = NativeFolderPicker()
    started = asyncio.Event()

    async def spawn(_initial, stdout, stderr):
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
