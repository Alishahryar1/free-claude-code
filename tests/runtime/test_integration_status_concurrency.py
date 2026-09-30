import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.harnesses import claude_integration as claude
from free_claude_code.harnesses import vscode_chat_integration as vscode
from tests.runtime.test_integration_startup import runtime as runtime


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other",
    [
        "claude_vscode_status",
        "codex_integration_status",
        "claude_desktop_status",
        "jetbrains_acp_status",
    ],
)
async def test_slow_status_does_not_block_other_integration(
    runtime, monkeypatch, other
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def held(path):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return {"connected": False, "paths": {}}

    monkeypatch.setattr(vscode, "status", held)
    task = asyncio.create_task(runtime.vscode_chat_status())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        result = await asyncio.wait_for(getattr(runtime, other)(), 1)
        assert result["connected"] is False
        assert not task.done()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close()


@pytest.mark.asyncio
async def test_cancelled_reader_drains_before_writer_enters(runtime, monkeypatch):
    entered, writer_entered, release = (
        asyncio.Event(),
        asyncio.Event(),
        threading.Event(),
    )
    loop = asyncio.get_running_loop()
    status, configure = vscode.status, vscode.configure

    def held(path):
        if not entered.is_set():
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return status(path)

    def record_write(*args, **kwargs):
        loop.call_soon_threadsafe(writer_entered.set)
        return configure(*args, **kwargs)

    monkeypatch.setattr(vscode, "status", held)
    monkeypatch.setattr(vscode, "configure", record_write)
    reader = asyncio.create_task(runtime.vscode_chat_status())
    writer = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        reader.cancel()
        await asyncio.sleep(0)
        reader.cancel()
        writer = asyncio.create_task(runtime.connect_vscode_chat())
        await runtime.provider_manager.wait_for_catalog()
        await asyncio.sleep(0)
        assert not reader.done()
        assert not writer_entered.is_set()
        assert (await asyncio.wait_for(runtime.jetbrains_acp_status(), 1))[
            "connected"
        ] is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(reader, 5)
        assert (await asyncio.wait_for(writer, 5))["connected"] is True
        assert writer_entered.is_set()
    finally:
        release.set()
        await asyncio.gather(
            reader, *([writer] if writer else []), return_exceptions=True
        )
        await runtime.close()


@pytest.mark.asyncio
async def test_status_cannot_observe_partial_multifile_write(runtime, monkeypatch):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    atomic = claude.atomic_write_text

    def held(path, content):
        atomic(path, content)
        if path == claude.claude_state_path():
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)

    monkeypatch.setattr(claude, "atomic_write_text", held)
    writer = asyncio.create_task(runtime.connect_claude_vscode())
    reader = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert claude.claude_state_path().exists()
        assert not claude.settings_path().exists()
        reader = asyncio.create_task(runtime.claude_vscode_status())
        assert (await asyncio.wait_for(runtime.jetbrains_acp_status(), 1))[
            "connected"
        ] is False
        assert not reader.done()
        release.set()
        assert (await asyncio.wait_for(writer, 5))["connected"] is True
        assert (await asyncio.wait_for(reader, 5))["connected"] is True
    finally:
        release.set()
        await asyncio.gather(
            writer, *([reader] if reader else []), return_exceptions=True
        )
        await runtime.close()


@pytest.mark.asyncio
async def test_continuously_changed_generation_has_bounded_retry(runtime, monkeypatch):
    entered, release = asyncio.Queue(), threading.Semaphore(0)
    loop = asyncio.get_running_loop()

    def held(path, state_path, url, token, connected=None):
        loop.call_soon_threadsafe(entered.put_nowait, token)
        assert release.acquire(timeout=5)
        return {"connected": False, "paths": {}}

    monkeypatch.setattr(claude, "configure", held)
    task = asyncio.create_task(runtime.claude_vscode_status())
    try:
        for attempt in range(2):
            await asyncio.wait_for(entered.get(), 5)
            await runtime.provider_manager.replace(
                runtime.settings.model_copy(
                    update={"proxy_auth_token": f"rotated-{attempt}"}
                ),
                commit=AsyncMock(),
            )
            release.release()
        with pytest.raises(ApplicationUnavailableError, match="configuration changed"):
            await asyncio.wait_for(task, 5)
        assert entered.empty()
    finally:
        release.release()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["restart", "shutdown"])
async def test_availability_is_rechecked_after_status_worker(
    runtime, monkeypatch, change
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def held(path):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return {"connected": False, "paths": {}}

    monkeypatch.setattr(vscode, "status", held)
    task = asyncio.create_task(runtime.vscode_chat_status())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if change == "shutdown":
            await asyncio.wait_for(runtime.close(), 1)
        else:
            runtime._pending_fields = ["PORT"]
        release.set()
        with pytest.raises(ApplicationUnavailableError):
            await asyncio.wait_for(task, 5)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close()


@pytest.mark.asyncio
async def test_startup_finishing_during_skipped_status_repeats_inspection(
    runtime, monkeypatch
):
    update = runtime._integrations._claude_update
    update.state = "starting"
    loop = asyncio.get_running_loop()
    path = claude.settings_path()
    reads = 0

    def resolve_path():
        nonlocal reads
        reads += 1
        if reads == 1:
            loop.call_soon_threadsafe(update.complete)
        return path

    monkeypatch.setattr(claude, "settings_path", resolve_path)
    monkeypatch.setattr(
        claude, "configure", lambda *args: {"connected": True, "paths": {}}
    )
    try:
        result = await runtime.claude_vscode_status()
        assert result["connected"] is True
        assert result["update"]["state"] == "ready"
        assert reads == 2
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_same_integration_status_readers_overlap(runtime, monkeypatch):
    both_entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    count = 0
    count_lock = threading.Lock()

    def held(path):
        nonlocal count
        with count_lock:
            count += 1
            if count == 2:
                loop.call_soon_threadsafe(both_entered.set)
        assert release.wait(5)
        return {"connected": False, "paths": {}}

    monkeypatch.setattr(vscode, "status", held)
    tasks = [asyncio.create_task(runtime.vscode_chat_status()) for _ in range(2)]
    try:
        await asyncio.wait_for(both_entered.wait(), 1)
        assert all(not task.done() for task in tasks)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_failure", [False, True])
async def test_generation_change_discards_old_status_outcome(
    runtime, monkeypatch, first_failure
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    tokens = []
    initial_token = runtime.settings.proxy_auth_token

    def held(path, state_path, url, token, connected=None):
        tokens.append(token)
        if len(tokens) == 1:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            if first_failure:
                raise ValueError("Old configuration read failed")
        return {"connected": token == "rotated", "paths": {}}

    monkeypatch.setattr(claude, "configure", held)
    task = asyncio.create_task(runtime.claude_vscode_status())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await runtime.provider_manager.replace(
            runtime.settings.model_copy(update={"proxy_auth_token": "rotated"}),
            commit=AsyncMock(),
        )
        release.set()
        assert (await asyncio.wait_for(task, 5))["connected"] is True
        assert tokens == [initial_token, "rotated"]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close()
