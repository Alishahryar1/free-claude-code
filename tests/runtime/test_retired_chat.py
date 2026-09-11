import asyncio
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from free_claude_code.config import paths
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.core.interprocess_lock import InterprocessFileLock
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.integrations.discovery import LocalInstallations
from free_claude_code.runtime.integrations.service import IntegrationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from free_claude_code.runtime.retired_chat import remove_retired_chat_history


@pytest.fixture(autouse=True)
def _offline_discovery(monkeypatch):
    monkeypatch.setattr(
        ProviderRuntimeManager, "warm_referenced_model_cache", AsyncMock()
    )
    monkeypatch.setattr(
        ProviderRuntimeManager, "start_model_list_refresh", lambda _: None
    )


@pytest.fixture
def cleanup_logs():
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        yield messages
    finally:
        logger.remove(sink)


def _runtime():
    store = ManagedConfigStore()
    store.initialize()
    manager = ProviderRuntimeManager(
        store.read().settings,
        runtime_factory=lambda snapshot: ProviderRuntime(snapshot, {}),
    )
    return ApplicationRuntime(
        manager,
        integrations=IntegrationService(LocalInstallations.current()),
        configuration=ConfigurationService(store),
        transcriber=None,
    )


async def _start_and_close(runtime=None):
    runtime = runtime or _runtime()
    try:
        await runtime.start()
    finally:
        assert await runtime.close()


def _history():
    directory = paths.config_dir_path() / "chat"
    directory.mkdir(parents=True)
    database = directory / "chat.db"
    with closing(sqlite3.connect(database)) as connection, connection:
        for name in (
            "chat_settings",
            "chat_sessions",
            "chat_turns",
            "chat_generations",
            "chat_generation_segments",
            "chat_compactions",
        ):
            connection.execute(f"CREATE TABLE {name} (saved TEXT)")
            connection.execute(f"INSERT INTO {name} VALUES (?)", ("old private data",))
    owned = [database]
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = directory / f"chat.db{suffix}"
        sidecar.write_bytes(b"old private sidecar")
        owned.append(sidecar)
    return directory, owned


@pytest.mark.asyncio
async def test_startup_removes_retired_history_and_preserves_other_files():
    runtime = _runtime()
    directory, owned = _history()
    root = paths.config_dir_path()
    preserved = {root / ".env": (root / ".env").read_bytes()}
    for relative in (
        "code/code.db",
        "code/code.db-wal",
        "messaging/sessions.json",
        "accounts.json",
        "chat/notes.txt",
        "chat/chat.db.backup",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep this file")
        preserved[path] = path.read_bytes()
    await _start_and_close(runtime)
    assert not [path for path in owned if path.exists()]
    assert {path: path.read_bytes() for path in preserved} == preserved
    assert (directory / "chat.lock").is_file()
    await _start_and_close()
    assert not [path for path in owned if path.exists()]


@pytest.mark.asyncio
async def test_fresh_and_repeated_startup_do_not_create_chat_directory():
    await _start_and_close()
    await _start_and_close()
    assert not (paths.config_dir_path() / "chat").exists()


def test_startup_removes_history_on_uvloop():
    uvloop = pytest.importorskip("uvloop")
    _, owned = _history()
    asyncio.run(_start_and_close(), loop_factory=uvloop.new_event_loop)
    assert not [path for path in owned if path.exists()]


@pytest.mark.asyncio
async def test_busy_history_is_left_intact_and_removed_on_next_startup(cleanup_logs):
    directory, owned = _history()
    original = {path: path.read_bytes() for path in owned}
    lock = InterprocessFileLock(directory / "chat.lock")
    assert lock.acquire()
    try:
        await _start_and_close()
        assert {path: path.read_bytes() for path in owned} == original
        assert any(
            "cleanup deferred: history is in use" in entry for entry in cleanup_logs
        )
    finally:
        lock.release()
    await _start_and_close()
    assert not [path for path in owned if path.exists()]


@pytest.mark.asyncio
async def test_partial_deletion_releases_lock_and_retries(monkeypatch, cleanup_logs):
    directory, owned = _history()
    unlink = Path.unlink

    def denied_sidecar(path, *args, **kwargs):
        if path == owned[1]:
            raise PermissionError("file in use")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", denied_sidecar)
        await _start_and_close()
    assert not owned[0].exists()
    assert all(path.exists() for path in owned[1:])
    assert any(
        "cleanup deferred until next startup: exc_type=PermissionError" in entry
        for entry in cleanup_logs
    )
    probe = InterprocessFileLock(directory / "chat.lock")
    try:
        assert probe.acquire()
    finally:
        probe.release()
    await _start_and_close()
    assert not [path for path in owned if path.exists()]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["resolve", "lstat", "open"])
async def test_inaccessible_history_does_not_block_startup(
    monkeypatch, cleanup_logs, operation
):
    directory, owned = _history()
    original = getattr(Path, operation)
    denied = {"resolve": directory, "lstat": owned[0], "open": directory / "chat.lock"}[
        operation
    ]

    def inaccessible(path, *args, **kwargs):
        if path == denied:
            raise PermissionError("access denied")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, operation, inaccessible)
        await _start_and_close()
    assert all(path.exists() for path in owned)
    assert any("cleanup deferred" in entry for entry in cleanup_logs)
    await _start_and_close()
    assert not [path for path in owned if path.exists()]


@pytest.mark.asyncio
async def test_redirected_directory_is_left_untouched(tmp_path, cleanup_logs):
    root = paths.config_dir_path()
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    database = outside / "chat.db"
    database.write_bytes(b"not owned by FCC")
    try:
        (root / "chat").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks unavailable: {exc}")
    await _start_and_close()
    assert database.read_bytes() == b"not owned by FCC"
    assert list(outside.iterdir()) == [database]
    assert any(
        "cleanup deferred: redirected directory" in entry for entry in cleanup_logs
    )


def test_empty_chat_directory_does_not_create_a_lock():
    directory = paths.config_dir_path() / "chat"
    directory.mkdir(parents=True)
    remove_retired_chat_history()
    assert list(directory.iterdir()) == []


def test_disappearing_files_are_safe_and_deletion_holds_the_lock(monkeypatch):
    directory, owned = _history()
    unlink = Path.unlink
    removed = []

    def disappearing(path, *args, **kwargs):
        if path in owned:
            probe = InterprocessFileLock(directory / "chat.lock")
            try:
                assert not probe.acquire()
            finally:
                probe.release()
            removed.append(path)
            unlink(path)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", disappearing)
    remove_retired_chat_history()
    assert removed == owned
    assert list(directory.iterdir()) == [directory / "chat.lock"]


def test_concurrent_cleanups_leave_no_history():
    directory, owned = _history()
    ready = threading.Barrier(4)

    def cleanup(_):
        ready.wait(timeout=5)
        remove_retired_chat_history()

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(cleanup, range(4)))
    assert not [path for path in owned if path.exists()]
    assert list(directory.iterdir()) == [directory / "chat.lock"]


@pytest.mark.asyncio
async def test_startup_cancellation_waits_for_cleanup_and_lock_release(monkeypatch):
    directory, owned = _history()
    runtime = _runtime()
    entered, release = threading.Event(), threading.Event()
    unlink = Path.unlink

    def held_unlink(path, *args, **kwargs):
        if path == owned[0]:
            entered.set()
            assert release.wait(5), "cleanup worker was not released"
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", held_unlink)
    startup = asyncio.create_task(runtime.start())
    probe = InterprocessFileLock(directory / "chat.lock")
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        startup.cancel()
        await asyncio.sleep(0)
        assert not startup.done()
        assert not runtime.is_closed
        assert not probe.acquire()
    finally:
        release.set()
        await asyncio.gather(startup, return_exceptions=True)
        probe.release()
        await runtime.close()
    assert startup.cancelled()
    assert runtime.is_closed
    assert not [path for path in owned if path.exists()]
    try:
        assert probe.acquire()
    finally:
        probe.release()
