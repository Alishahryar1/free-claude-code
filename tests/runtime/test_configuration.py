import asyncio
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from free_claude_code.config import loader
from free_claude_code.config.admin.persistence import prepare_admin_update
from free_claude_code.config.env_files import FCC_CONFIG_SCHEMA_ENV
from free_claude_code.config.env_migrations import CONFIG_SCHEMA_VERSION
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.paths import config_lock_path
from free_claude_code.core.interprocess_lock import InterprocessFileLock
from free_claude_code.runtime.configuration import ConfigurationService


def test_reads_are_fresh_read_only_snapshots():
    store = ManagedConfigStore()
    store.initialize()
    for port in (8123, 8124):
        store.path.write_text(
            f"{FCC_CONFIG_SCHEMA_ENV}={CONFIG_SCHEMA_VERSION}\nPORT={port}\n"
        )
        with patch(
            "free_claude_code.config.loader.consolidate_managed_config",
            side_effect=AssertionError("read migrated"),
        ):
            snapshot = store.read()
        assert snapshot.settings.port == port
        assert snapshot.managed["PORT"] == str(port)


def test_read_does_not_recreate_missing_storage():
    store = ManagedConfigStore()
    with pytest.raises(ValueError, match="initializ"):
        store.read()
    assert not store.path.exists()


def test_preparation_uses_captured_disk_and_process_state(monkeypatch):
    store = ManagedConfigStore()
    store.initialize()
    snapshot = store.read({"PORT": "8123"})
    monkeypatch.setenv("PORT", "9999")
    store.path.unlink()
    prepared = prepare_admin_update({"PORT": "8124"}, snapshot, snapshot.settings)
    assert prepared.settings is not None
    assert prepared.settings.port == 8123
    assert "PORT" not in prepared.target_values


@pytest.mark.asyncio
async def test_read_waits_for_storage_lock_without_blocking_event_loop():
    store = ManagedConfigStore()
    store.initialize()
    service = ConfigurationService(store)
    with InterprocessFileLock(config_lock_path()):
        reader = asyncio.create_task(service.admin_values())
        done, _ = await asyncio.wait({reader}, timeout=0.1)
    try:
        assert not done
    finally:
        await asyncio.wait_for(reader, 5)


@pytest.mark.parametrize("schema", ["", "999"])
def test_reads_and_commits_fail_closed_on_uninitialized_or_newer_schema(schema):
    store = ManagedConfigStore()
    store.initialize()
    text = f"FCC_CONFIG_SCHEMA={schema}\nANTHROPIC_AUTH_TOKEN=preserve\n"
    store.path.write_text(text)
    with pytest.raises(ValueError, match="schema"):
        store.read()
    with pytest.raises(ValueError, match="schema"):
        store.commit({"FCC_CONFIG_SCHEMA": "1"})
    assert store.path.read_text() == text


def test_cooperative_commits_are_serialized_and_leave_complete_files():
    store = ManagedConfigStore()
    store.initialize()
    original = loader.atomic_write_managed_config
    writing = threading.Lock()
    seen = []

    def checked_write(values, **kwargs):
        probe = InterprocessFileLock(config_lock_path())
        try:
            assert not probe.acquire(), "commit does not hold the shared write lock"
        finally:
            probe.release()
        assert writing.acquire(blocking=False), "cooperative writes overlap"
        try:
            seen.append(values["PORT"])
            return original(values, **kwargs)
        finally:
            writing.release()

    def write(port):
        ManagedConfigStore().commit({"FCC_CONFIG_SCHEMA": "1", "PORT": str(port)})

    with (
        patch(
            "free_claude_code.config.loader.atomic_write_managed_config", checked_write
        ),
        ThreadPoolExecutor(max_workers=8) as executor,
    ):
        list(executor.map(write, range(8120, 8128)))
    assert len(seen) == 8
    assert store.read().managed["PORT"] == seen[-1]
    assert not list(store.path.parent.glob("*.tmp"))


def test_atomic_commit_waits_for_an_open_snapshot_reader(monkeypatch):
    store = ManagedConfigStore()
    store.initialize()
    store.commit({"FCC_CONFIG_SCHEMA": "1", "PORT": "8123"})
    opened, release = threading.Event(), threading.Event()
    original_open = io.open
    reader_thread = None

    def hold_reader(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path == store.path and threading.current_thread() is reader_thread:
            opened.set()
            if not release.wait(5):
                handle.close()
                raise TimeoutError("snapshot reader was not released")
        return handle

    def read_snapshot():
        nonlocal reader_thread
        reader_thread = threading.current_thread()
        return store.read({})

    monkeypatch.setattr(io, "open", hold_reader)
    with ThreadPoolExecutor(max_workers=2) as executor:
        reader = executor.submit(read_snapshot)
        writer = None
        try:
            assert opened.wait(5)
            writer = executor.submit(
                store.commit, {"FCC_CONFIG_SCHEMA": "1", "PORT": "8124"}
            )
            with pytest.raises(TimeoutError):
                writer.result(timeout=0.1)
        finally:
            release.set()
        assert reader.result(timeout=5).settings.port == 8123
        assert writer.result(timeout=5) is None
    assert store.read({}).settings.port == 8124
    assert not list(store.path.parent.glob("*.tmp"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["initialize", "admin_config", "admin_values", "prepare", "commit"]
)
async def test_storage_work_does_not_block_event_loop(operation):
    store = ManagedConfigStore()
    store.initialize()
    service = ConfigurationService(store)
    active = store.read().settings
    prepared = await service.prepare({"PORT": "8123"}, active)
    entered = threading.Event()
    release = threading.Event()
    heartbeat = asyncio.Event()
    method = {"admin_config": "read", "admin_values": "read", "prepare": "read"}.get(
        operation, operation
    )
    original = getattr(store, method)

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5), "event loop failed to release storage"
        assert heartbeat.is_set()
        return original(*args, **kwargs)

    async def observe():
        while not entered.is_set():
            await asyncio.sleep(0)
        heartbeat.set()
        release.set()

    observer = asyncio.create_task(observe())
    try:
        with patch.object(store, method, side_effect=blocked):
            if operation == "prepare":
                await service.prepare({"PORT": "8124"}, active)
            elif operation == "commit":
                await service.commit(prepared)
            else:
                await getattr(service, operation)()
        await asyncio.wait_for(observer, 5)
    finally:
        release.set()
        observer.cancel()
