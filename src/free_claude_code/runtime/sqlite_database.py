"""Connection, transaction, and lifetime ownership of FCC's shared database.

The application holds the owner lock until both features close and pooled
connections are disposed. Each transaction exclusively leases one connection.
"""

import asyncio
import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import cast

from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.pool import QueuePool

from free_claude_code.core.async_tasks import run_sync_owned
from free_claude_code.core.interprocess_lock import InterprocessFileLock

from .sqlite_migrations import MIGRATIONS

# Historical columns identify supported files without depending on live models.
_BASE_COLUMNS = {
    "code_sessions": "id cwd model reasoning_effort harness title title_search cwd_search "
    "auto_title native_thread_id native_may_have_input revision status error created_at updated_at",
    "code_runs": "session_id id ordinal text model reasoning_effort status submission_started "
    "native_turn_id stop_requested error error_details created_at finished_at",
    "code_items": "session_id id run_id sequence native_turn_id native_item_id kind title text detail complete raw",
    "code_prompts": "session_id id generation request_id native_turn_id native_item_id kind form raw status response_id error",
    "code_deleted": "id",
}
_SIDECARS = ("-wal", "-shm", "-journal")
_MESSAGING_COLUMNS = {
    "messaging_trees": "platform chat_id root_id",
    "messaging_nodes": "platform chat_id root_id node_id parent_id parent_reference_id status_message_id session_id state",
    "messaging_references": "platform chat_id reference_id node_id kind",
    "messaging_managed_messages": "sequence platform chat_id message_id ts direction kind",
    "messaging_legacy_import": "source outcome trees messages skipped cleanup_pending",
}


def _connect(
    path: Path, *, existing: bool = False, check_same_thread: bool = True
) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.as_uri() + ("?mode=rw" if existing else "?mode=rwc"),
        uri=True,
        timeout=10,
        autocommit=True,
        check_same_thread=check_same_thread,
    )
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON").close()
    except BaseException:
        connection.close()
        raise
    return connection


def _schema_version(connection: sqlite3.Connection) -> int:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if not 0 <= version <= MIGRATIONS[-1][0]:
        raise sqlite3.DatabaseError(f"Unsupported FCC database version {version}.")
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'"
        )
    }
    if version == 0 and not tables:
        return version
    columns = _BASE_COLUMNS | (_MESSAGING_COLUMNS if version >= 4 else {})
    if tables != columns.keys():
        raise sqlite3.DatabaseError(
            "Unrecognized FCC database schema. Saved data was preserved."
        )
    for table, baseline in columns.items():
        expected = set(baseline.split())
        if version >= 2 and table in ("code_sessions", "code_runs"):
            expected.add("mode")
        if version >= 2 and table == "code_sessions":
            expected.add("native_permission_defaults")
        if version >= 3 and table == "code_sessions":
            expected.add("context_used_tokens")
        actual = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if not expected <= actual:
            raise sqlite3.DatabaseError(f"Incomplete FCC database schema in {table}.")
    return version


def _sidecars(path: Path) -> bool:
    return any(
        Path(f"{path}{suffix}").exists() or Path(f"{path}{suffix}").is_symlink()
        for suffix in _SIDECARS
    )


def _check_path(path: Path) -> None:
    if path.is_symlink():
        raise sqlite3.DatabaseError(
            "FCC database path is redirected. Saved data was preserved."
        )
    if not path.exists() and _sidecars(path):
        raise sqlite3.DatabaseError(
            "FCC database has orphan journal files. Saved data was preserved."
        )


def _relocate(source: Path, destination: Path) -> None:
    with closing(_connect(source, existing=True)) as connection:
        _schema_version(connection)
        if connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal":
            busy, pages, checkpointed = connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            if busy or pages != checkpointed:
                raise sqlite3.DatabaseError(
                    "FCC database checkpoint is busy. Stop FCC before updating."
                )
        if connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
            raise sqlite3.DatabaseError(
                "FCC database journals are still in use. Stop FCC before updating."
            )
    if _sidecars(source):
        raise sqlite3.DatabaseError(
            "FCC database journal files remain. Saved data was preserved."
        )
    _check_path(destination)
    if destination.exists():
        raise sqlite3.DatabaseError(
            "FCC has both legacy and current databases. Resolve the file conflict before starting Code."
        )
    source.rename(destination)


def initialize_database(path: Path, legacy_path: Path | None = None) -> None:
    """Apply pending migrations, relocating the legacy file first if present."""
    if tuple(version for version, _ in MIGRATIONS) != tuple(
        range(1, len(MIGRATIONS) + 1)
    ):
        raise sqlite3.DatabaseError(
            "FCC database migration versions must be contiguous."
        )
    path = path.parent.resolve() / path.name
    _check_path(path)
    if legacy_path is not None:
        legacy_path = (
            legacy_path.parent.parent.resolve()
            / legacy_path.parent.name
            / legacy_path.name
        )
        if legacy_path.parent.resolve() != legacy_path.parent:
            raise sqlite3.DatabaseError(
                "Legacy Code directory is redirected. Saved data was preserved."
            )
        _check_path(legacy_path)
        if legacy_path.exists() and path.exists():
            raise sqlite3.DatabaseError(
                "FCC has both legacy and current databases. Resolve the file conflict before starting Code."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if legacy_path is not None and legacy_path.exists():
        _relocate(legacy_path, path)
    with closing(_connect(path)) as connection:
        version = _schema_version(connection)
        if connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
            raise sqlite3.DatabaseError("FCC database could not enable WAL.")
        for next_version, upgrade in MIGRATIONS:
            if next_version <= version:
                continue
            connection.execute("BEGIN IMMEDIATE")
            try:
                upgrade(connection)
                if (
                    connection.execute("PRAGMA foreign_key_check").fetchone()
                    is not None
                ):
                    raise sqlite3.DatabaseError(
                        "FCC database has an invalid record link."
                    )
                connection.execute(f"PRAGMA user_version={next_version}")
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise


class SQLiteDatabase:
    """Application-owned initialization and lifetime for both durable features."""

    def __init__(
        self, path: Path, lock_path: Path, *, legacy_path: Path | None = None
    ) -> None:
        self.path = path
        self._legacy_path = legacy_path
        self._owner = InterprocessFileLock(lock_path)
        self._lifecycle = asyncio.Lock()
        self._started = False
        self._closing = False
        self._startup_error: Exception | None = None
        self._operations: set[asyncio.Task] = set()
        self._writer: QueuePool | None = None
        self._readers: QueuePool | None = None
        self._write_admission = asyncio.Semaphore(1)
        self._read_admission = asyncio.Semaphore(4)

    async def start(self) -> None:
        async with self._lifecycle:
            if self._started:
                return
            if self._closing:
                raise sqlite3.OperationalError("FCC database is closing")
            if self._startup_error is not None:
                raise self._startup_error
            try:
                await run_sync_owned(self._initialize)
            except Exception as exc:
                self._startup_error = exc
                await run_sync_owned(self._dispose)
                raise
            except BaseException:
                await run_sync_owned(self._dispose)
                raise
            self._started = True

    def _initialize(self) -> None:
        if not self._owner.acquire():
            raise sqlite3.OperationalError(
                "FCC storage is already owned by another FCC server"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            self.path.parent.chmod(0o700)
        initialize_database(self.path, self._legacy_path)
        if os.name != "nt":
            for path in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
                if path.exists():
                    path.chmod(0o600)
        self._writer = self._pool(1)
        self._readers = self._pool(4)

    def _pool(self, size: int) -> QueuePool:
        return QueuePool(
            lambda: cast(DBAPIConnection, _connect(self.path, check_same_thread=False)),
            pool_size=size,
            max_overflow=0,
            timeout=10,
            reset_on_return=None,
        )

    def _dispose(self) -> None:
        try:
            if self._writer is not None:
                self._writer.dispose()
                self._writer = None
        finally:
            if self._readers is not None:
                self._readers.dispose()
                self._readers = None
        self._owner.release()

    def execute[T](
        self, operation: Callable[[sqlite3.Connection], T], *, write: bool = True
    ) -> T:
        """Run a writer-pool transaction inside an already admitted worker.

        Callbacks consume the connection synchronously and return materialized
        results. They must not control the outer transaction, retain or close
        the connection, or change connection-wide settings.
        Legacy import uses this primitive for several transactions in one worker.
        """
        return self._transaction(self._writer, operation, write=write)

    def _transaction[T](
        self,
        pool: QueuePool | None,
        operation: Callable[[sqlite3.Connection], T],
        *,
        write: bool,
    ) -> T:
        if pool is None:
            raise sqlite3.OperationalError("FCC storage is closed")
        try:
            lease = pool.connect()
        except PoolTimeoutError as exc:
            raise sqlite3.OperationalError(
                "Timed out waiting for FCC storage connection capacity"
            ) from exc
        try:
            connection = lease.driver_connection
            if not isinstance(connection, sqlite3.Connection):
                raise sqlite3.OperationalError("Invalid FCC storage connection")
            if connection.in_transaction:
                raise sqlite3.OperationalError("FCC storage connection is not clean")
        except BaseException as exc:
            lease.invalidate(exc)
            lease.close()
            raise
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            result = operation(connection)
            connection.execute("COMMIT")
            return result
        except BaseException as exc:
            try:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                if connection.in_transaction:
                    raise sqlite3.OperationalError(
                        "FCC storage rollback did not finish"
                    )
            except BaseException as cleanup_error:
                raise exc from cleanup_error
            finally:
                # A traceback can retain a SELECT cursor after ROLLBACK. Never
                # let that cursor's snapshot reach the next borrower.
                lease.invalidate(exc)
            raise
        finally:
            lease.close()

    async def run[T](
        self, operation: Callable[[sqlite3.Connection], T], *, write: bool = True
    ) -> T:
        """Admit and supervise one transaction through committed result delivery."""
        if write:
            return await self.work(lambda: self.execute(operation))
        return await self._work(
            lambda: self._transaction(self._readers, operation, write=False),
            self._read_admission,
        )

    async def work[T](self, operation: Callable[[], T]) -> T:
        return await self._work(operation, self._write_admission)

    async def _work[T](
        self, operation: Callable[[], T], admission: asyncio.Semaphore
    ) -> T:
        if not self._started or self._closing:
            raise sqlite3.OperationalError("FCC storage is closed")

        async def admitted() -> T:
            async with admission:
                return await run_sync_owned(operation)

        task = asyncio.create_task(admitted())
        self._operations.add(task)
        try:
            # Deliver the result before cancellation can separate SQL commit from
            # the caller's in-memory publication/rollback.
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    if task.done():
                        return task.result()
        finally:
            self._operations.discard(task)

    async def close(self) -> None:
        async with self._lifecycle:
            self._closing = True
            if self._operations:
                drain = asyncio.gather(*self._operations, return_exceptions=True)
                while not drain.done():
                    try:
                        await asyncio.shield(drain)
                    except asyncio.CancelledError:
                        continue
            self._started = False
            await run_sync_owned(self._dispose)
