"""Offline relocation and transactional schema initialization of FCC's database.

The application holds the shared owner lock until both features close. Connections
opened here never escape initialization, including when migration fails.
"""

import asyncio
import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

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


def _connect(path: Path, *, existing: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.as_uri() + ("?mode=rw" if existing else "?mode=rwc"),
        uri=True,
        timeout=10,
        autocommit=True,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
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
                self._owner.release()
                raise
            except BaseException:
                self._owner.release()
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

    def execute[T](
        self, operation: Callable[[sqlite3.Connection], T], *, write: bool = True
    ) -> T:
        with closing(_connect(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                result = operation(connection)
                connection.execute("COMMIT")
                return result
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    async def run[T](
        self, operation: Callable[[sqlite3.Connection], T], *, write: bool = True
    ) -> T:
        return await self.work(lambda: self.execute(operation, write=write))

    async def work[T](self, operation: Callable[[], T]) -> T:
        if not self._started or self._closing:
            raise sqlite3.OperationalError("FCC storage is closed")
        task = asyncio.create_task(run_sync_owned(operation))
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
            await run_sync_owned(self._owner.release)
