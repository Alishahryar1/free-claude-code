import asyncio
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Protocol, override

import httpx
import pytest
from anyio import to_thread
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from free_claude_code.application.code_sessions.models import CodeUnavailableError
from free_claude_code.runtime import sqlite_database
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from free_claude_code.runtime.sqlite_database import (
    initialize_database,
    query_rows,
    required_column,
)
from tests.assertions import require_present
from tests.fixture_types import DatabaseFactory


@pytest.mark.asyncio
async def test_transactions_reuse_connections_until_database_closes(
    database_factory: DatabaseFactory, tmp_path: Path
):
    database = database_factory(tmp_path / "fcc.db", tmp_path / "fcc.lock")
    await database.start()
    observed: list[sqlite3.Connection] = []

    def observe(connection: sqlite3.Connection):
        observed.append(connection)
        assert connection.in_transaction
        assert connection.autocommit is True
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        row = query_rows(connection, "SELECT 1 AS value").fetchone()
        assert row is not None
        return required_column(row, "value", int)

    for write in (True, True, False, False):
        assert await database.run(observe, write=write) == 1
    assert observed[0] is observed[1]
    assert observed[2] is observed[3]
    assert observed[0] is not observed[2]
    await database.close()
    for connection in observed:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def _deleted_count(connection: sqlite3.Connection) -> int:
    row = query_rows(connection, "SELECT count(*) FROM code_deleted").fetchone()
    assert row is not None
    return required_column(row, 0, int)


@pytest.mark.asyncio
async def test_queued_writers_leave_workers_available_and_drain_on_close(
    database_factory: DatabaseFactory, tmp_path: Path
):
    database = database_factory(tmp_path / "fcc.db", tmp_path / "fcc.lock")
    await database.start()
    entered, release = threading.Event(), threading.Event()
    limiter = to_thread.current_default_thread_limiter()
    previous_limit = limiter.total_tokens
    limiter.total_tokens = 2

    def write(connection: sqlite3.Connection, value: str):
        if value == "0":
            entered.set()
            assert release.wait(10)
        connection.execute("INSERT INTO code_deleted VALUES (?)", (value,)).close()
        return value

    writers = [asyncio.create_task(database.run(lambda c: write(c, "0")))]
    readers: list[asyncio.Task[int] | asyncio.Task[str]] = []
    closing_task = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        writers.extend(
            asyncio.create_task(database.run(lambda c, i=i: write(c, str(i))))
            for i in range(1, 12)
        )
        # Let each accepted call register its admission task before the probes.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        writers[-1].cancel()
        readers = [
            asyncio.create_task(
                database.run(
                    _deleted_count,
                    write=False,
                )
            ),
            asyncio.create_task(to_thread.run_sync(lambda: "available")),
        ]
        done, pending = await asyncio.wait(readers, timeout=3)
        assert not pending
        assert {task.result() for task in done} == {0, "available"}
        closing_task = asyncio.create_task(database.close())
        await asyncio.sleep(0)
        assert not closing_task.done()
        with pytest.raises(sqlite3.OperationalError, match="closed"):
            await database.run(lambda c: None)
        other = database_factory(database.path, tmp_path / "fcc.lock")
        with pytest.raises(sqlite3.OperationalError, match="another FCC"):
            await other.start()
    finally:
        release.set()
        outcomes = await asyncio.gather(*writers, *readers, return_exceptions=True)
        if closing_task is not None:
            await closing_task
        limiter.total_tokens = previous_limit
    assert outcomes[:12] == [str(i) for i in range(12)]
    reopened = database_factory(database.path, tmp_path / "fcc.lock")
    await reopened.start()
    assert (
        await reopened.run(
            _deleted_count,
            write=False,
        )
        == 12
    )


@pytest.mark.asyncio
async def test_reader_leases_are_bounded_and_can_move_between_threads(
    database_factory: DatabaseFactory, tmp_path: Path
):
    database = database_factory(tmp_path / "fcc.db", tmp_path / "fcc.lock")
    await database.start()
    four_readers, release = threading.Event(), threading.Event()
    observed: list[sqlite3.Connection] = []
    guard = threading.Lock()

    def read(connection: sqlite3.Connection):
        with guard:
            observed.append(connection)
            if len(observed) == 4:
                four_readers.set()
        assert release.wait(10)
        row = query_rows(connection, "SELECT 1").fetchone()
        assert row is not None
        return required_column(row, 0, int)

    tasks = [asyncio.create_task(database.run(read, write=False)) for _ in range(10)]
    try:
        assert await asyncio.to_thread(four_readers.wait, 3)
        assert len(observed) == 4
    finally:
        release.set()
        results = await asyncio.gather(*tasks)
    assert results == [1] * 10
    assert len(set(observed)) == 4
    # Force two different worker threads to reuse the single writer connection.
    seen = []

    def cross_threads():
        errors = []
        finished, release_thread = threading.Event(), threading.Event()

        def execute(wait: bool):
            try:
                database.execute(lambda c: seen.append(c))
            except BaseException as exc:
                errors.append(exc)
            finally:
                if wait:
                    finished.set()
                    assert release_thread.wait(5)

        thread = threading.Thread(target=execute, args=(True,))
        thread.start()
        try:
            assert finished.wait(3)
            execute(False)
        finally:
            release_thread.set()
            thread.join(3)
        assert not thread.is_alive()
        assert not errors

    await database.work(cross_threads)
    assert seen[0] is seen[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["begin", "callback", "commit", "rollback", "closed"]
)
async def test_failed_transactions_discard_the_connection(
    database_factory: DatabaseFactory, tmp_path: Path, failure: str
):
    database = database_factory(tmp_path / "fcc.db", tmp_path / "fcc.lock")
    await database.start()
    observed: list[sqlite3.Connection] = []

    def authorize(
        action: int, arg1: str | None, arg2: object, name: object, trigger: object
    ):
        denied = {"begin": "BEGIN", "commit": "COMMIT", "rollback": "ROLLBACK"}
        if action == sqlite3.SQLITE_TRANSACTION and arg1 == denied.get(failure):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def configure(connection: sqlite3.Connection):
        observed.append(connection)
        connection.set_authorizer(authorize)

    # Configure before BEGIN without wrapping/replacing sqlite3's transaction logic.
    def configure_lease():
        with closing(require_present(database._writer).connect()) as lease:
            configure(require_present(lease.driver_connection))

    await database.work(configure_lease)

    def fail(connection: sqlite3.Connection):
        connection.execute("INSERT INTO code_deleted VALUES ('failed')").close()
        if failure == "closed":
            connection.close()
        if failure in ("callback", "rollback", "closed"):
            raise ValueError("original callback failure")

    error_type = (
        ValueError
        if failure in ("callback", "rollback", "closed")
        else sqlite3.DatabaseError
    )
    with pytest.raises(error_type) as error:
        await database.run(fail)
    if failure in ("rollback", "closed"):
        assert isinstance(error.value.__cause__, sqlite3.Error)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        observed[0].execute("SELECT 1")

    def verify(connection: sqlite3.Connection):
        assert connection is not observed[0]
        assert (
            connection.execute("SELECT count(*) FROM code_deleted").fetchone()[0] == 0
        )
        connection.execute("INSERT INTO code_deleted VALUES ('success')").close()

    await database.run(verify)


@pytest.mark.asyncio
async def test_checkout_timeout_uses_sqlite_error_contract(
    database_factory: DatabaseFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database = database_factory(tmp_path / "fcc.db", tmp_path / "fcc.lock")
    await database.start()

    def timeout():
        raise PoolTimeoutError("injected capacity timeout")

    monkeypatch.setattr(database._writer, "connect", timeout)
    with pytest.raises(sqlite3.OperationalError, match="capacity") as error:
        await database.run(lambda c: pytest.fail("callback must not run"))
    assert isinstance(error.value.__cause__, PoolTimeoutError)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_startup_failure_disposes_connections_before_releasing_owner(
    tmp_path: Path, cancel: bool, initialized_database: Path
):
    shutil.copyfile(initialized_database, tmp_path / "fcc.db")
    entered, release = threading.Event(), threading.Event()
    observed: list[sqlite3.Connection] = []

    class GatedDatabase(sqlite_database.SQLiteDatabase):
        @override
        def _initialize(self):
            super()._initialize()
            self.execute(lambda c: observed.append(c))
            entered.set()
            assert release.wait(10)
            if not cancel:
                raise sqlite3.OperationalError("injected startup failure")

    database = GatedDatabase(tmp_path / "fcc.db", tmp_path / "fcc.lock")
    starting = asyncio.create_task(database.start())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if cancel:
            starting.cancel()
        competing = sqlite_database.SQLiteDatabase(database.path, tmp_path / "fcc.lock")
        with pytest.raises(sqlite3.OperationalError, match="another FCC"):
            await competing.start()
        await competing.close()
    finally:
        release.set()
    try:
        with pytest.raises(
            asyncio.CancelledError if cancel else sqlite3.OperationalError
        ):
            await starting
        assert observed[0] is not None
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            observed[0].execute("SELECT 1")
        reopened = sqlite_database.SQLiteDatabase(database.path, tmp_path / "fcc.lock")
        try:
            await reopened.start()
        finally:
            await reopened.close()
    finally:
        await database.close()


@pytest.fixture(scope="module")
def initialized_database(tmp_path_factory: pytest.TempPathFactory):
    """Build the schema once, outside the ownership test's synchronization window."""
    path = tmp_path_factory.mktemp("initialized-database") / "fcc.db"
    initialize_database(path)
    return path


def test_connection_setup_failure_closes_raw_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    connect = sqlite3.connect
    observed: list[sqlite3.Connection] = []

    class ParameterSequence(Protocol):
        def __len__(self) -> int: ...

        def __getitem__(self, index: int, /) -> object: ...

    class FailedSetup(sqlite3.Connection):
        @override
        def execute(
            self, sql: str, parameters: ParameterSequence | Mapping[str, object] = (), /
        ):
            if sql.startswith("PRAGMA foreign_keys"):
                raise sqlite3.OperationalError("injected configuration failure")
            return super().execute(sql, parameters)

    def create(
        database: str,
        *,
        uri: bool,
        timeout: float,
        autocommit: bool,
        check_same_thread: bool,
    ):
        connection = connect(
            database,
            uri=uri,
            timeout=timeout,
            autocommit=autocommit,
            check_same_thread=check_same_thread,
            factory=FailedSetup,
        )
        observed.append(connection)
        return connection

    monkeypatch.setattr(sqlite_database.sqlite3, "connect", create)
    with pytest.raises(sqlite3.OperationalError, match="configuration"):
        sqlite_database._connect(tmp_path / "fcc.db")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        observed[0].execute("SELECT 1")


@pytest.mark.asyncio
async def test_partial_pool_creation_cleans_up_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database = sqlite_database.SQLiteDatabase(
        tmp_path / "fcc.db", tmp_path / "fcc.lock"
    )
    create_pool = database._pool
    observed: list[sqlite3.Connection] = []

    def partial(size: int):
        if size == 4:
            raise RuntimeError("reader pool creation failed")
        pool = create_pool(size)
        with closing(pool.connect()) as lease:
            connection = lease.driver_connection
            assert isinstance(connection, sqlite3.Connection)
            observed.append(connection)
        return pool

    monkeypatch.setattr(database, "_pool", partial)
    try:
        with pytest.raises(RuntimeError, match="reader pool"):
            await database.start()
        assert observed[0] is not None
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            observed[0].execute("SELECT 1")
        other = sqlite_database.SQLiteDatabase(database.path, tmp_path / "fcc.lock")
        try:
            await other.start()
        finally:
            await other.close()
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_disposal_failure_retains_owner_and_can_be_retried(
    database_factory: DatabaseFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database = database_factory(tmp_path / "fcc.db", tmp_path / "fcc.lock")
    await database.start()
    reader_connections = []
    await database.run(lambda c: reader_connections.append(c), write=False)
    writer = database._writer
    dispose = require_present(writer).dispose

    def fail():
        raise RuntimeError("injected disposal failure")

    monkeypatch.setattr(writer, "dispose", fail)
    try:
        with pytest.raises(RuntimeError, match="disposal"):
            await database.close()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            reader_connections[0].execute("SELECT 1")
        competing = database_factory(database.path, tmp_path / "fcc.lock")
        with pytest.raises(sqlite3.OperationalError, match="another FCC"):
            await competing.start()
    finally:
        monkeypatch.setattr(writer, "dispose", dispose)
        await database.close()
    reopened = database_factory(database.path, tmp_path / "fcc.lock")
    await reopened.start()


def historical_database(path: Path, version: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = (Path(__file__).parent / "fixtures/code_schema_v0.sql").read_text()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript(schema)
        if version >= 1:
            connection.execute("DROP TABLE code_prompts")
            connection.execute(
                "CREATE TABLE code_prompts("
                "session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, "
                "id TEXT NOT NULL, generation TEXT NOT NULL, request_id TEXT NOT NULL, "
                "native_turn_id TEXT, native_item_id TEXT, kind TEXT NOT NULL, "
                "form TEXT NOT NULL, raw TEXT NOT NULL, "
                "status TEXT NOT NULL CHECK(status IN ('pending','answering','resolved','expired')), "
                "response_id TEXT, error TEXT, PRIMARY KEY(session_id,id), "
                "UNIQUE(session_id,generation,request_id), UNIQUE(session_id,response_id), "
                "FOREIGN KEY(session_id,id) REFERENCES code_items(session_id,id) ON DELETE CASCADE)"
            )
        if version >= 2:
            for table in ("code_sessions", "code_runs"):
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN mode TEXT NOT NULL DEFAULT 'config' "
                    "CHECK(mode IN ('config','ask','auto_review','full_access'))"
                )
            connection.execute(
                "ALTER TABLE code_sessions ADD COLUMN native_permission_defaults TEXT "
                "CHECK(native_permission_defaults IS NULL OR "
                "(json_valid(native_permission_defaults) AND json_type(native_permission_defaults) = 'object'))"
            )
        if version >= 3:
            connection.execute(
                "ALTER TABLE code_sessions ADD COLUMN context_used_tokens INTEGER "
                "CHECK(context_used_tokens IS NULL OR context_used_tokens >= 0)"
            )
        connection.execute(f"PRAGMA user_version={version}")
        connection.execute(
            "INSERT INTO code_sessions (id,cwd,model,harness,title,title_search,cwd_search,"
            "auto_title,native_thread_id,native_may_have_input,revision,status,created_at,updated_at) "
            "VALUES ('s','/work','provider/model','codex','Saved','saved','/work',0,'thread',1,7,'ready',100,200)"
        )
        connection.execute(
            "INSERT INTO code_runs (session_id,id,ordinal,text,model,status,submission_started,"
            "native_turn_id,stop_requested,error_details,created_at,finished_at) "
            "VALUES ('s','r',1,'User input','provider/model','completed',1,'turn',0,'{}',100,150)"
        )
        connection.execute(
            "INSERT INTO code_items VALUES ('s','i','r',1,'turn','native-item',"
            "'assistant','Title','Saved answer','Detail',1,'{\"payload\": true}')"
        )
        if version >= 1:
            connection.execute(
                "INSERT INTO code_items VALUES ('s','p','r',2,NULL,NULL,'prompt','','','',1,'{}')"
            )
        connection.execute(
            "INSERT INTO code_prompts VALUES ('s','p','g','42','turn',NULL,'questions',"
            "'{\"question\": \"Continue?\"}','{\"saved\": true}','resolved','response',NULL)"
        )
        connection.execute("INSERT INTO code_deleted VALUES ('deleted-session')")
        if version >= 2:
            connection.execute(
                "UPDATE code_sessions SET mode='full_access', native_permission_defaults='{\"original\": true}'"
            )
        if version >= 3:
            connection.execute("UPDATE code_sessions SET context_used_tokens=1234")


def snapshot(path: Path) -> dict[str, list[dict[str, object]]]:
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        return {
            table: [
                dict(row) for row in query_rows(connection, f"SELECT * FROM {table}")
            ]
            for table in (
                "code_sessions",
                "code_runs",
                "code_items",
                "code_prompts",
                "code_deleted",
            )
        }


@pytest.mark.parametrize("version", [0, 1, 2, 3])
def test_historical_database_moves_and_upgrades_without_losing_records(
    tmp_path: Path, version: int
):
    old, target = tmp_path / "code/code.db", tmp_path / "fcc.db"
    historical_database(old, version)
    before = snapshot(old)
    for _ in range(2):
        initialize_database(target, old)
        assert not old.exists()
        after = snapshot(target)
        for table, rows in before.items():
            for previous, saved in zip(
                rows, after[table], strict=table != "code_items" or version != 0
            ):
                assert {key: saved[key] for key in previous} == previous
        assert len(after["code_items"]) == 2
        assert after["code_items"][-1]["kind"] == "prompt"
        with closing(sqlite3.connect(target)) as connection:
            assert connection.execute("PRAGMA user_version").fetchone() == (5,)
            assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_fresh_database_uses_only_destination_and_is_repeatable(tmp_path: Path):
    target, old = tmp_path / "fcc.db", tmp_path / "code/code.db"
    initialize_database(target, old)
    initialize_database(target, old)
    assert not old.exists()
    assert all(not rows for rows in snapshot(target).values())


def test_two_databases_are_preserved_without_choosing_a_winner(tmp_path: Path):
    old, target = tmp_path / "code/code.db", tmp_path / "fcc.db"
    historical_database(old, 1)
    historical_database(target, 3)
    before = [path.read_bytes() for path in (old, target)]
    with pytest.raises(sqlite3.DatabaseError, match="both"):
        initialize_database(target, old)
    assert [path.read_bytes() for path in (old, target)] == before


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
@pytest.mark.parametrize("location", ["fcc.db", "code/code.db"])
def test_orphan_journals_are_not_treated_as_a_fresh_database(
    tmp_path: Path, suffix: str, location: str
):
    orphan = tmp_path / (location + suffix)
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"preserve")
    with pytest.raises(sqlite3.DatabaseError, match="journal"):
        initialize_database(tmp_path / "fcc.db", tmp_path / "code/code.db")
    assert orphan.read_bytes() == b"preserve"
    assert not (tmp_path / "fcc.db").exists()


@pytest.mark.parametrize("version", [-1, 6])
def test_unsupported_version_is_not_moved_or_mutated(tmp_path: Path, version: int):
    old, target = tmp_path / "code/code.db", tmp_path / "fcc.db"
    historical_database(old, 3)
    with closing(sqlite3.connect(old)) as connection:
        connection.execute(f"PRAGMA user_version={version}")
    before = old.read_bytes()
    with pytest.raises(sqlite3.DatabaseError, match="version"):
        initialize_database(target, old)
    assert old.read_bytes() == before
    assert not target.exists()


@pytest.mark.parametrize(
    "statement", ["CREATE TABLE unrelated(id)", "CREATE TABLE code_sessions(id)"]
)
def test_unrecognized_schema_is_not_blessed_or_moved(tmp_path: Path, statement: str):
    old, target = tmp_path / "code.db", tmp_path / "fcc.db"
    with closing(sqlite3.connect(old)) as connection:
        connection.execute(statement)
    with pytest.raises(sqlite3.DatabaseError, match="schema"):
        initialize_database(target, old)
    assert old.exists() and not target.exists()


def test_committed_wal_survives_unclean_exit_and_relocation(tmp_path: Path):
    old, target = tmp_path / "code/code.db", tmp_path / "fcc.db"
    historical_database(old, 3)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os, sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.execute('PRAGMA journal_mode=WAL')
c.execute("UPDATE code_items SET text='Committed in WAL' WHERE id='i'")
c.commit()
os._exit(0)
""",
            str(old),
        ],
        check=True,
        timeout=10,
    )
    assert Path(f"{old}-wal").stat().st_size > 0
    initialize_database(target, old)
    assert snapshot(target)["code_items"][0]["text"] == "Committed in WAL"
    assert not any(old.parent.glob("code.db*"))


def test_failed_rename_preserves_source_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    old, target = tmp_path / "code/code.db", tmp_path / "fcc.db"
    historical_database(old, 3)
    before = snapshot(old)
    with monkeypatch.context() as patch:

        def fail_rename(self: object, target: object):
            raise PermissionError("move blocked")

        patch.setattr(Path, "rename", fail_rename)
        with pytest.raises(PermissionError):
            initialize_database(target, old)
    assert old.exists() and not target.exists()
    assert snapshot(old) == before
    initialize_database(target, old)
    assert snapshot(target) == before


@pytest.mark.parametrize("failing_version", [1, 2])
def test_migration_failure_after_move_rolls_back_and_retries_at_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_version: int
):
    old, target = tmp_path / "code/code.db", tmp_path / "fcc.db"
    historical_database(old, 0)
    migrations = sqlite_database.MIGRATIONS
    original = migrations[failing_version - 1][1]

    def fail_upgrade(connection: sqlite3.Connection):
        original(connection)
        connection.execute("CREATE TABLE must_rollback(id)")
        raise sqlite3.DatabaseError("injected migration failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            sqlite_database,
            "MIGRATIONS",
            tuple(
                (version, fail_upgrade if version == failing_version else upgrade)
                for version, upgrade in migrations
            ),
        )
        with pytest.raises(sqlite3.DatabaseError, match="injected"):
            initialize_database(target, old)
    assert target.exists() and not old.exists()
    with closing(sqlite3.connect(target)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (
            failing_version - 1,
        )
        assert not connection.execute(
            "SELECT name FROM sqlite_schema WHERE name='must_rollback'"
        ).fetchall()
        table_info: list[tuple[int, str, str, int, str | None, int]] = (
            connection.execute("PRAGMA table_info(code_sessions)").fetchall()
        )
        columns = {row[1] for row in table_info}
        assert "mode" not in columns
        assert (
            connection.execute("SELECT count(*) FROM code_items").fetchone()[0]
            == failing_version
        )
    initialize_database(target, old)
    assert len(snapshot(target)["code_items"]) == 2


def test_fresh_schema_failure_rolls_back_all_ddl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "fcc.db"
    original = sqlite_database.MIGRATIONS[0][1]

    def fail_upgrade(connection: sqlite3.Connection):
        original(connection)
        raise sqlite3.DatabaseError("injected fresh failure")

    with monkeypatch.context() as patch:
        patch.setattr(sqlite_database, "MIGRATIONS", ((1, fail_upgrade),))
        with pytest.raises(sqlite3.DatabaseError, match="injected"):
            initialize_database(target)
    with closing(sqlite3.connect(target)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert not connection.execute("SELECT name FROM sqlite_schema").fetchall()
    initialize_database(target)
    assert all(not rows for rows in snapshot(target).values())


def test_current_schema_does_not_execute_historical_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "fcc.db"
    historical_database(target, 3)
    initialize_database(target)

    def already_applied(connection: object):
        pytest.fail("An applied migration was executed again")

    monkeypatch.setattr(
        sqlite_database,
        "MIGRATIONS",
        tuple((version, already_applied) for version, _ in sqlite_database.MIGRATIONS),
    )
    initialize_database(target)


def test_messaging_schema_failure_rolls_back_without_changing_code_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "fcc.db"
    historical_database(target, 3)
    before = snapshot(target)
    migrations = sqlite_database.MIGRATIONS

    def fail(connection: sqlite3.Connection):
        migrations[3][1](connection)
        raise sqlite3.DatabaseError("interrupted messaging schema")

    with monkeypatch.context() as patch:
        patch.setattr(
            sqlite_database,
            "MIGRATIONS",
            tuple(
                (version, fail if version == 4 else upgrade)
                for version, upgrade in migrations
            ),
        )
        with pytest.raises(sqlite3.DatabaseError, match="interrupted"):
            initialize_database(target)
    assert snapshot(target) == before
    with closing(sqlite3.connect(target)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (3,)
        assert not connection.execute(
            "SELECT name FROM sqlite_schema WHERE name LIKE 'messaging_%'"
        ).fetchall()
    initialize_database(target)
    assert snapshot(target) == before


def test_busy_wal_is_not_renamed_and_retries_after_reader_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    old, target = tmp_path / "code/code.db", tmp_path / "fcc.db"
    historical_database(old, 3)
    connect = sqlite_database._connect

    def no_wait(path: Path, *, existing: bool = False, check_same_thread: bool = True):
        connection = connect(
            path, existing=existing, check_same_thread=check_same_thread
        )
        connection.execute("PRAGMA busy_timeout=0")
        return connection

    monkeypatch.setattr(sqlite_database, "_connect", no_wait)
    with (
        closing(sqlite3.connect(old)) as writer,
        closing(sqlite3.connect(old)) as reader,
    ):
        writer.execute("PRAGMA journal_mode=WAL")
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM code_items").fetchall()
        writer.execute("UPDATE code_items SET text='newest committed' WHERE id='i'")
        writer.commit()
        with pytest.raises(sqlite3.DatabaseError, match="checkpoint is busy"):
            initialize_database(target, old)
        assert old.exists() and not target.exists()
    initialize_database(target, old)
    assert snapshot(target)["code_items"][0]["text"] == "newest committed"


@pytest.mark.parametrize("redirect", ["directory", "database"])
def test_redirected_legacy_storage_is_not_moved(tmp_path: Path, redirect: str):
    elsewhere = tmp_path / "elsewhere/code.db"
    historical_database(elsewhere, 3)
    old, target = tmp_path / "code/code.db", tmp_path / "fcc.db"
    try:
        if redirect == "directory":
            old.parent.symlink_to(elsewhere.parent, target_is_directory=True)
        else:
            old.parent.mkdir()
            old.symlink_to(elsewhere)
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    before = elsewhere.read_bytes()
    with pytest.raises(sqlite3.DatabaseError, match="redirected"):
        initialize_database(target, old)
    assert elsewhere.read_bytes() == before
    assert not target.exists()


@pytest.mark.asyncio
async def test_restart_recovery_runs_after_current_schema_relocation(
    database_factory: DatabaseFactory, tmp_path: Path
):
    old, target, lock = (
        tmp_path / "code/code.db",
        tmp_path / "fcc.db",
        tmp_path / "code/code.lock",
    )
    historical_database(old, 3)
    for iteration in range(2):
        current = old if iteration == 0 else target
        with closing(sqlite3.connect(current)) as connection, connection:
            connection.execute(
                "UPDATE code_runs SET status='running', finished_at=NULL"
            )
            connection.execute("UPDATE code_prompts SET status='answering'")
        store = SQLiteCodeStore(database_factory(target, lock, legacy_path=old))
        try:
            await store.start()
            run = await store.get_run("s", "r")
            assert run is not None
            assert run.status == "interrupted" and run.finished_at is not None
            assert (await store.prompts("s"))[0].status == "expired"
        finally:
            await store.close()
            await store.database.close()


@pytest.mark.asyncio
async def test_old_owner_blocks_move_and_new_owner_uses_same_lock(
    database_factory: DatabaseFactory, tmp_path: Path
):
    old, target, lock = (
        tmp_path / "code/code.db",
        tmp_path / "fcc.db",
        tmp_path / "code/code.lock",
    )
    historical_database(old, 3)
    first = SQLiteCodeStore(database_factory(old, lock))
    second = SQLiteCodeStore(database_factory(target, lock, legacy_path=old))
    await first.start()
    try:
        with pytest.raises(CodeUnavailableError, match="another FCC"):
            await second.start()
        assert old.exists() and not target.exists()
    finally:
        await first.close()
        await first.database.close()
    second = SQLiteCodeStore(database_factory(target, lock, legacy_path=old))
    await second.start()
    try:
        assert (await second.get_session("s")).title == "Saved"
        assert lock.exists() and not old.exists()
    finally:
        await second.close()
        await second.database.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission contract")
def test_relocated_database_keeps_private_permissions(tmp_path: Path):
    old, target = tmp_path / "code/code.db", tmp_path / "fcc.db"
    historical_database(old, 3)
    old.chmod(0o600)
    initialize_database(target, old)
    assert target.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_migration", [False, True])
async def test_production_startup_keeps_http_available_and_chat_cleanup_independent(
    monkeypatch: pytest.MonkeyPatch, fail_migration: bool
):
    from free_claude_code.config import paths
    from free_claude_code.config.settings import Settings
    from free_claude_code.core.interprocess_lock import InterprocessFileLock
    from free_claude_code.runtime import bootstrap
    from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
    from free_claude_code.runtime.retired_chat import remove_retired_chat_history

    old, target = paths.legacy_code_database_path(), paths.fcc_database_path()
    historical_database(old, 3)
    chat = paths.config_dir_path() / "chat/chat.db"
    chat.parent.mkdir()
    chat.write_bytes(b"retired history")
    chat_lock = InterprocessFileLock(chat.parent / "chat.lock")
    assert chat_lock.acquire()
    entered, release = threading.Event(), threading.Event()

    def gated_migration(*args: Path):
        entered.set()
        assert release.wait(10)
        if fail_migration:
            raise sqlite3.DatabaseError("injected migration failure")
        initialize_database(*args)

    monkeypatch.setattr(sqlite_database, "initialize_database", gated_migration)

    def replace_configure_logging(*args: object, **kwargs: object):
        return None

    monkeypatch.setattr(bootstrap, "configure_logging", replace_configure_logging)

    def replace_start_model_list_refresh(_: object):
        return None

    monkeypatch.setattr(
        ProviderRuntimeManager,
        "start_model_list_refresh",
        replace_start_model_list_refresh,
    )
    app = bootstrap.build_asgi_app(
        Settings().model_copy(update={"messaging_platform": "none"})
    )
    code = app.runtime._code_service
    assert code is not None
    try:
        await asyncio.wait_for(app.runtime.start(), 3)
        assert await asyncio.to_thread(entered.wait, 3)
        assert code.storage_status()["state"] == "starting"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            assert (await client.get("/health")).status_code == 200
            release.set()
            await asyncio.wait_for(asyncio.gather(*app.runtime._startup_tasks), 5)
            assert code.storage_status()["state"] == (
                "failed" if fail_migration else "ready"
            )
            assert (await client.get("/health")).status_code == 200
        assert chat.exists()
        assert old.exists() == fail_migration
        assert target.exists() != fail_migration
    finally:
        release.set()
        chat_lock.release()
        await app.runtime.close()
    remove_retired_chat_history()
    assert not chat.exists()
    assert (old if fail_migration else target).exists()
