"""Relational, transaction-owned state for FCC coding conversations."""

import asyncio
import json
import os
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path

import anyio.to_thread

from free_claude_code.application.code_sessions.models import (
    CodeConflictError,
    CodeItem,
    CodeItemPage,
    CodeNotFoundError,
    CodePage,
    CodePrompt,
    CodeRun,
    CodeSession,
    CodeUnavailableError,
    Record,
    now_ms,
)
from free_claude_code.core.interprocess_lock import InterprocessFileLock

_JSON_FIELDS = frozenset(
    {"raw", "form", "error_details", "request_id", "native_permission_defaults"}
)
_RUN_TRANSITIONS = {
    "preparing": {
        "preparing",
        "running",
        "stopping",
        "completed",
        "failed",
        "interrupted",
    },
    "running": {"running", "stopping", "completed", "failed", "interrupted"},
    "stopping": {"stopping", "completed", "failed", "interrupted"},
    "completed": {"completed"},
    "failed": {"failed"},
    "interrupted": {"interrupted"},
}
_PROMPT_TRANSITIONS = {
    "pending": {"pending", "answering", "resolved", "expired"},
    "answering": {"answering", "resolved", "expired"},
    "resolved": {"resolved"},
    "expired": {"expired"},
}


def _values(record: Record) -> dict[str, object]:
    values = record.model_dump()
    for key in values.keys() & _JSON_FIELDS:
        if values[key] is not None:
            values[key] = json.dumps(values[key], sort_keys=True, separators=(",", ":"))
    if isinstance(record, CodeSession):
        values.update(
            title_search=record.title.casefold(), cwd_search=record.cwd.casefold()
        )
    return values


def _record[T: Record](model: type[T], row: sqlite3.Row) -> T:
    values = {key: row[key] for key in model.model_fields}
    for key in values.keys() & _JSON_FIELDS:
        if values[key] is not None:
            values[key] = json.loads(values[key])
    return model.model_validate(values)


def _insert(connection: sqlite3.Connection, table: str, record: Record) -> None:
    values = _values(record)
    connection.execute(
        f"INSERT INTO {table} ({','.join(values)}) VALUES ({','.join('?' for _ in values)})",
        tuple(values.values()),
    )


def _update(
    connection: sqlite3.Connection,
    table: str,
    values: dict[str, object],
    where: str,
    parameters: tuple[object, ...],
) -> None:
    result = connection.execute(
        f"UPDATE {table} SET {','.join(f'{key} = ?' for key in values)} WHERE {where}",
        (*values.values(), *parameters),
    )
    if result.rowcount != 1:
        raise CodeConflictError("This Code session changed. Refresh it and try again.")


def _session(connection: sqlite3.Connection, session_id: str) -> CodeSession:
    row = connection.execute(
        "SELECT * FROM code_sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row is None:
        raise CodeNotFoundError("Code session not found.")
    return _record(CodeSession, row)


def _idle(connection: sqlite3.Connection, session_id: str) -> None:
    if (
        connection.execute(
            "SELECT 1 FROM code_runs WHERE session_id = ? AND status IN ('preparing','running','stopping')",
            (session_id,),
        ).fetchone()
        or connection.execute(
            "SELECT 1 FROM code_prompts WHERE session_id = ? AND status IN ('pending','answering')",
            (session_id,),
        ).fetchone()
    ):
        raise CodeConflictError("This session is busy. Your draft has been kept.")


def _write_session(
    connection: sqlite3.Connection,
    session: CodeSession,
    expected_revision: int,
    *,
    settings: bool,
) -> None:
    current = _session(connection, session.id)
    if current.revision != expected_revision or session.revision not in {
        expected_revision,
        expected_revision + 1,
    }:
        raise CodeConflictError(
            "This session changed. Refresh its state and try again."
        )
    if (
        not settings
        and current.native_permission_defaults is not None
        and session.native_permission_defaults != current.native_permission_defaults
    ):
        raise CodeConflictError("The original permission settings cannot be replaced.")
    fields = (
        {
            "title",
            "title_search",
            "auto_title",
            "model",
            "reasoning_effort",
            "mode",
            "revision",
            "updated_at",
        }
        if settings
        else {
            "native_thread_id",
            "native_permission_defaults",
            "native_may_have_input",
            "revision",
            "updated_at",
            "status",
            "error",
        }
    )
    _update(
        connection,
        "code_sessions",
        {key: value for key, value in _values(session).items() if key in fields},
        "id = ? AND revision = ?",
        (session.id, expected_revision),
    )


def _write_run(connection: sqlite3.Connection, run: CodeRun) -> None:
    row = connection.execute(
        "SELECT * FROM code_runs WHERE session_id = ? AND id = ?",
        (run.session_id, run.id),
    ).fetchone()
    if row is None:
        raise CodeNotFoundError("Code turn not found.")
    previous = _record(CodeRun, row)
    if (
        run.status not in _RUN_TRANSITIONS[previous.status]
        or (
            previous.status in {"completed", "failed", "interrupted"}
            and (
                run.finished_at != previous.finished_at
                or (previous.error is not None and run.error != previous.error)
            )
        )
        or (previous.stop_requested and not run.stop_requested)
        or (previous.submission_started and not run.submission_started)
        or any(
            getattr(previous, key) != getattr(run, key)
            for key in ("ordinal", "text", "model", "reasoning_effort", "mode")
        )
        or (
            previous.native_turn_id is not None
            and previous.native_turn_id != run.native_turn_id
        )
    ):
        raise CodeConflictError("This Code turn no longer accepts that update.")
    values = _values(run)
    for key in (
        "id",
        "session_id",
        "ordinal",
        "text",
        "model",
        "reasoning_effort",
        "mode",
        "created_at",
    ):
        values.pop(key)
    _update(
        connection,
        "code_runs",
        values,
        "session_id = ? AND id = ? AND status = ?",
        (run.session_id, run.id, previous.status),
    )


def _write_item(connection: sqlite3.Connection, item: CodeItem) -> None:
    row = connection.execute(
        "SELECT * FROM code_items WHERE session_id = ? AND id = ?",
        (item.session_id, item.id),
    ).fetchone()
    if row is None:
        _insert(connection, "code_items", item)
        return
    previous = _record(CodeItem, row)
    if (
        previous.run_id != item.run_id
        or previous.sequence != item.sequence
        or previous.kind != item.kind
        or (previous.complete and not item.complete)
        or any(
            getattr(previous, key) is not None
            and getattr(previous, key) != getattr(item, key)
            for key in ("native_turn_id", "native_item_id")
        )
    ):
        raise CodeConflictError("This transcript entry belongs to different work.")
    values = _values(item)
    for key in ("id", "session_id", "run_id", "sequence"):
        values.pop(key)
    _update(
        connection,
        "code_items",
        values,
        "session_id = ? AND id = ?",
        (item.session_id, item.id),
    )


def _write_prompt(connection: sqlite3.Connection, prompt: CodePrompt) -> None:
    item = connection.execute(
        "SELECT kind FROM code_items WHERE session_id = ? AND id = ?",
        (prompt.session_id, prompt.id),
    ).fetchone()
    if item is None or item["kind"] != "prompt":
        raise CodeConflictError("This prompt requires its own transcript entry.")
    row = connection.execute(
        "SELECT * FROM code_prompts WHERE session_id = ? AND id = ?",
        (prompt.session_id, prompt.id),
    ).fetchone()
    if row is None:
        _insert(connection, "code_prompts", prompt)
        return
    previous = _record(CodePrompt, row)
    if (
        prompt.status not in _PROMPT_TRANSITIONS[previous.status]
        or any(
            getattr(previous, key) != getattr(prompt, key)
            for key in (
                "generation",
                "request_id",
                "native_turn_id",
                "native_item_id",
                "kind",
            )
        )
        or (
            previous.response_id is not None
            and previous.response_id != prompt.response_id
        )
    ):
        raise CodeConflictError("This native prompt is no longer active.")
    values = _values(prompt)
    for key in ("id", "session_id"):
        values.pop(key)
    _update(
        connection,
        "code_prompts",
        values,
        "session_id = ? AND id = ? AND status = ?",
        (prompt.session_id, prompt.id, previous.status),
    )


def _migrate_prompt_entries(connection: sqlite3.Connection) -> None:
    """Give old prompts a permanent run-end position without rewriting history."""
    for row in connection.execute(
        "SELECT * FROM code_prompts ORDER BY rowid"
    ).fetchall():
        prompt = _record(CodePrompt, row)
        run = connection.execute(
            "SELECT id FROM code_runs WHERE session_id = ? AND native_turn_id = ?",
            (prompt.session_id, prompt.native_turn_id),
        ).fetchone()
        if run is None and prompt.native_item_id is not None:
            matches = connection.execute(
                "SELECT DISTINCT run_id AS id FROM code_items WHERE session_id = ? AND native_item_id = ?",
                (prompt.session_id, prompt.native_item_id),
            ).fetchall()
            if len(matches) == 1:
                run = matches[0]
        if run is None:
            run = connection.execute(
                "SELECT id FROM code_runs WHERE session_id = ? ORDER BY ordinal DESC LIMIT 1",
                (prompt.session_id,),
            ).fetchone()
        if run is None:
            raise CodeUnavailableError("A saved prompt has no saved conversation turn.")
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM code_items WHERE session_id = ?",
            (prompt.session_id,),
        ).fetchone()[0]
        _insert(
            connection,
            "code_items",
            CodeItem(
                id=prompt.id,
                session_id=prompt.session_id,
                run_id=run["id"],
                sequence=sequence,
                kind="prompt",
                complete=True,
            ),
        )
    connection.execute("CREATE TABLE code_prompts_new" + _PROMPT_COLUMNS)
    connection.execute("INSERT INTO code_prompts_new SELECT * FROM code_prompts")
    connection.execute("DROP TABLE code_prompts")
    connection.execute("ALTER TABLE code_prompts_new RENAME TO code_prompts")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise CodeUnavailableError("Code session history has an invalid record link.")
    connection.execute("PRAGMA user_version = 1")


class SQLiteCodeStore:
    def __init__(self, database_path: Path, lock_path: Path) -> None:
        self._path = database_path
        self._lock = InterprocessFileLock(lock_path)
        self._started = False
        self._lifecycle = asyncio.Lock()

    async def start(self) -> None:
        async with self._lifecycle:
            if self._started:
                return
            initialization = asyncio.create_task(
                anyio.to_thread.run_sync(self._initialize)
            )
            try:
                await asyncio.shield(initialization)
            except BaseException:
                await asyncio.gather(initialization, return_exceptions=True)
                self._lock.release()
                raise
            self._started = True

    def _initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._lock.acquire():
            raise CodeUnavailableError(
                "Code sessions is already owned by another FCC server."
            )
        if os.name != "nt":
            self._path.parent.chmod(0o700)
        self._execute(self._initialize_schema)
        if os.name != "nt":
            for path in (
                self._path,
                Path(f"{self._path}-wal"),
                Path(f"{self._path}-shm"),
            ):
                if path.exists():
                    path.chmod(0o600)

    async def close(self) -> None:
        async with self._lifecycle:
            self._started = False
            await anyio.to_thread.run_sync(self._lock.release)

    def _execute[T](self, operation: Callable[[sqlite3.Connection], T]) -> T:
        try:
            with closing(sqlite3.connect(self._path, timeout=10)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                with connection:
                    return operation(connection)
        except sqlite3.IntegrityError as exc:
            raise CodeConflictError(
                "This Code operation conflicts with existing session state."
            ) from exc
        except sqlite3.Error as exc:
            raise CodeUnavailableError("Code session storage is unavailable.") from exc

    async def _run[T](self, operation: Callable[[sqlite3.Connection], T]) -> T:
        if not self._started:
            raise CodeUnavailableError("Code session storage is closed.")
        return await anyio.to_thread.run_sync(self._execute, operation)

    def _initialize_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(_SCHEMA)
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute("PRAGMA user_version").fetchone()[0] == 0:
            _migrate_prompt_entries(connection)
        if connection.execute("PRAGMA user_version").fetchone()[0] == 1:
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
            connection.execute("PRAGMA user_version = 2")
        connection.execute(
            "UPDATE code_runs SET status = 'interrupted', finished_at = ?, error = ? "
            "WHERE status IN ('preparing','running','stopping')",
            (
                now_ms(),
                "FCC restarted before this turn finished. Its input was not resent.",
            ),
        )
        connection.execute(
            "UPDATE code_prompts SET status = 'expired' WHERE status IN ('pending','answering')"
        )

    async def create(self, session: CodeSession) -> CodeSession:
        def operation(connection: sqlite3.Connection) -> CodeSession:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM code_deleted WHERE id = ?", (session.id,)
            ).fetchone():
                raise CodeConflictError(
                    "This session was deleted. Create a new session."
                )
            row = connection.execute(
                "SELECT * FROM code_sessions WHERE id = ?", (session.id,)
            ).fetchone()
            if row:
                previous = _record(CodeSession, row)
                if previous.cwd != session.cwd or previous.harness != session.harness:
                    raise CodeConflictError(
                        "This session ID was already used for another folder."
                    )
                return previous
            _insert(connection, "code_sessions", session)
            return session

        return await self._run(operation)

    async def get_session(self, session_id: str) -> CodeSession:
        return await self._run(lambda connection: _session(connection, session_id))

    async def list_sessions(
        self, cursor: tuple[int, str] | None, limit: int, query: str = ""
    ) -> CodePage:
        def operation(connection: sqlite3.Connection) -> CodePage:
            clauses, parameters = [], []
            if query.strip():
                clauses.append(
                    "(instr(title_search, ?) > 0 OR instr(cwd_search, ?) > 0)"
                )
                parameters.extend([query.strip().casefold()] * 2)
            if cursor:
                clauses.append("(updated_at, id) < (?, ?)")
                parameters.extend(cursor)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = connection.execute(
                "SELECT * FROM code_sessions"
                + where
                + " ORDER BY updated_at DESC, id DESC LIMIT ?",
                (*parameters, limit + 1),
            ).fetchall()
            sessions = tuple(_record(CodeSession, row) for row in rows[:limit])
            last = sessions[-1] if sessions and len(rows) > limit else None
            return CodePage(sessions, (last.updated_at, last.id) if last else None)

        return await self._run(operation)

    async def pending_deletions(self) -> tuple[CodeSession, ...]:
        return await self._run(
            lambda connection: tuple(
                _record(CodeSession, row)
                for row in connection.execute(
                    "SELECT * FROM code_sessions WHERE status != 'ready'"
                )
            )
        )

    async def is_deleted(self, session_id: str) -> bool:
        return await self._run(
            lambda connection: (
                connection.execute(
                    "SELECT 1 FROM code_deleted WHERE id = ?", (session_id,)
                ).fetchone()
                is not None
            )
        )

    async def get_run(self, session_id: str, run_id: str) -> CodeRun | None:
        def operation(connection: sqlite3.Connection) -> CodeRun | None:
            row = connection.execute(
                "SELECT * FROM code_runs WHERE session_id = ? AND id = ?",
                (session_id, run_id),
            ).fetchone()
            return _record(CodeRun, row) if row else None

        return await self._run(operation)

    async def runs(self, session_id: str) -> tuple[CodeRun, ...]:
        return await self._run(
            lambda connection: tuple(
                _record(CodeRun, row)
                for row in connection.execute(
                    "SELECT * FROM code_runs WHERE session_id = ? ORDER BY ordinal",
                    (session_id,),
                )
            )
        )

    async def latest_run(self, session_id: str) -> CodeRun | None:
        def operation(connection: sqlite3.Connection) -> CodeRun | None:
            row = connection.execute(
                "SELECT * FROM code_runs WHERE session_id = ? ORDER BY ordinal DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            return _record(CodeRun, row) if row else None

        return await self._run(operation)

    async def items(
        self, session_id: str, before: tuple[int, int] | None, limit: int | None
    ) -> tuple[CodeItem, ...]:
        return (await self.item_page(session_id, before, limit)).items

    async def item_page(
        self, session_id: str, before: tuple[int, int] | None, limit: int | None
    ) -> CodeItemPage:
        def operation(connection: sqlite3.Connection) -> CodeItemPage:
            connection.execute("BEGIN")
            parameters: list[object] = [session_id]
            query = "SELECT i.*, r.ordinal AS run_ordinal FROM code_items i JOIN code_runs r ON (r.session_id, r.id) = (i.session_id, i.run_id) WHERE i.session_id = ?"
            if before:
                query += " AND (r.ordinal, i.sequence) < (?, ?)"
                parameters.extend(before)
            query += " ORDER BY r.ordinal DESC, i.sequence DESC"
            if limit is not None:
                query += " LIMIT ?"
                parameters.append(limit + 1)
            rows = connection.execute(query, parameters).fetchall()
            selected = rows if limit is None else rows[:limit]
            run_ids = {row["run_id"] for row in selected}
            runs = tuple(
                _record(CodeRun, row)
                for row in connection.execute(
                    "SELECT * FROM code_runs WHERE session_id = ? ORDER BY ordinal",
                    (session_id,),
                )
                if row["id"] in run_ids
            )
            last = (
                selected[-1]
                if selected and limit is not None and len(rows) > limit
                else None
            )
            return CodeItemPage(
                tuple(_record(CodeItem, row) for row in reversed(selected)),
                runs,
                (last["run_ordinal"], last["sequence"]) if last else None,
            )

        return await self._run(operation)

    async def prompts(self, session_id: str) -> tuple[CodePrompt, ...]:
        return await self._run(
            lambda connection: tuple(
                _record(CodePrompt, row)
                for row in connection.execute(
                    "SELECT * FROM code_prompts WHERE session_id = ? ORDER BY id",
                    (session_id,),
                )
            )
        )

    async def update_settings(
        self, session: CodeSession, expected_revision: int
    ) -> CodeSession:
        def operation(connection: sqlite3.Connection) -> CodeSession:
            connection.execute("BEGIN IMMEDIATE")
            previous = _session(connection, session.id)
            if previous.status != "ready":
                raise CodeConflictError("This session is being deleted.")
            if (previous.model, previous.reasoning_effort, previous.mode) != (
                session.model,
                session.reasoning_effort,
                session.mode,
            ):
                _idle(connection, session.id)
            _write_session(connection, session, expected_revision, settings=True)
            return _session(connection, session.id)

        return await self._run(operation)

    async def admit_run(
        self, session: CodeSession, run: CodeRun, item: CodeItem, expected_revision: int
    ) -> tuple[CodeSession, CodeRun]:
        def operation(connection: sqlite3.Connection) -> tuple[CodeSession, CodeRun]:
            connection.execute("BEGIN IMMEDIATE")
            previous = _session(connection, session.id)
            row = connection.execute(
                "SELECT * FROM code_runs WHERE session_id = ? AND id = ?",
                (session.id, run.id),
            ).fetchone()
            if row:
                receipt = _record(CodeRun, row)
                if receipt.text != run.text:
                    raise CodeConflictError(
                        "This Send ID was already used for a different message."
                    )
                return previous, receipt
            if (
                previous.status != "ready"
                or previous.model != run.model
                or previous.mode != run.mode
                or run.session_id != session.id
            ):
                raise CodeConflictError(
                    "This session changed. Refresh it and try again."
                )
            _idle(connection, session.id)
            _write_session(connection, session, expected_revision, settings=True)
            ordinal = connection.execute(
                "SELECT coalesce(max(ordinal), 0) + 1 FROM code_runs WHERE session_id = ?",
                (session.id,),
            ).fetchone()[0]
            admitted = run.model_copy(update={"ordinal": ordinal})
            if (
                item.session_id != session.id
                or item.run_id != run.id
                or item.id != run.id
                or item.kind != "user"
            ):
                raise CodeConflictError(
                    "The submitted message belongs to different work."
                )
            _insert(connection, "code_runs", admitted)
            _insert(connection, "code_items", item)
            return _session(connection, session.id), admitted

        return await self._run(operation)

    async def claim_prompt(
        self, session_id: str, prompt_id: str, response_id: str, generation: str
    ) -> CodePrompt:
        def operation(connection: sqlite3.Connection) -> CodePrompt:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM code_prompts WHERE session_id = ? AND id = ?",
                (session_id, prompt_id),
            ).fetchone()
            if row is None:
                raise CodeNotFoundError("This prompt no longer exists.")
            prompt = _record(CodePrompt, row)
            if prompt.response_id == response_id:
                return prompt
            if prompt.status != "pending" or prompt.generation != generation:
                raise CodeConflictError(
                    "This prompt was already answered or is no longer active."
                )
            claimed = prompt.model_copy(
                update={"status": "answering", "response_id": response_id}
            )
            _write_prompt(connection, claimed)
            return claimed

        return await self._run(operation)

    async def save_progress(
        self,
        session: CodeSession,
        expected_revision: int,
        *,
        run: CodeRun | None = None,
        items: Sequence[CodeItem] = (),
        prompts: Sequence[CodePrompt] = (),
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            _write_session(connection, session, expected_revision, settings=False)
            if run is not None:
                if run.session_id != session.id:
                    raise CodeConflictError("This turn belongs to another session.")
                _write_run(connection, run)
            for item in items:
                if item.session_id != session.id:
                    raise CodeConflictError("This entry belongs to another session.")
                _write_item(connection, item)
            for prompt in prompts:
                if prompt.session_id != session.id:
                    raise CodeConflictError("This prompt belongs to another session.")
                _write_prompt(connection, prompt)

        await self._run(operation)

    async def delete(self, session_id: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO code_deleted(id) VALUES (?)", (session_id,)
            )
            connection.execute("DELETE FROM code_sessions WHERE id = ?", (session_id,))

        await self._run(operation)


_PROMPT_COLUMNS = """(
    session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, id TEXT NOT NULL,
    generation TEXT NOT NULL, request_id TEXT NOT NULL, native_turn_id TEXT, native_item_id TEXT,
    kind TEXT NOT NULL, form TEXT NOT NULL, raw TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','answering','resolved','expired')), response_id TEXT, error TEXT,
    PRIMARY KEY(session_id,id), UNIQUE(session_id,generation,request_id), UNIQUE(session_id,response_id),
    FOREIGN KEY(session_id,id) REFERENCES code_items(session_id,id) ON DELETE CASCADE
)"""

_SCHEMA = (
    """
CREATE TABLE IF NOT EXISTS code_sessions(
    id TEXT PRIMARY KEY NOT NULL, cwd TEXT NOT NULL, model TEXT NOT NULL, reasoning_effort TEXT,
    harness TEXT NOT NULL CHECK(harness = 'codex'), title TEXT NOT NULL, title_search TEXT NOT NULL,
    cwd_search TEXT NOT NULL, auto_title INTEGER NOT NULL CHECK(auto_title IN (0,1)), native_thread_id TEXT,
    native_may_have_input INTEGER NOT NULL CHECK(native_may_have_input IN (0,1)),
    revision INTEGER NOT NULL CHECK(revision > 0), status TEXT NOT NULL CHECK(status IN ('ready','deleting','delete_uncertain')),
    error TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS code_sessions_recent ON code_sessions(updated_at DESC, id DESC);
CREATE TABLE IF NOT EXISTS code_runs(
    session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal > 0), text TEXT NOT NULL, model TEXT NOT NULL, reasoning_effort TEXT,
    status TEXT NOT NULL CHECK(status IN ('preparing','running','stopping','completed','interrupted','failed')),
    submission_started INTEGER NOT NULL CHECK(submission_started IN (0,1)), native_turn_id TEXT,
    stop_requested INTEGER NOT NULL CHECK(stop_requested IN (0,1)), error TEXT, error_details TEXT NOT NULL,
    created_at INTEGER NOT NULL, finished_at INTEGER,
    PRIMARY KEY(session_id,id), UNIQUE(session_id,ordinal), UNIQUE(session_id,native_turn_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS code_one_active_run ON code_runs(session_id)
    WHERE status IN ('preparing','running','stopping');
CREATE TABLE IF NOT EXISTS code_items(
    session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, id TEXT NOT NULL,
    run_id TEXT NOT NULL, sequence INTEGER NOT NULL CHECK(sequence > 0), native_turn_id TEXT, native_item_id TEXT,
    kind TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL, detail TEXT NOT NULL,
    complete INTEGER NOT NULL CHECK(complete IN (0,1)), raw TEXT NOT NULL,
    PRIMARY KEY(session_id,id), FOREIGN KEY(session_id,run_id) REFERENCES code_runs(session_id,id) ON DELETE CASCADE,
    UNIQUE(session_id,sequence), UNIQUE(session_id,native_turn_id,native_item_id)
);
CREATE INDEX IF NOT EXISTS code_items_run ON code_items(session_id,run_id,sequence);
CREATE TABLE IF NOT EXISTS code_deleted(id TEXT PRIMARY KEY NOT NULL);
"""
    + "CREATE TABLE IF NOT EXISTS code_prompts"
    + _PROMPT_COLUMNS
    + ";"
)
