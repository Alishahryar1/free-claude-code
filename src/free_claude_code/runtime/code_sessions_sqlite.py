"""Relational, transaction-owned state for FCC coding conversations."""

import asyncio
import json
import sqlite3
from collections.abc import Callable, Sequence

from free_claude_code.application.code_sessions.models import (
    ACTIVE_RUN_STATUSES,
    CodeConflictError,
    CodeExecutionSeed,
    CodeHistory,
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

from .sqlite_database import SQLiteDatabase

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


def _by_ids[T: Record](
    connection: sqlite3.Connection,
    model: type[T],
    session_id: str,
    ids: Sequence[str],
) -> tuple[T, ...]:
    tables: dict[type[Record], str] = {
        CodeItem: "code_items",
        CodeRun: "code_runs",
        CodePrompt: "code_prompts",
    }
    table = tables[model]
    records: list[T] = []
    unique = tuple(dict.fromkeys(ids))
    for start in range(0, len(unique), 500):
        batch = unique[start : start + 500]
        records.extend(
            _record(model, row)
            for row in connection.execute(
                f"SELECT * FROM {table} WHERE session_id = ? AND id IN ({','.join('?' for _ in batch)})",
                (session_id, *batch),
            )
        )
    return tuple(records)


def _latest_run(connection: sqlite3.Connection, session_id: str) -> CodeRun | None:
    row = connection.execute(
        "SELECT * FROM code_runs WHERE session_id = ? ORDER BY ordinal DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    return _record(CodeRun, row) if row else None


def _active_prompts(
    connection: sqlite3.Connection, session_id: str
) -> tuple[CodePrompt, ...]:
    return tuple(
        _record(CodePrompt, row)
        for row in connection.execute(
            "SELECT * FROM code_prompts WHERE session_id = ? AND status IN ('pending', 'answering') ORDER BY id",
            (session_id,),
        )
    )


def _item_page(
    connection: sqlite3.Connection,
    session_id: str,
    before: tuple[int, int] | None,
    limit: int | None,
) -> CodeItemPage:
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
    runs = _by_ids(connection, CodeRun, session_id, [row["run_id"] for row in selected])
    last = (
        selected[-1] if selected and limit is not None and len(rows) > limit else None
    )
    return CodeItemPage(
        tuple(_record(CodeItem, row) for row in reversed(selected)),
        tuple(sorted(runs, key=lambda run: run.ordinal)),
        (last["run_ordinal"], last["sequence"]) if last else None,
    )


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
            "context_used_tokens",
            "revision",
            "updated_at",
        }
        if settings
        else {
            "native_thread_id",
            "native_permission_defaults",
            "native_may_have_input",
            "context_used_tokens",
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


class SQLiteCodeStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database
        self._started = False
        self._lifecycle = asyncio.Lock()

    async def start(self) -> None:
        async with self._lifecycle:
            if self._started:
                return
            try:
                await self.database.start()
                await self._execute(self._recover, write=True)
            except sqlite3.IntegrityError as exc:
                raise CodeConflictError(
                    "Saved Code history conflicts with its schema."
                ) from exc
            except (sqlite3.Error, OSError) as exc:
                raise CodeUnavailableError(
                    f"Code storage initialization failed: {exc}"
                ) from exc
            self._started = True

    async def close(self) -> None:
        async with self._lifecycle:
            self._started = False

    async def _execute[T](
        self, operation: Callable[[sqlite3.Connection], T], *, write: bool
    ) -> T:
        try:
            return await self.database.run(operation, write=write)
        except sqlite3.IntegrityError as exc:
            raise CodeConflictError(
                "This Code operation conflicts with existing session state."
            ) from exc
        except sqlite3.Error as exc:
            raise CodeUnavailableError("Code session storage is unavailable.") from exc

    async def _run[T](
        self, operation: Callable[[sqlite3.Connection], T], *, write: bool
    ) -> T:
        if not self._started:
            raise CodeUnavailableError("Code session storage is closed.")
        return await self._execute(operation, write=write)

    def _recover(self, connection: sqlite3.Connection) -> None:
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

        return await self._run(operation, write=True)

    async def get_session(self, session_id: str) -> CodeSession:
        return await self._run(
            lambda connection: _session(connection, session_id), write=False
        )

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

        return await self._run(operation, write=False)

    async def pending_deletions(self) -> tuple[CodeSession, ...]:
        return await self._run(
            lambda connection: tuple(
                _record(CodeSession, row)
                for row in connection.execute(
                    "SELECT * FROM code_sessions WHERE status != 'ready'"
                )
            ),
            write=False,
        )

    async def is_deleted(self, session_id: str) -> bool:
        return await self._run(
            lambda connection: (
                connection.execute(
                    "SELECT 1 FROM code_deleted WHERE id = ?", (session_id,)
                ).fetchone()
                is not None
            ),
            write=False,
        )

    async def get_run(self, session_id: str, run_id: str) -> CodeRun | None:
        def operation(connection: sqlite3.Connection) -> CodeRun | None:
            row = connection.execute(
                "SELECT * FROM code_runs WHERE session_id = ? AND id = ?",
                (session_id, run_id),
            ).fetchone()
            return _record(CodeRun, row) if row else None

        return await self._run(operation, write=False)

    async def runs(self, session_id: str) -> tuple[CodeRun, ...]:
        return await self._run(
            lambda connection: tuple(
                _record(CodeRun, row)
                for row in connection.execute(
                    "SELECT * FROM code_runs WHERE session_id = ? ORDER BY ordinal",
                    (session_id,),
                )
            ),
            write=False,
        )

    async def latest_run(self, session_id: str) -> CodeRun | None:
        def operation(connection: sqlite3.Connection) -> CodeRun | None:
            row = connection.execute(
                "SELECT * FROM code_runs WHERE session_id = ? ORDER BY ordinal DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            return _record(CodeRun, row) if row else None

        return await self._run(operation, write=False)

    async def items(
        self, session_id: str, before: tuple[int, int] | None, limit: int | None
    ) -> tuple[CodeItem, ...]:
        return (await self.item_page(session_id, before, limit)).items

    async def item_page(
        self, session_id: str, before: tuple[int, int] | None, limit: int | None
    ) -> CodeItemPage:
        def operation(connection: sqlite3.Connection) -> CodeItemPage:
            return _item_page(connection, session_id, before, limit)

        return await self._run(operation, write=False)

    async def read_history(
        self,
        session_id: str,
        before: tuple[int, int] | None,
        include_item_ids: Sequence[str],
    ) -> CodeHistory:
        def operation(connection: sqlite3.Connection) -> CodeHistory:
            session = _session(connection, session_id)
            run = _latest_run(connection, session_id)
            active = run is not None and run.status in ACTIVE_RUN_STATUSES
            boundary = before
            if active and run is not None:
                start = (run.ordinal, 0)
                boundary = min(before, start) if before else start
            page = _item_page(connection, session_id, boundary, 50)
            active_prompts = _active_prompts(connection, session_id)
            extras = _by_ids(
                connection,
                CodeItem,
                session_id,
                (*include_item_ids, *(prompt.id for prompt in active_prompts)),
            )
            items = {item.id: item for item in (*page.items, *extras)}
            if active and run is not None:
                for row in connection.execute(
                    "SELECT * FROM code_items WHERE session_id = ? AND run_id = ?",
                    (session_id, run.id),
                ):
                    item = _record(CodeItem, row)
                    items[item.id] = item
            run_ids = {item.run_id for item in items.values()}
            if run is not None:
                run_ids.add(run.id)
            runs = _by_ids(connection, CodeRun, session_id, tuple(run_ids))
            ordinals = {run.id: run.ordinal for run in runs}
            selected = tuple(
                sorted(
                    items.values(),
                    key=lambda item: (ordinals[item.run_id], item.sequence),
                )
            )
            prompts = _by_ids(
                connection,
                CodePrompt,
                session_id,
                [item.id for item in selected if item.kind == "prompt"],
            )
            return CodeHistory(
                session,
                run,
                selected,
                prompts,
                tuple(sorted(runs, key=lambda run: run.ordinal)),
                tuple(prompt.id for prompt in active_prompts),
                page.next_before,
            )

        return await self._run(operation, write=False)

    async def execution_seed(self, session_id: str) -> CodeExecutionSeed:
        def operation(connection: sqlite3.Connection) -> CodeExecutionSeed:
            session = _session(connection, session_id)
            run = _latest_run(connection, session_id)
            sequence = connection.execute(
                "SELECT coalesce(max(sequence), 0) FROM code_items WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            prompts = _active_prompts(connection, session_id)
            items = _by_ids(
                connection, CodeItem, session_id, [prompt.id for prompt in prompts]
            )
            ids = {item.run_id for item in items}
            if run is not None:
                ids.add(run.id)
            return CodeExecutionSeed(
                session,
                run,
                sequence,
                items,
                prompts,
                _by_ids(connection, CodeRun, session_id, tuple(ids)),
            )

        return await self._run(operation, write=False)

    async def get_native_item(
        self, session_id: str, turn_id: str, item_id: str
    ) -> CodeItem | None:
        def operation(connection: sqlite3.Connection) -> CodeItem | None:
            row = connection.execute(
                "SELECT * FROM code_items WHERE session_id = ? AND native_turn_id = ? AND native_item_id = ?",
                (session_id, turn_id, item_id),
            ).fetchone()
            return _record(CodeItem, row) if row else None

        return await self._run(operation, write=False)

    async def run_items(self, session_id: str, run_id: str) -> tuple[CodeItem, ...]:
        return await self._run(
            lambda connection: tuple(
                _record(CodeItem, row)
                for row in connection.execute(
                    "SELECT * FROM code_items WHERE session_id = ? AND run_id = ? ORDER BY sequence",
                    (session_id, run_id),
                )
            ),
            write=False,
        )

    async def get_prompt(self, session_id: str, prompt_id: str) -> CodePrompt | None:
        return await self._run(
            lambda connection: next(
                iter(_by_ids(connection, CodePrompt, session_id, (prompt_id,))), None
            ),
            write=False,
        )

    async def has_prompt_request(
        self, session_id: str, generation: str, request_id: str | int
    ) -> bool:
        return await self._run(
            lambda connection: (
                connection.execute(
                    "SELECT 1 FROM code_prompts WHERE session_id = ? AND generation = ? AND request_id = ?",
                    (session_id, generation, json.dumps(request_id)),
                ).fetchone()
                is not None
            ),
            write=False,
        )

    async def has_native_turn(self, session_id: str, turn_id: str) -> bool:
        return await self._run(
            lambda connection: (
                connection.execute(
                    "SELECT 1 FROM code_runs WHERE session_id = ? AND native_turn_id = ?",
                    (session_id, turn_id),
                ).fetchone()
                is not None
            ),
            write=False,
        )

    async def prompts(self, session_id: str) -> tuple[CodePrompt, ...]:
        return await self._run(
            lambda connection: tuple(
                _record(CodePrompt, row)
                for row in connection.execute(
                    "SELECT * FROM code_prompts WHERE session_id = ? ORDER BY id",
                    (session_id,),
                )
            ),
            write=False,
        )

    async def update_settings(
        self, session: CodeSession, expected_revision: int
    ) -> CodeSession:
        def operation(connection: sqlite3.Connection) -> CodeSession:
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

        return await self._run(operation, write=True)

    async def admit_run(
        self, session: CodeSession, run: CodeRun, item: CodeItem, expected_revision: int
    ) -> tuple[CodeSession, CodeRun]:
        def operation(connection: sqlite3.Connection) -> tuple[CodeSession, CodeRun]:
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

        return await self._run(operation, write=True)

    async def claim_prompt(
        self, session_id: str, prompt_id: str, response_id: str, generation: str
    ) -> CodePrompt:
        def operation(connection: sqlite3.Connection) -> CodePrompt:
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

        return await self._run(operation, write=True)

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

        await self._run(operation, write=True)

    async def delete(self, session_id: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                "INSERT OR IGNORE INTO code_deleted(id) VALUES (?)", (session_id,)
            )
            connection.execute("DELETE FROM code_sessions WHERE id = ?", (session_id,))

        await self._run(operation, write=True)
