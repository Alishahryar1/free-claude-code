import asyncio
import json
import sqlite3
import threading
import uuid
from contextlib import closing
from typing import Literal

import pytest
import pytest_asyncio

from free_claude_code.application.code_sessions.models import (
    CodeConflictError,
    CodeItem,
    CodePrompt,
    CodeRun,
    CodeSession,
    CodeUnavailableError,
)
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore


@pytest.mark.asyncio
async def test_cancelled_initialization_drains_thread_before_releasing_owner_lock(
    tmp_path,
):
    entered = threading.Event()
    release = threading.Event()
    initialized = threading.Event()

    class GatedStore(SQLiteCodeStore):
        def _initialize(self):
            entered.set()
            assert release.wait(5)
            super()._initialize()
            initialized.set()

    store = GatedStore(tmp_path / "code.db", tmp_path / "code.lock")
    starting = asyncio.create_task(store.start())
    await asyncio.to_thread(entered.wait, 3)
    starting.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await starting
    await asyncio.to_thread(initialized.wait, 3)
    second = SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock")
    try:
        await second.start()
        session = await second.create(
            CodeSession(id=str(uuid.uuid4()), cwd=str(tmp_path), model="provider/model")
        )
        assert (await second.get_session(session.id)).id == session.id
    finally:
        await store.close()
        await second.close()


@pytest.mark.asyncio
async def test_store_has_one_process_owner_and_can_reopen_after_close(tmp_path):
    first = SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock")
    second = SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock")
    await first.start()
    try:
        with pytest.raises(CodeUnavailableError, match="another FCC"):
            await second.start()
    finally:
        await first.close()
    await second.start()
    await second.close()


@pytest.mark.asyncio
async def test_reused_item_id_cannot_overwrite_another_session(tmp_path):
    store = SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock")
    await store.start()
    try:
        first, second = [
            await store.create(
                CodeSession(
                    id=str(uuid.uuid4()), cwd=str(tmp_path), model="provider/model"
                )
            )
            for _ in range(2)
        ]
        shared_id = str(uuid.uuid4())
        first, first_run = await _admit(store, first, text="first")
        original = CodeItem(
            id=shared_id,
            session_id=first.id,
            sequence=2,
            run_id=first_run.id,
            kind="tool",
            text="original output",
            complete=True,
        )
        await store.save_progress(first, first.revision, items=(original,))
        await _admit(store, second, run_id=shared_id, text="second")
        assert (await store.items(first.id, None, None))[-1] == original
        assert (await store.items(second.id, None, None))[0].text == "second"
    finally:
        await store.close()


async def _admit(store, session, *, run_id=None, text="message", sequence=1):
    run_id = run_id or str(uuid.uuid4())
    run = CodeRun(id=run_id, session_id=session.id, text=text, model=session.model)
    item = CodeItem(
        id=run_id,
        session_id=session.id,
        run_id=run_id,
        sequence=sequence,
        kind="user",
        text=text,
        complete=True,
    )
    return await store.admit_run(
        session.model_copy(update={"revision": session.revision + 1}),
        run,
        item,
        session.revision,
    )


@pytest_asyncio.fixture
async def store(tmp_path):
    result = SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock")
    await result.start()
    yield result
    await result.close()


async def _session(store):
    return await store.create(
        CodeSession(id=str(uuid.uuid4()), cwd="/work", model="provider/model")
    )


@pytest.mark.asyncio
async def test_mode_and_original_defaults_survive_restart_and_cannot_be_rewritten(
    store, tmp_path
):
    session = await _session(store)
    with closing(sqlite3.connect(tmp_path / "code.db")) as connection:
        assert connection.execute(
            "SELECT native_permission_defaults FROM code_sessions"
        ).fetchone() == (None,)
    defaults = {
        "approvalPolicy": "on-request",
        "approvalsReviewer": "user",
        "sandbox": {"type": "readOnly"},
    }
    session = session.model_copy(update={"native_permission_defaults": defaults})
    await store.save_progress(session, session.revision)
    for replacement in (None, {"different": True}):
        with pytest.raises(CodeConflictError):
            await store.save_progress(
                session.model_copy(update={"native_permission_defaults": replacement}),
                session.revision,
            )
    session = await store.update_settings(
        session.model_copy(
            update={"mode": "full_access", "revision": session.revision + 1}
        ),
        session.revision,
    )
    await store.close()
    await store.start()
    saved = await store.get_session(session.id)
    assert saved.mode == "full_access"
    assert saved.native_permission_defaults == defaults


@pytest.mark.asyncio
async def test_context_usage_progress_and_settings_use_existing_session_writes(
    store,
):
    session = await _session(store)
    snapshot = session.model_copy(update={"context_used_tokens": 12_438})
    await store.save_progress(snapshot, session.revision)
    saved = await store.get_session(session.id)
    assert saved.context_used_tokens == 12_438
    assert saved.revision == session.revision
    assert saved.updated_at == session.updated_at

    changed = snapshot.model_copy(
        update={
            "model": "provider/other",
            "revision": snapshot.revision + 1,
        }
    )
    changed = await store.update_settings(changed, snapshot.revision)
    assert changed.model == "provider/other"
    assert changed.context_used_tokens == 12_438


@pytest.mark.asyncio
async def test_mode_is_guarded_by_sqlite_busy_and_run_immutability(store):
    session, run = await _admit(store, await _session(store))
    with pytest.raises(CodeConflictError):
        await store.update_settings(
            session.model_copy(
                update={"mode": "full_access", "revision": session.revision + 1}
            ),
            session.revision,
        )
    with pytest.raises(CodeConflictError):
        await store.save_progress(
            session,
            session.revision,
            run=run.model_copy(update={"mode": "full_access"}),
        )


@pytest.mark.asyncio
async def test_version_one_database_gains_mode_without_losing_history(store, tmp_path):
    session, run = await _admit(store, await _session(store))
    items = await store.items(session.id, None, None)
    await store.close()
    with closing(sqlite3.connect(tmp_path / "code.db")) as connection, connection:
        connection.execute("ALTER TABLE code_sessions DROP COLUMN mode")
        connection.execute(
            "ALTER TABLE code_sessions DROP COLUMN native_permission_defaults"
        )
        connection.execute("ALTER TABLE code_sessions DROP COLUMN context_used_tokens")
        connection.execute("ALTER TABLE code_runs DROP COLUMN mode")
        connection.execute("PRAGMA user_version = 1")
    for _ in range(2):
        await store.start()
        saved = await store.get_session(session.id)
        assert saved.mode == "config"
        assert saved.native_permission_defaults is None
        assert (await store.get_run(session.id, run.id)).mode == "config"
        assert await store.items(session.id, None, None) == items
        with closing(sqlite3.connect(tmp_path / "code.db")) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute("UPDATE code_sessions SET mode = 'unknown'")
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE code_sessions SET native_permission_defaults = '[]'"
                )
        await store.close()


@pytest.mark.asyncio
async def test_version_two_database_gains_nullable_context_usage(store, tmp_path):
    session = await _session(store)
    await store.close()
    with closing(sqlite3.connect(tmp_path / "code.db")) as connection, connection:
        connection.execute("ALTER TABLE code_sessions DROP COLUMN context_used_tokens")
        connection.execute("PRAGMA user_version = 2")
    for _ in range(2):
        await store.start()
        assert (await store.get_session(session.id)).context_used_tokens is None
        with closing(sqlite3.connect(tmp_path / "code.db")) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
            column = next(
                row
                for row in connection.execute("PRAGMA table_info(code_sessions)")
                if row[1] == "context_used_tokens"
            )
            assert column[3] == 0
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE code_sessions SET context_used_tokens = -1 WHERE id = ?",
                    (session.id,),
                )
        await store.close()


@pytest.mark.asyncio
async def test_admission_is_atomic_and_idempotent(store):
    session = await _session(store)
    results = await asyncio.gather(
        _admit(store, session), _admit(store, session), return_exceptions=True
    )
    assert sum(isinstance(result, CodeConflictError) for result in results) == 1
    session, run = next(result for result in results if isinstance(result, tuple))
    assert (
        len(await store.runs(session.id))
        == len(await store.items(session.id, None, None))
        == 1
    )
    assert await _admit(store, session, run_id=run.id) == (session, run)
    await store.save_progress(
        session, session.revision, run=run.model_copy(update={"status": "completed"})
    )
    collision = CodeItem(
        id=str(uuid.uuid4()),
        session_id=session.id,
        run_id=run.id,
        sequence=2,
        kind="tool",
        text="keep me",
        complete=True,
    )
    await store.save_progress(session, session.revision, items=(collision,))
    with pytest.raises(CodeConflictError):
        await _admit(store, session, run_id=collision.id, sequence=3)
    assert (await store.get_session(session.id)).revision == session.revision
    assert len(await store.runs(session.id)) == 1
    assert (await store.items(session.id, None, None))[-1] == collision


@pytest.mark.asyncio
async def test_schema_rejects_second_active_run_and_cross_session_item_link(
    store, tmp_path
):
    session, run = await _admit(store, await _session(store))
    other = await _session(store)
    with closing(sqlite3.connect(tmp_path / "code.db")) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(code_runs)")]
        expressions = [
            "'another'"
            if column == "id"
            else "ordinal + 1"
            if column == "ordinal"
            else column
            for column in columns
        ]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"INSERT INTO code_runs SELECT {','.join(expressions)} FROM code_runs"
            )
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(code_items)")
        ]
        expressions = ["?" if column == "session_id" else column for column in columns]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"INSERT INTO code_items SELECT {','.join(expressions)} FROM code_items",
                (other.id,),
            )
    assert await store.get_run(session.id, run.id) == run


@pytest.mark.asyncio
async def test_prompt_claim_and_settings_are_guarded_in_storage(store):
    session, run = await _admit(store, await _session(store))
    await store.save_progress(
        session, session.revision, run=run.model_copy(update={"status": "completed"})
    )
    prompts = tuple(
        CodePrompt(
            id=str(uuid.uuid4()),
            session_id=session.id,
            generation="g",
            request_id=request_id,
            kind="question",
            form={},
            raw={},
        )
        for request_id in (1, "1")
    )
    items = tuple(
        CodeItem(
            id=prompt.id,
            session_id=session.id,
            run_id=run.id,
            sequence=index + 2,
            kind="prompt",
            complete=True,
        )
        for index, prompt in enumerate(prompts)
    )
    await store.save_progress(session, session.revision, items=items, prompts=prompts)
    results = await asyncio.gather(
        *(
            store.claim_prompt(session.id, prompts[0].id, str(uuid.uuid4()), "g")
            for _ in range(2)
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(result, CodeConflictError) for result in results) == 1
    claimed = next(result for result in results if isinstance(result, CodePrompt))
    assert (
        await store.claim_prompt(session.id, claimed.id, claimed.response_id, "g")
        == claimed
    )
    with pytest.raises(CodeConflictError):
        await store.claim_prompt(session.id, prompts[1].id, claimed.response_id, "g")
    settings = session.model_copy(
        update={"model": "provider/other", "revision": session.revision + 1}
    )
    with pytest.raises(CodeConflictError):
        await store.update_settings(settings, session.revision)
    await store.save_progress(
        session,
        session.revision,
        prompts=(
            claimed.model_copy(update={"status": "resolved"}),
            prompts[1].model_copy(update={"status": "expired"}),
        ),
    )
    assert (
        await store.update_settings(settings, session.revision)
    ).model == "provider/other"
    with pytest.raises(CodeConflictError):
        await store.update_settings(
            session.model_copy(update={"title": "stale"}), session.revision
        )


@pytest.mark.asyncio
async def test_recovered_old_output_pages_with_its_original_run_outcome(store):
    session, old = await _admit(store, await _session(store))
    old = old.model_copy(update={"status": "failed", "error": "first failed"})
    await store.save_progress(session, session.revision, run=old)
    session, newer = await _admit(store, session, sequence=2)
    tail = CodeItem(
        id=str(uuid.uuid4()),
        session_id=session.id,
        run_id=old.id,
        sequence=3,
        kind="assistant",
        text="recovered",
        complete=True,
    )
    await store.save_progress(session, session.revision, items=(tail,))
    page = await store.item_page(session.id, None, 2)
    assert [item.run_id for item in page.items] == [old.id, newer.id]
    assert page.runs[0].error == "first failed"
    assert page.next_before == (old.ordinal, 3)
    older = await store.item_page(session.id, page.next_before, 2)
    assert [item.sequence for item in older.items] == [1]


@pytest.mark.asyncio
async def test_prompt_entry_and_form_are_atomic_and_linked_in_storage(store, tmp_path):
    session, run = await _admit(store, await _session(store))
    prompt = CodePrompt(
        id=str(uuid.uuid4()),
        session_id=session.id,
        generation="g",
        request_id=1,
        kind="questions",
        form={},
        raw={},
    )
    item = CodeItem(
        id=prompt.id,
        session_id=session.id,
        run_id=run.id,
        sequence=2,
        kind="prompt",
        complete=True,
    )
    with pytest.raises(CodeConflictError):
        await store.save_progress(session, session.revision, prompts=(prompt,))
    with pytest.raises(CodeConflictError):
        await store.save_progress(
            session,
            session.revision,
            items=(item.model_copy(update={"kind": "text"}),),
            prompts=(prompt,),
        )
    assert await store.prompts(session.id) == ()
    assert len(await store.items(session.id, None, None)) == 1
    await store.save_progress(
        session, session.revision, items=(item,), prompts=(prompt,)
    )
    other = await _session(store)
    with closing(sqlite3.connect(tmp_path / "code.db")) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE code_prompts SET session_id = ?", (other.id,))
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE code_prompts SET id = 'unlinked'")
    assert await store.prompts(session.id) == (prompt,)


_LEGACY_PROMPTS = """
CREATE TABLE code_prompts(
    session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE,
    id TEXT NOT NULL, generation TEXT NOT NULL, request_id TEXT NOT NULL,
    native_turn_id TEXT, native_item_id TEXT, kind TEXT NOT NULL,
    form TEXT NOT NULL, raw TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','answering','resolved','expired')),
    response_id TEXT, error TEXT,
    PRIMARY KEY(session_id,id), UNIQUE(session_id,generation,request_id),
    UNIQUE(session_id,response_id)
)
"""


async def _legacy_prompt_database(store, path):
    session, old = await _admit(store, await _session(store))
    await store.save_progress(
        session,
        session.revision,
        run=old.model_copy(
            update={
                "native_turn_id": "old-turn",
                "submission_started": True,
                "status": "completed",
            }
        ),
    )
    tool = CodeItem(
        id=str(uuid.uuid4()),
        session_id=session.id,
        run_id=old.id,
        sequence=2,
        native_turn_id="old-turn",
        native_item_id="shared-tool",
        kind="tool",
        text="Original tool output",
        complete=True,
    )
    await store.save_progress(session, session.revision, items=(tool,))
    session, latest = await _admit(store, session, text="Later turn", sequence=3)
    await store.save_progress(
        session,
        session.revision,
        run=latest.model_copy(
            update={
                "status": "completed",
            }
        ),
    )
    original = await store.items(session.id, None, None)
    await store.close()
    cases: list[
        tuple[str | None, str | None, Literal["resolved", "expired", "pending"]]
    ] = [
        ("old-turn", None, "resolved"),
        (None, "shared-tool", "expired"),
        (None, None, "resolved"),
        ("missing-turn", None, "pending"),
        ("old-turn", None, "expired"),
    ]
    prompts = tuple(
        CodePrompt(
            id=str(uuid.uuid4()),
            session_id=session.id,
            generation="g",
            request_id=index,
            native_turn_id=turn,
            native_item_id=item,
            kind="questions",
            form={"title": "Old question"},
            raw={"keep": "native payload"},
            status=status,
            response_id=str(uuid.uuid4()) if status == "resolved" else None,
        )
        for index, (turn, item, status) in enumerate(cases)
    )
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DROP TABLE code_prompts")
        connection.execute(_LEGACY_PROMPTS)
        connection.execute("ALTER TABLE code_sessions DROP COLUMN mode")
        connection.execute(
            "ALTER TABLE code_sessions DROP COLUMN native_permission_defaults"
        )
        connection.execute("ALTER TABLE code_sessions DROP COLUMN context_used_tokens")
        connection.execute("ALTER TABLE code_runs DROP COLUMN mode")
        connection.execute("PRAGMA user_version = 0")
        for prompt in prompts:
            connection.execute(
                "INSERT INTO code_prompts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    session.id,
                    prompt.id,
                    prompt.generation,
                    json.dumps(prompt.request_id),
                    prompt.native_turn_id,
                    prompt.native_item_id,
                    prompt.kind,
                    json.dumps(prompt.form),
                    json.dumps(prompt.raw),
                    prompt.status,
                    prompt.response_id,
                    prompt.error,
                ),
            )
    return session, old, latest, original, prompts


@pytest.mark.asyncio
async def test_legacy_prompts_migrate_once_to_run_ends_without_resequencing(
    store, tmp_path
):
    path = tmp_path / "code.db"
    session, old, latest, original, prompts = await _legacy_prompt_database(store, path)
    for _ in range(2):
        await store.start()
        items = await store.items(session.id, None, None)
        assert [item.id for item in items] == [
            original[0].id,
            original[1].id,
            prompts[0].id,
            prompts[1].id,
            prompts[4].id,
            original[2].id,
            prompts[2].id,
            prompts[3].id,
        ]
        assert [item for item in items if item.kind != "prompt"] == list(original)
        entries = {item.id: item for item in items if item.kind == "prompt"}
        for index, prompt in enumerate(prompts):
            entry = entries[prompt.id]
            assert entry.run_id == (old.id if index in (0, 1, 4) else latest.id)
            assert entry.sequence == len(original) + index + 1
            assert entry.native_item_id is None and entry.native_turn_id is None
            assert entry.raw == {} and entry.text == ""
        saved = {prompt.id: prompt for prompt in await store.prompts(session.id)}
        for prompt in prompts:
            expected = (
                prompt.model_copy(update={"status": "expired"})
                if prompt.status == "pending"
                else prompt
            )
            assert saved[prompt.id] == expected
        with closing(sqlite3.connect(path)) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
            assert any(
                row[2] == "code_items"
                for row in connection.execute("PRAGMA foreign_key_list(code_prompts)")
            )
        await store.close()


@pytest.mark.asyncio
async def test_failed_prompt_migration_rolls_back_items_schema_and_version(
    store, tmp_path
):
    path = tmp_path / "code.db"
    _, _, _, original, prompts = await _legacy_prompt_database(store, path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TRIGGER refuse_second_prompt BEFORE INSERT ON code_items "
            "WHEN NEW.id = '"
            + prompts[1].id
            + "' BEGIN SELECT RAISE(ABORT, 'blocked'); END"
        )
    with pytest.raises(CodeConflictError):
        await store.start()
    with closing(sqlite3.connect(path)) as connection, connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM code_items").fetchone()[
            0
        ] == len(original)
        assert connection.execute("SELECT count(*) FROM code_prompts").fetchone()[
            0
        ] == len(prompts)
        assert not any(
            row[2] == "code_items"
            for row in connection.execute("PRAGMA foreign_key_list(code_prompts)")
        )
        connection.execute("DROP TRIGGER refuse_second_prompt")
    await store.start()
    assert len(await store.items(prompts[0].session_id, None, None)) == len(
        original
    ) + len(prompts)
