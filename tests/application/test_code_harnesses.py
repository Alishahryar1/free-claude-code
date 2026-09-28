import asyncio
import uuid

import pytest

from free_claude_code.application.code_sessions import CodeService
from free_claude_code.application.code_sessions.models import (
    CodeConflictError,
    CodeModeOption,
    CodeValidationError,
)
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from tests.code_sessions_support import FakeHarness


@pytest.fixture
def harnesses():
    codex, claude = FakeHarness(), FakeHarness()
    claude.id, claude.name, claude.prepare_on_open = "claude", "Claude Code", True
    claude.modes = (
        CodeModeOption(id="config", name="Use config"),
        CodeModeOption(id="plan", name="Plan"),
    )
    return {"codex": codex, "claude": claude}


@pytest.mark.asyncio
async def test_fixed_harness_routes_preparation_and_send(
    database_factory, tmp_path, harnesses
):
    code = CodeService(
        SQLiteCodeStore(database_factory(tmp_path / "code.db", tmp_path / "lock")),
        harnesses,
    )
    await code.start()
    try:
        session = await code.create_session(str(uuid.uuid4()), str(tmp_path), "claude")
        prepared = await code.prepare_session(
            session.id, session.revision, expected_epoch=code.epoch
        )
        assert prepared.capabilities is not None
        assert not harnesses["codex"].connections
        assert len(harnesses["claude"].connections) == 1
        assert not harnesses["claude"].connections[0].inputs
        with pytest.raises(CodeValidationError):
            await code.update_settings(
                session.id, session.revision, {"harness": "codex"}
            )
        with pytest.raises(CodeValidationError):
            await code.update_settings(
                session.id, session.revision, {"mode": "auto_review"}
            )
        with pytest.raises(CodeConflictError):
            await code.create_session(session.id, str(tmp_path), "codex")
        await code.send(
            session.id,
            str(uuid.uuid4()),
            session.revision,
            "hello",
            expected_epoch=code.epoch,
        )
        await harnesses["claude"].wait_inputs(1)
        assert not harnesses["codex"].connections
    finally:
        await code.close()


@pytest.mark.asyncio
async def test_parallel_prepare_shares_connection_and_setup_sends_nothing(
    database_factory, tmp_path, harnesses
):
    code = CodeService(
        SQLiteCodeStore(database_factory(tmp_path / "code.db", tmp_path / "lock")),
        harnesses,
    )
    await code.start()
    try:
        session = await code.create_session(str(uuid.uuid4()), str(tmp_path), "claude")
        harness = harnesses["claude"]
        harness.creation_gate.clear()
        tasks = [
            asyncio.create_task(
                code.prepare_session(
                    session.id, session.revision, expected_epoch=code.epoch
                )
            )
            for _ in range(2)
        ]
        await harness.creating.wait()
        assert len(harness.connections) == 1
        harness.creation_gate.set()
        results = await asyncio.gather(*tasks)
        assert results[0].capabilities is not None
        assert results[1].capabilities is not None
        assert results[0].capabilities.generation == results[1].capabilities.generation
        assert not harness.connections[0].inputs
    finally:
        await code.close()


@pytest.mark.asyncio
async def test_missing_binary_does_not_block_saved_history(
    database_factory, tmp_path, harnesses
):
    code = CodeService(
        SQLiteCodeStore(database_factory(tmp_path / "code.db", tmp_path / "lock")),
        harnesses,
    )
    await code.start()
    try:
        session = await code.create_session(str(uuid.uuid4()), str(tmp_path), "claude")
        for harness in harnesses.values():
            harness.availability = lambda: (False, "missing")
        assert (await code.get_detail(session.id)).session == session
        subscription, _ = await code.subscribe()
        await subscription.aclose()
        assert not any(item.available for item in code.harnesses())
    finally:
        await code.close()


@pytest.mark.asyncio
async def test_send_waits_for_preparation_without_duplicate_process(
    database_factory, tmp_path, harnesses
):
    code = CodeService(
        SQLiteCodeStore(database_factory(tmp_path / "code.db", tmp_path / "lock")),
        harnesses,
    )
    await code.start()
    try:
        session = await code.create_session(str(uuid.uuid4()), str(tmp_path), "claude")
        harness = harnesses["claude"]
        harness.creation_gate.clear()
        preparing = asyncio.create_task(
            code.prepare_session(
                session.id, session.revision, expected_epoch=code.epoch
            )
        )
        await harness.creating.wait()
        with pytest.raises(CodeConflictError):
            await code.update_settings(session.id, session.revision, {"mode": "plan"})
        sending = asyncio.create_task(
            code.send(
                session.id,
                str(uuid.uuid4()),
                session.revision,
                "only once",
                expected_epoch=code.epoch,
            )
        )
        await asyncio.sleep(0)
        assert not harness.connections[0].inputs
        harness.creation_gate.set()
        await asyncio.gather(preparing, sending)
        await harness.wait_inputs(1)
        assert len(harness.connections) == 1
        assert len(harness.connections[0].inputs) == 1
    finally:
        await code.close()


@pytest.mark.asyncio
async def test_deletion_quiesces_initialization_before_removing_session(
    database_factory, tmp_path, harnesses
):
    code = CodeService(
        SQLiteCodeStore(database_factory(tmp_path / "code.db", tmp_path / "lock")),
        harnesses,
    )
    await code.start()
    try:
        session = await code.create_session(str(uuid.uuid4()), str(tmp_path), "claude")
        harness = harnesses["claude"]
        harness.creation_gate.clear()
        preparing = asyncio.create_task(
            code.prepare_session(
                session.id, session.revision, expected_epoch=code.epoch
            )
        )
        await harness.creating.wait()
        await code.delete_session(session.id, session.revision)
        await asyncio.wait_for(code.wait_idle(session.id), 2)
        await asyncio.gather(preparing, return_exceptions=True)
        assert not (await code.list_sessions()).sessions
        assert all(
            connection.closed and not connection.inputs
            for connection in harness.connections
        )
    finally:
        await code.close()


@pytest.mark.asyncio
async def test_browser_cancellation_leaves_preparation_owned(
    database_factory, tmp_path, harnesses
):
    code = CodeService(
        SQLiteCodeStore(database_factory(tmp_path / "code.db", tmp_path / "lock")),
        harnesses,
    )
    await code.start()
    try:
        session = await code.create_session(str(uuid.uuid4()), str(tmp_path), "claude")
        harness = harnesses["claude"]
        harness.creation_gate.clear()
        caller = asyncio.create_task(
            code.prepare_session(
                session.id, session.revision, expected_epoch=code.epoch
            )
        )
        await harness.creating.wait()
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        assert not harness.connections[0].closed
        harness.creation_gate.set()
        detail = await code.prepare_session(
            session.id, session.revision, expected_epoch=code.epoch
        )
        assert detail.capabilities is not None
        assert len(harness.connections) == 1
    finally:
        await code.close()
