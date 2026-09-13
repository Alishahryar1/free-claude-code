import asyncio
import uuid
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from free_claude_code.application.code_sessions import CodeConflictError, CodeService
from free_claude_code.application.code_sessions.models import (
    CodeUnavailableError,
    CodeValidationError,
    HarnessEvent,
    ItemUpdate,
    PromptRequest,
)
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from free_claude_code.runtime.codex_protocol import CodexProtocol
from tests.code_sessions_support import CodexPackets, FakeConnection, FakeHarness


def new_id():
    return str(uuid.uuid4())


@pytest_asyncio.fixture
async def code(tmp_path):
    harness = FakeHarness()
    store = SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock")
    service = CodeService(store, harness)
    await service.start()
    try:
        yield service, harness, tmp_path
    finally:
        await service.close()


async def session_for(code):
    service, _, directory = code
    return await service.create_session(new_id(), str(directory))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "finished", "status"),
    [(BrokenPipeError, False, "failed"), (ConnectionResetError, True, "completed")],
)
async def test_child_lookup_pipe_failure_delivers_queued_close(
    code, monkeypatch, failure, finished, status
):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "Delegate", expected_epoch=service.epoch
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    connection = harness.connections[0]
    packets = CodexPackets(connection)
    await packets.spawn()
    await packets.review()
    if finished:
        await connection.finish("turn-1")
    native = packets.native
    monkeypatch.delattr(native, "rpc")
    native._alive = True
    native._process = Mock(
        stdin=Mock(drain=AsyncMock(side_effect=failure("Pipe closed")))
    )
    await native._packet(
        {
            "method": "guardianWarning",
            "params": {"threadId": "unregistered-child", "message": "Child warning"},
        }
    )
    native._queue.put_nowait(
        HarnessEvent(
            connection.generation,
            connection.thread_id,
            "closed",
            message="Native process ended",
        )
    )
    native._queue.put_nowait(None)
    try:
        await native._dispatch()
        detail = await service.get_detail(session.id)
        assert detail.active_review_ids == ()
        assert detail.run.status == status
        assert [item.kind for item in detail.items] == ["user", "subagent_auto_review"]
        assert not detail.items[-1].complete
    finally:
        for _, future in native._pending.values():
            future.cancel()


@pytest.mark.asyncio
async def test_pending_child_review_outside_page_keeps_its_run_and_history_cursor(code):
    service, harness, _ = code
    session = await session_for(code)
    first = await service.send(
        session.id, new_id(), session.revision, "Delegate", expected_epoch=service.epoch
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    connection = harness.connections[0]
    packets = CodexPackets(connection)
    await packets.spawn()
    await packets.review()
    review = (await service.get_detail(session.id)).items[-1]
    await connection.finish("turn-1")
    session = (await service.get_detail(session.id)).session
    await service.send(
        session.id, new_id(), session.revision, "Continue", expected_epoch=service.epoch
    )
    await asyncio.wait_for(harness.wait_inputs(2), 3)
    for index in range(55):
        await connection.text("turn-2", str(index), f"Output {index}", complete=True)
    await connection.finish("turn-2")
    newest = await service.get_detail(session.id)
    assert len(newest.items) == 51
    assert newest.items[0].id == review.id and newest.items[0].run_id == first.id
    assert first.id in {run.id for run in newest.runs}
    assert newest.active_review_ids == (review.id,)
    assert newest.next_before == (2, 9)
    await packets.review(status="approved")
    resolved = await service.get_detail(session.id)
    assert resolved.active_review_ids == ()
    assert review.id not in {item.id for item in resolved.items}
    assert resolved.next_before == newest.next_before
    refreshed = await service.get_detail(
        session.id, include_item_ids=(review.id, review.id, "missing")
    )
    assert len(refreshed.items) == 51
    assert refreshed.items[0].id == review.id and refreshed.items[0].complete
    assert first.id in {run.id for run in refreshed.runs}
    assert refreshed.next_before == newest.next_before
    older = await service.get_detail(session.id, before=newest.next_before)
    saved = next(item for item in older.items if item.id == review.id)
    assert saved.complete and saved.title == "Sub-agent Auto-review: Approved"
    assert saved.sequence == review.sequence and saved.run_id == first.id
    merged = {item.id: item for item in (*older.items, *resolved.items)}
    assert sorted(item.sequence for item in merged.values()) == list(range(1, 59))


@pytest.mark.asyncio
@pytest.mark.parametrize("ownership", ["notification", "metadata"])
async def test_owned_child_reviews_and_warnings_reach_saved_parent_transcript(
    code, ownership
):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "Delegate", expected_epoch=service.epoch
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    connection = harness.connections[0]
    packets = CodexPackets(connection)
    if ownership == "notification":
        await packets.spawn()
        await packets.spawn("grandchild", "child")
        await packets.spawn("unrelated", "outside")
    else:
        packets.parents.update(
            {
                "child": connection.thread_id,
                "grandchild": "child",
                "unrelated": "outside",
            }
        )
    for child in (connection.thread_id, "child", "grandchild", "unrelated"):
        await packets.review(child, status="approved", turn="turn-1")
        await packets.warning(child)
    detail = await service.get_detail(session.id)
    assert [item.kind for item in detail.items] == [
        "user",
        "auto_review",
        "notice",
        "subagent_auto_review",
        "notice",
        "subagent_auto_review",
        "notice",
    ]
    reviews = [item for item in detail.items if item.kind == "subagent_auto_review"]
    assert len({item.native_item_id for item in reviews}) == 2
    assert [item.raw["threadId"] for item in reviews] == ["child", "grandchild"]
    assert all(item.title == "Sub-agent Auto-review: Approved" for item in reviews)
    assert detail.items[-1].text == "Sub-agent: Native warning"
    assert detail.items[-1].raw == {
        "threadId": "grandchild",
        "message": "Native warning",
    }
    assert await service._store.items(session.id, None, None) == detail.items
    assert detail.run.status == "running"


@pytest.mark.asyncio
async def test_child_review_stays_with_first_entry_after_next_turn_begins(code):
    service, harness, _ = code
    session = await session_for(code)
    first = await service.send(
        session.id, new_id(), session.revision, "Delegate", expected_epoch=service.epoch
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    connection = harness.connections[0]
    packets = CodexPackets(connection)
    await packets.spawn()
    await packets.review()
    before = await service.get_detail(session.id)
    assert len(before.items) == 2
    review = before.items[-1]
    assert before.active_review_ids == (review.id,)
    await connection.finish("turn-1")
    await packets.send(
        "turn/completed",
        {"threadId": "child", "turn": {"id": "child-turn", "status": "completed"}},
    )
    assert (await service.get_detail(session.id)).active_review_ids == (review.id,)
    subscription, ready = await service.subscribe()
    events = aiter(subscription)
    try:
        assert ready["sessions"][0]["active_review_ids"] == [review.id]
        session = (await service.get_detail(session.id)).session
        harness.start_gate.clear()
        harness.started.clear()
        second = await service.send(
            session.id,
            new_id(),
            session.revision,
            "Continue",
            expected_epoch=service.epoch,
        )
        await asyncio.wait_for(harness.wait_inputs(2), 3)
        await packets.review(status="denied")
        await packets.review(status="denied")
        await packets.review()
        await packets.warning()
        await packets.review(review_id="next", status="approved", turn="turn-2")
        await packets.send(
            "turn/started", {"threadId": "child", "turn": {"id": "turn-2"}}
        )
        await packets.send(
            "turn/completed",
            {"threadId": "child", "turn": {"id": "turn-2", "status": "completed"}},
        )
        detail = await service.get_detail(session.id)
        saved = next(item for item in detail.items if item.id == review.id)
        assert (saved.run_id, saved.sequence, saved.native_turn_id) == (
            first.id,
            review.sequence,
            "child-turn",
        )
        assert saved.title == "Sub-agent Auto-review: Denied" and saved.complete
        assert detail.active_review_ids == ()
        assert detail.run.id == second.id and detail.run.native_turn_id is None
        assert all(item.run_id == second.id for item in detail.items[-3:])
        while (event := await asyncio.wait_for(anext(events), 3)).data.get(
            "item", {}
        ).get("id") != review.id:
            pass
        assert event.data["runs"][0]["id"] == first.id
        assert event.data["run"]["id"] == second.id
        harness.start_gate.set()
        await asyncio.wait_for(harness.started.wait(), 3)
        assert (await service.get_detail(session.id)).run.native_turn_id == "turn-2"
        await connection.finish("turn-2")
    finally:
        harness.start_gate.set()
        await subscription.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("closure", ["native", "service"])
async def test_pending_child_review_becomes_unavailable_when_idle_connection_ends(
    code, closure
):
    service, harness, directory = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "Delegate", expected_epoch=service.epoch
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    connection = harness.connections[0]
    packets = CodexPackets(connection)
    await packets.spawn()
    await connection.finish("turn-1")
    await packets.review()
    before = await service.get_detail(session.id)
    assert len(before.items) == 2
    review = before.items[-1]
    assert before.active_review_ids == (review.id,)
    subscription, _ = await service.subscribe()
    events = aiter(subscription)
    try:
        if closure == "native":
            await connection.close()
        else:
            await service._close_connection(await service._owner(session.id))
        event = await asyncio.wait_for(anext(events), 3)
        assert event.data["active_review_ids"] == []
        await packets.review(status="approved")
        detail = await service.get_detail(session.id)
        assert detail.items == before.items
        assert detail.active_review_ids == ()
        assert detail.run.status == "completed"
        await service.close()
        restarted = CodeService(
            SQLiteCodeStore(directory / "code.db", directory / "code.lock"), harness
        )
        await restarted.start()
        try:
            restored = await restarted.get_detail(session.id)
            assert restored.items == before.items
            assert restored.active_review_ids == ()
        finally:
            await restarted.close()
    finally:
        await subscription.aclose()


@pytest.mark.asyncio
async def test_mode_is_captured_per_turn_and_does_not_recreate_native_process(code):
    service, harness, _ = code
    session = await session_for(code)
    modes = ["full_access", "config", "ask", "auto_review"]
    for ordinal, mode in enumerate(modes, 1):
        session = await service.update_settings(
            session.id, session.revision, {"mode": mode}
        )
        run = await service.send(
            session.id,
            new_id(),
            session.revision,
            "hello",
            expected_epoch=service.epoch,
        )
        assert run.mode == mode
        await asyncio.wait_for(harness.wait_inputs(ordinal), 3)
        current = (await service.get_detail(session.id)).session
        with pytest.raises(CodeConflictError):
            await service.update_settings(current.id, current.revision, {"mode": "ask"})
        await harness.connections[0].finish(f"turn-{ordinal}")
        await service.wait_idle(session.id)
        session = (await service.get_detail(session.id)).session
    assert len(harness.connections) == 1
    assert harness.connections[0].modes == modes
    assert session.native_permission_defaults == harness.permission_defaults


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_output", [False, True])
async def test_reobserved_child_review_publishes_liveness_on_new_native_connection(
    code,
    pending_output,
):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "Delegate", expected_epoch=service.epoch
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    first = harness.connections[0]
    packets = CodexPackets(first)
    await packets.spawn()
    await packets.review()
    review = (await service.get_detail(session.id)).items[-1]
    await first.finish("turn-1")
    harness.configurations[harness.model] = "replacement"
    session = (await service.get_detail(session.id)).session
    await service.send(
        session.id, new_id(), session.revision, "Continue", expected_epoch=service.epoch
    )
    await asyncio.wait_for(harness.wait_inputs(2), 3)
    second = harness.connections[1]
    if not pending_output:
        await second.finish("turn-2")
    assert (await service.get_detail(session.id)).active_review_ids == ()
    packets = CodexPackets(second)
    await packets.spawn()
    subscription, _ = await service.subscribe()
    try:
        if pending_output:
            await second.text("turn-2", "reply", "In progress")
        await packets.review()
        event = await asyncio.wait_for(anext(aiter(subscription)), 3)
        while event.data["active_review_ids"] != [review.id]:
            event = await asyncio.wait_for(anext(aiter(subscription)), 3)
        assert event.data["active_review_ids"] == [review.id]
        assert event.data.get("item", {}).get("id") == review.id
        assert event.data["runs"][0]["id"] == review.run_id
        detail = await service.get_detail(session.id)
        assert next(item for item in detail.items if item.id == review.id) == review
        assert detail.run.status == ("running" if pending_output else "completed")
    finally:
        await subscription.aclose()


@pytest.mark.asyncio
async def test_forced_interrupt_clears_pending_child_review_liveness(code, monkeypatch):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "Delegate", expected_epoch=service.epoch
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    connection = harness.connections[0]
    packets = CodexPackets(connection)
    await packets.spawn()
    await packets.review()
    assert (await service.get_detail(session.id)).active_review_ids

    async def unavailable(_turn_id):
        raise CodeUnavailableError("Native interrupt transport failed")

    monkeypatch.setattr(connection, "interrupt", unavailable)
    await service.stop(session.id, (await service.get_detail(session.id)).run.id)
    await asyncio.wait_for(service.wait_idle(session.id), 3)
    detail = await service.get_detail(session.id)
    assert detail.active_review_ids == ()
    assert detail.run.status == "interrupted"
    assert not detail.items[-1].complete


@pytest.mark.asyncio
async def test_mode_change_and_send_compete_for_one_revision(code):
    service, harness, _ = code
    session = await session_for(code)
    results = await asyncio.gather(
        service.update_settings(session.id, session.revision, {"mode": "full_access"}),
        service.send(
            session.id,
            new_id(),
            session.revision,
            "hello",
            expected_epoch=service.epoch,
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(result, CodeConflictError) for result in results) == 1
    detail = await service.get_detail(session.id)
    if detail.run:
        assert detail.run.mode == "config"
        await asyncio.wait_for(harness.started.wait(), 3)
        await harness.connections[0].finish("turn-1")
    else:
        assert detail.session.mode == "full_access"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "plan", "never", {}, 1])
async def test_invalid_mode_is_rejected_without_changing_session(code, mode):
    service, _, _ = code
    session = await session_for(code)
    with pytest.raises(CodeValidationError):
        await service.update_settings(session.id, session.revision, {"mode": mode})
    assert (await service.get_detail(session.id)).session == session


@pytest.mark.asyncio
async def test_native_defaults_are_saved_before_input_and_preserved_on_resume(code):
    service, harness, _ = code
    session = await session_for(code)
    original = {"policy": "configured-original"}
    harness.permission_defaults = original
    session = await service.update_settings(
        session.id, session.revision, {"mode": "full_access"}
    )
    await service.send(
        session.id, new_id(), session.revision, "first", expected_epoch=service.epoch
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    saved = await service._store.get_session(session.id)
    assert saved.native_permission_defaults == original
    assert harness.connections[0].defaults == [original]
    await harness.connections[0].finish("turn-1")
    await service.wait_idle(session.id)
    harness.permission_defaults = {"policy": "current-override"}
    harness.configurations[harness.model] = "recreate-process"
    session = (await service.get_detail(session.id)).session
    session = await service.update_settings(
        session.id, session.revision, {"mode": "config"}
    )
    await service.send(
        session.id, new_id(), session.revision, "second", expected_epoch=service.epoch
    )
    await asyncio.wait_for(harness.wait_inputs(2), 3)
    assert len(harness.connections) == 2
    assert harness.connections[1].defaults == [original]
    await harness.connections[1].finish("turn-2")


@pytest.mark.asyncio
async def test_failure_saving_native_defaults_never_submits_input(code, monkeypatch):
    service, harness, _ = code
    session = await session_for(code)
    save = service._store.save_progress

    async def reject_defaults(session, *args, **kwargs):
        if session.native_permission_defaults is not None:
            raise CodeUnavailableError("Unable to save permission defaults")
        await save(session, *args, **kwargs)

    monkeypatch.setattr(service._store, "save_progress", reject_defaults)
    await service.send(
        session.id, new_id(), session.revision, "first", expected_epoch=service.epoch
    )
    await asyncio.wait_for(service.wait_idle(session.id), 3)
    assert not any(connection.inputs for connection in harness.connections)
    assert not (await service._store.latest_run(session.id)).submission_started


@pytest.mark.asyncio
async def test_native_reviews_and_notices_keep_identity_order_and_survive_restart(code):
    service, harness, directory = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "hello", expected_epoch=service.epoch
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    connection = harness.connections[0]
    protocol = CodexProtocol(connection.generation)
    payload = {
        "threadId": connection.thread_id,
        "turnId": "turn-1",
        "targetItemId": "same-command",
        "action": {"command": "echo test"},
        "review": {"status": "inProgress"},
    }
    for review_id in ("one", "two"):
        await connection.sink(
            protocol.notification(
                "item/autoApprovalReview/started", {**payload, "reviewId": review_id}
            )
        )
    await connection.text("turn-1", "reply", "After reviews", complete=True)
    before = await service.get_detail(session.id)
    for review_id in ("two", "one"):
        await connection.sink(
            protocol.notification(
                "item/autoApprovalReview/completed",
                {
                    **payload,
                    "reviewId": review_id,
                    "review": {"status": "denied", "rationale": "Native refusal"},
                },
            )
        )
    await connection.sink(
        protocol.notification(
            "guardianWarning",
            {"threadId": connection.thread_id, "message": "Native warning"},
        )
    )
    await connection.sink(
        HarnessEvent("old-generation", connection.thread_id, "notice", message="stale")
    )
    await connection.finish("turn-1")
    await service.wait_idle(session.id)
    detail = await service.get_detail(session.id)
    assert [item.id for item in detail.items[:4]] == [item.id for item in before.items]
    assert [item.kind for item in detail.items] == [
        "user",
        "auto_review",
        "auto_review",
        "text",
        "notice",
    ]
    assert all(item.title == "Auto-review: Denied" for item in detail.items[1:3])
    assert detail.run.status == "completed"
    assert detail.items[-1].text == "Native warning"
    await service.close()
    restarted = CodeService(
        SQLiteCodeStore(directory / "code.db", directory / "code.lock"), harness
    )
    await restarted.start()
    try:
        assert (await restarted.get_detail(session.id)).items == detail.items
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_selected_model_and_effort_are_session_settings_captured_on_send(code):
    service, harness, _ = code
    harness.configurations["provider/other"] = "other"
    session = await session_for(code)
    for effort in (None, "high", None):
        session = await service.update_settings(
            session.id,
            session.revision,
            {"model": "provider/other", "reasoning_effort": effort},
        )
        run = await service.send(
            session.id,
            new_id(),
            session.revision,
            "hello",
            expected_epoch=service.epoch,
        )
        await harness.wait_inputs(run.ordinal)
        connection = harness.connections[0]
        with pytest.raises(CodeConflictError):
            await service.update_settings(
                session.id, session.revision + 1, {"reasoning_effort": "low"}
            )
        await connection.finish(f"turn-{run.ordinal}")
        session = (await service.get_detail(session.id)).session
    assert connection.efforts == ["medium", "high", "medium"]
    assert {value[2] for value in connection.inputs} == {"provider/other"}
    assert len(harness.connections) == 1
    harness.configurations.pop("provider/other")
    with pytest.raises(CodeValidationError, match="unavailable"):
        await service.send(
            session.id,
            new_id(),
            session.revision,
            "kept draft",
            expected_epoch=service.epoch,
        )
    assert (await service.get_detail(session.id)).session.model == "provider/other"


@pytest.mark.asyncio
async def test_catalog_effort_removal_requires_choice_but_model_change_resolves_default(
    code,
):
    service, harness, _ = code
    session = await session_for(code)
    session = await service.update_settings(
        session.id, session.revision, {"reasoning_effort": "xhigh"}
    )
    harness.efforts = ("low", "medium")
    with pytest.raises(CodeValidationError, match="effort"):
        await service.send(
            session.id,
            new_id(),
            session.revision,
            "hello",
            expected_epoch=service.epoch,
        )
    harness.configurations["provider/other"] = "other"
    session = await service.update_settings(
        session.id, session.revision, {"model": "provider/other"}
    )
    assert session.reasoning_effort is None
    assert session.model == "provider/other"


@pytest.mark.asyncio
async def test_failed_outcome_is_durable_once_after_partial_output_and_older_pagination(
    code,
):
    service, harness, _ = code
    session = await session_for(code)
    run = await service.send(
        session.id, new_id(), session.revision, "first", expected_epoch=service.epoch
    )
    await harness.started.wait()
    connection = harness.connections[0]
    for kind in ("text", "reasoning", "tool"):
        await connection.text("turn-1", kind, f"partial {kind}", kind=kind)
    await connection.finish("turn-1", "failed", "Provider rejected the request")
    await connection.finish("turn-1", "failed", "duplicate")
    detail = await service.get_detail(session.id)
    assert detail.session.error is None
    assert detail.runs == (detail.run,)
    assert detail.run.error == "Provider rejected the request"
    assert [item.text for item in detail.items] == [
        "first",
        "partial text",
        "partial reasoning",
        "partial tool",
    ]
    await service.send(
        session.id,
        new_id(),
        detail.session.revision,
        "second",
        expected_epoch=service.epoch,
    )
    await harness.wait_inputs(2)
    for index in range(55):
        await connection.text("turn-2", str(index), str(index), complete=True)
    active = await service.get_detail(session.id)
    assert active.items[0].text == "first"
    assert len(active.items) == 60
    await connection.finish("turn-2")
    latest = await service.get_detail(session.id)
    assert run.id not in {value.id for value in latest.runs}
    older = await service.get_detail(session.id, before=latest.next_before)
    assert (
        next(value for value in older.runs if value.id == run.id).error
        == detail.run.error
    )


@pytest.mark.asyncio
async def test_retry_notice_does_not_finish_run_and_stale_error_is_ignored(code):
    service, harness, directory = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "hello", expected_epoch=service.epoch
    )
    await harness.started.wait()
    connection = harness.connections[0]
    await connection.text("turn-1", "before", "Before retry", complete=True)
    await connection.sink(
        HarnessEvent(
            connection.generation,
            connection.thread_id,
            "error",
            turn_id="turn-1",
            message="retry",
            will_retry=True,
        )
    )
    detail = await service.get_detail(session.id)
    assert detail.run.status == "running"
    assert [item.kind for item in detail.items] == ["user", "text", "notice"]
    assert detail.items[-1].text == "Retrying… retry"
    assert detail.items[-1].sequence > detail.items[-2].sequence
    await connection.sink(
        HarnessEvent(
            connection.generation, connection.thread_id, "error", turn_id="turn-1"
        )
    )
    detail = await service.get_detail(session.id)
    assert detail.run.status == "running"
    assert detail.items[-1].text == "Codex reported an error."
    await connection.text("turn-1", "text", "success", complete=True)
    await connection.finish("turn-1")
    subscription, _ = await service.subscribe()
    cursor = service.cursor
    await connection.sink(
        HarnessEvent(
            connection.generation,
            connection.thread_id,
            "error",
            turn_id="turn-1",
            message="stale",
        )
    )
    assert service.cursor == cursor
    await subscription.aclose()
    assert (await service.get_detail(session.id)).run.error is None
    detail = await service.get_detail(session.id)
    await service.close()
    restarted = CodeService(
        SQLiteCodeStore(directory / "code.db", directory / "code.lock"), harness
    )
    await restarted.start()
    try:
        assert (await restarted.get_detail(session.id)).items == detail.items
    finally:
        await restarted.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("client_id", [True, False])
async def test_acknowledgement_loss_recovers_old_turn_before_newer_send(
    code, client_id
):
    service, harness, _ = code
    session = await session_for(code)
    harness.start_gate.clear()
    first = await service.send(
        session.id, new_id(), session.revision, "first", expected_epoch=service.epoch
    )
    await harness.submitted.wait()
    connection = harness.connections[0]
    harness.histories[connection.thread_id] = [
        ItemUpdate(
            "turn-1",
            "native-user",
            "user",
            text="first",
            complete=True,
            client_id=first.id if client_id else None,
        ),
        ItemUpdate("turn-1", "tail", "text", text="recovered tail", complete=True),
    ]
    await connection.close()
    await service.wait_idle(session.id)
    detail = await service.get_detail(session.id)
    assert detail.run.status == "failed"
    await service.send(
        session.id,
        new_id(),
        detail.session.revision,
        "second",
        expected_epoch=service.epoch,
    )
    await harness.wait_inputs(2)
    recovered = await service.get_detail(session.id)
    assert [item.text for item in recovered.items] == [
        "first",
        "recovered tail",
        "second",
    ]
    assert recovered.runs[0].status == "failed"
    assert recovered.items[0].id == first.id


@pytest.mark.asyncio
async def test_prompt_membership_and_deleted_subscription_are_authoritative(code):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "first", expected_epoch=service.epoch
    )
    await harness.started.wait()
    connection = harness.connections[0]
    await connection.prompt(0, turn_id=None)
    detail = await service.get_detail(session.id)
    assert detail.active_prompt_ids == (detail.prompts[0].id,)
    await connection.resolve(0)
    assert (await service.get_detail(session.id)).active_prompt_ids == ()
    await connection.finish("turn-1")
    session = (await service.get_detail(session.id)).session
    await service.delete_session(session.id, session.revision)
    await service.wait_idle(session.id)
    subscription, ready = await service.subscribe()
    assert ready["sessions"] == []
    assert session.id not in service._owners
    await subscription.aclose()


@pytest.mark.asyncio
async def test_ambiguous_native_history_fails_recovery_without_resending_input(code):
    service, harness, _ = code
    session = await session_for(code)
    harness.start_gate.clear()
    await service.send(
        session.id, new_id(), session.revision, "first", expected_epoch=service.epoch
    )
    await harness.submitted.wait()
    connection = harness.connections[0]
    harness.histories[connection.thread_id] = [
        ItemUpdate(turn_id, "item", "text", text="unmatched", complete=True)
        for turn_id in ("unknown-1", "unknown-2")
    ]
    await connection.close()
    await service.wait_idle(session.id)
    detail = await service.get_detail(session.id)
    await service.send(
        session.id,
        new_id(),
        detail.session.revision,
        "second",
        expected_epoch=service.epoch,
    )
    await service.wait_idle(session.id)
    detail = await service.get_detail(session.id)
    assert detail.run.status == "failed"
    assert "could not be matched" in detail.run.error
    assert [item.text for item in detail.items] == ["first", "second"]
    assert sum(len(connection.inputs) for connection in harness.connections) == 1


@pytest.mark.asyncio
async def test_settings_and_send_compete_for_the_same_revision(code):
    service, harness, _ = code
    session = await session_for(code)
    outcomes = await asyncio.gather(
        service.update_settings(
            session.id, session.revision, {"reasoning_effort": "high"}
        ),
        service.send(
            session.id,
            new_id(),
            session.revision,
            "hello",
            expected_epoch=service.epoch,
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(result, CodeConflictError) for result in outcomes) == 1
    detail = await service.get_detail(session.id)
    assert detail.session.revision == session.revision + 1
    if detail.run:
        assert detail.run.reasoning_effort == "medium"
    else:
        assert detail.session.reasoning_effort == "high"
        assert harness.connections == []


@pytest.mark.asyncio
async def test_old_service_epoch_cannot_admit_a_new_send_but_can_read_receipt(code):
    service, harness, _ = code
    session = await session_for(code)
    with pytest.raises(CodeConflictError, match="restarted"):
        await service.send(
            session.id, new_id(), session.revision, "stale", expected_epoch="previous"
        )
    assert harness.connections == []
    run = await service.send(
        session.id, new_id(), session.revision, "accepted", expected_epoch=service.epoch
    )
    repeated = await service.send(
        session.id, run.id, session.revision, "accepted", expected_epoch="previous"
    )
    assert repeated.id == run.id


@pytest.mark.asyncio
async def test_poorer_native_history_keeps_captured_source_and_reasoning(code):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "first", expected_epoch=service.epoch
    )
    await harness.started.wait()
    connection = harness.connections[0]
    original = ItemUpdate(
        "turn-1",
        "reasoning",
        "reasoning",
        text="Full captured reasoning",
        complete=True,
        raw={
            "item": {
                "id": "reasoning",
                "summary": ["original"],
                "opaque": {"signature": "retained"},
            },
            "stream": {"reasoning": {"0": "full"}},
        },
    )
    await connection.sink(
        HarnessEvent(
            connection.generation,
            connection.thread_id,
            "item",
            turn_id="turn-1",
            item=original,
        )
    )
    await connection.finish("turn-1")
    harness.histories[connection.thread_id] = [
        ItemUpdate(
            "turn-1",
            "reasoning",
            "reasoning",
            text="Shorter native history",
            complete=True,
            raw={"item": {"id": "reasoning", "summary": []}, "stream": {}},
        )
    ]
    harness.configurations[harness.model] = "changed"
    detail = await service.get_detail(session.id)
    await service.send(
        session.id,
        new_id(),
        detail.session.revision,
        "next",
        expected_epoch=service.epoch,
    )
    await harness.wait_inputs(2)
    retained = next(
        item
        for item in (await service.get_detail(session.id)).items
        if item.native_item_id == "reasoning"
    )
    assert retained.text == original.text
    assert retained.raw == original.raw


@pytest.mark.asyncio
async def test_storage_start_failure_is_isolated_to_code(tmp_path):
    class BrokenStore(SQLiteCodeStore):
        async def start(self):
            raise CodeUnavailableError("Code disk unavailable")

    service = CodeService(
        BrokenStore(tmp_path / "code.db", tmp_path / "code.lock"), FakeHarness()
    )
    try:
        await service.start()
        assert service.availability() == (False, "Code disk unavailable")
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_shutdown_settles_run_when_native_interrupt_never_acknowledges(code):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id,
        new_id(),
        session.revision,
        "stop on shutdown",
        expected_epoch=service.epoch,
    )
    await harness.started.wait()
    harness.interrupt_gate.clear()
    await asyncio.wait_for(service.close(), 8)
    assert harness.connections[0].closed


@pytest.mark.asyncio
async def test_storage_failure_retires_native_work_even_with_pending_prompt(
    code, monkeypatch
):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "prompt", expected_epoch=service.epoch
    )
    await harness.started.wait()
    connection = harness.connections[0]
    await connection.prompt(1)

    async def fail_save(*args, **kwargs):
        raise CodeUnavailableError("Code disk unavailable")

    monkeypatch.setattr(service._store, "save_progress", fail_save)
    await connection.text("turn-1", "text", "unsaved", complete=True)
    await asyncio.wait_for(service.wait_idle(session.id), 3)
    assert connection.closed
    with pytest.raises(CodeUnavailableError):
        await service.get_detail(session.id)


@pytest.mark.asyncio
async def test_same_send_id_in_two_sessions_has_independent_receipts(code, monkeypatch):
    service, harness, _ = code
    first, second = await session_for(code), await session_for(code)
    original = service._store.get_run
    looked_up = 0
    barrier = asyncio.Event()

    async def simultaneous_lookup(session_id, run_id):
        nonlocal looked_up
        result = await original(session_id, run_id)
        looked_up += 1
        if looked_up == 2:
            barrier.set()
        await barrier.wait()
        return result

    monkeypatch.setattr(service._store, "get_run", simultaneous_lookup)
    operation = new_id()
    outcomes = await asyncio.gather(
        *(
            service.send(
                session.id,
                operation,
                session.revision,
                "once",
                expected_epoch=service.epoch,
            )
            for session in (first, second)
        ),
        return_exceptions=True,
    )
    assert {
        value.session_id for value in outcomes if not isinstance(value, BaseException)
    } == {first.id, second.id}
    await harness.wait_inputs(2)
    assert sum(len(connection.inputs) for connection in harness.connections) == 2


async def idle_native(code):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "one", expected_epoch=service.epoch
    )
    await harness.started.wait()
    await harness.connections[0].finish("turn-1")
    return (await service.get_detail(session.id)).session


@pytest.mark.asyncio
async def test_deletion_fences_commands_and_repeated_delete_has_one_native_call(code):
    service, harness, folder = code
    session = await idle_native(code)
    project = folder / "project.txt"
    project.write_text("keep")
    harness.delete_gate.clear()
    await service.delete_session(session.id, session.revision)
    await harness.deleting.wait()
    await service.delete_session(session.id, session.revision)
    with pytest.raises(CodeConflictError):
        await service.send(
            session.id, new_id(), session.revision, "late", expected_epoch=service.epoch
        )
    with pytest.raises(CodeConflictError):
        await service.update_settings(session.id, session.revision, {"title": "late"})
    harness.delete_gate.set()
    await service.wait_idle(session.id)
    assert harness.connections[0].deleted == [session.native_thread_id]
    assert project.read_text() == "keep"
    assert not (await service.list_sessions()).sessions


@pytest.mark.asyncio
async def test_lost_native_delete_ack_reconciles_without_repeating_delete(code):
    service, harness, _ = code
    session = await idle_native(code)
    harness.delete_before_error = True
    harness.delete_error = CodeUnavailableError("Delete acknowledgement lost")
    await service.delete_session(session.id, session.revision)
    await service.wait_idle(session.id)
    assert not (await service.list_sessions()).sessions
    assert sum(len(connection.deleted) for connection in harness.connections) == 1


@pytest.mark.asyncio
async def test_local_delete_commit_failure_keeps_fence_and_can_reconcile(
    code, monkeypatch
):
    service, harness, _ = code
    session = await idle_native(code)
    original = service._store.delete

    async def failing_delete(session_id):
        raise CodeUnavailableError("Delete commit failed")

    monkeypatch.setattr(service._store, "delete", failing_delete)
    await service.delete_session(session.id, session.revision)
    await asyncio.wait_for(service.wait_idle(session.id), 3)
    detail = await service.get_detail(session.id)
    assert detail.session.status == "delete_uncertain"
    assert detail.items
    with pytest.raises(CodeConflictError):
        await service.send(
            session.id,
            new_id(),
            detail.session.revision,
            "late",
            expected_epoch=service.epoch,
        )
    monkeypatch.setattr(service._store, "delete", original)
    await service.delete_session(session.id, detail.session.revision)
    await service.wait_idle(session.id)
    assert not (await service.list_sessions()).sessions
    assert sum(len(connection.deleted) for connection in harness.connections) == 1


@pytest.mark.asyncio
async def test_late_old_native_events_cannot_claim_a_new_turn_before_ack(code):
    service, harness, _ = code
    session = await idle_native(code)
    connection = harness.connections[0]
    harness.start_gate.clear()
    harness.started.clear()
    await service.send(
        session.id, new_id(), session.revision, "next", expected_epoch=service.epoch
    )
    await harness.wait_inputs(2)
    await connection.sink(
        HarnessEvent(
            connection.generation,
            connection.thread_id,
            "turn_started",
            turn_id="turn-1",
        )
    )
    await connection.text("turn-1", "late", "obsolete", complete=True)
    await connection.finish("turn-1")
    detail = await service.get_detail(session.id)
    assert detail.run.status == "preparing"
    assert not any(item.text == "obsolete" for item in detail.items)
    harness.start_gate.set()
    await harness.started.wait()
    detail = await service.get_detail(session.id)
    assert detail.run.native_turn_id == "turn-2"


@pytest.mark.asyncio
async def test_creation_identity_received_during_cleanup_is_saved(code, monkeypatch):
    service, harness, _ = code
    session = await session_for(code)
    original_close = FakeConnection.close

    async def failed_create(connection):
        raise CodeUnavailableError("Native creation acknowledgement timed out")

    async def close_with_late_reply(connection):
        connection.thread_id = "late-native-id"
        harness.histories[connection.thread_id] = []
        await original_close(connection)

    monkeypatch.setattr(FakeConnection, "create_thread", failed_create)
    monkeypatch.setattr(FakeConnection, "close", close_with_late_reply)
    await service.send(
        session.id,
        new_id(),
        session.revision,
        "uncertain create",
        expected_epoch=service.epoch,
    )
    await service.wait_idle(session.id)
    detail = await service.get_detail(session.id)
    assert detail.session.native_thread_id == "late-native-id"
    assert detail.run.status == "failed"
    assert not harness.connections[0].inputs


@pytest.mark.asyncio
async def test_failed_stop_persistence_closes_work_without_waiting_for_more_output(
    code, monkeypatch
):
    service, harness, _ = code
    session = await session_for(code)
    run = await service.send(
        session.id,
        new_id(),
        session.revision,
        "quiet work",
        expected_epoch=service.epoch,
    )
    await harness.started.wait()

    async def failed_save(*args, **kwargs):
        raise CodeUnavailableError("Disk failed")

    monkeypatch.setattr(service._store, "save_progress", failed_save)
    with pytest.raises(CodeUnavailableError):
        await service.stop(session.id, run.id)
    await asyncio.wait_for(service.wait_idle(session.id), 3)
    assert harness.connections[0].closed


@pytest.mark.asyncio
async def test_create_is_lazy_and_retry_does_not_duplicate(code):
    service, harness, directory = code
    session = await session_for(code)
    repeated = await service.create_session(session.id, str(directory))
    assert repeated == session
    assert harness.connections == []
    assert len((await service.list_sessions()).sessions) == 1


@pytest.mark.asyncio
async def test_competing_sends_admit_only_one_and_repeat_returns_receipt(code):
    service, harness, _ = code
    session = await session_for(code)
    command = new_id()
    results = await asyncio.gather(
        service.send(
            session.id, command, session.revision, "first", expected_epoch=service.epoch
        ),
        service.send(
            session.id,
            new_id(),
            session.revision,
            "second",
            expected_epoch=service.epoch,
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(result, CodeConflictError) for result in results) == 1
    run = results[0]
    assert not isinstance(run, BaseException)
    await harness.started.wait()
    repeat = await service.send(
        session.id, command, session.revision, "first", expected_epoch=service.epoch
    )
    assert repeat.id == run.id
    assert harness.connections[0].inputs == [(command, "first", "provider/model")]
    with pytest.raises(CodeConflictError):
        await service.send(
            session.id,
            command,
            session.revision,
            "changed",
            expected_epoch=service.epoch,
        )


@pytest.mark.asyncio
async def test_http_cancellation_does_not_cancel_native_creation(code):
    service, harness, _ = code
    session = await session_for(code)
    harness.creation_gate.clear()
    command = new_id()
    await service.send(
        session.id, command, session.revision, "continue", expected_epoch=service.epoch
    )
    await harness.creating.wait()
    # There is no HTTP-owned native waiter left to cancel after admission.
    detail = await service.get_detail(session.id)
    assert detail.run.id == command
    harness.creation_gate.set()
    await harness.started.wait()
    detail = await service.get_detail(session.id)
    assert detail.session.native_thread_id == "native-1"
    assert harness.connections[0].inputs[0][0] == command


@pytest.mark.asyncio
async def test_stop_before_native_ack_targets_that_run_only(code):
    service, harness, _ = code
    session = await session_for(code)
    harness.start_gate.clear()
    run = await service.send(
        session.id, new_id(), session.revision, "first", expected_epoch=service.epoch
    )
    await harness.submitted.wait()
    await service.stop(session.id, run.id)
    harness.start_gate.set()
    await harness.interrupted.wait()
    connection = harness.connections[0]
    assert connection.interrupts == ["turn-1"]
    await connection.finish("turn-1", "interrupted")
    detail = await service.get_detail(session.id)
    assert detail.run.status == "interrupted"
    second = await service.send(
        session.id,
        new_id(),
        detail.session.revision,
        "second",
        expected_epoch=service.epoch,
    )
    await harness.wait_inputs(2)
    await service.stop(session.id, run.id)
    assert second.id != run.id
    assert connection.interrupts == ["turn-1"]


@pytest.mark.asyncio
async def test_stop_during_creation_prevents_native_submission(code):
    service, harness, _ = code
    session = await session_for(code)
    harness.creation_gate.clear()
    run = await service.send(
        session.id,
        new_id(),
        session.revision,
        "do not run",
        expected_epoch=service.epoch,
    )
    await harness.creating.wait()
    await service.stop(session.id, run.id)
    harness.creation_gate.set()
    await service.wait_idle(session.id)
    detail = await service.get_detail(session.id)
    assert detail.run.status == "interrupted"
    assert detail.session.native_thread_id == "native-1"
    assert harness.connections[0].inputs == []


@pytest.mark.asyncio
async def test_browser_observers_are_independent_of_work(code):
    service, harness, _ = code
    session = await session_for(code)
    first, _ = await service.subscribe()
    second, _ = await service.subscribe()
    await service.send(
        session.id,
        new_id(),
        session.revision,
        "keep running",
        expected_epoch=service.epoch,
    )
    await harness.started.wait()
    await first.aclose()
    await second.aclose()
    connection = harness.connections[0]
    await connection.text("turn-1", "answer", "Still running", complete=True)
    await connection.finish("turn-1")
    detail = await service.get_detail(session.id)
    assert detail.items[-1].text == "Still running"
    assert detail.run.status == "completed"
    assert not connection.closed


@pytest.mark.asyncio
async def test_idle_delete_removes_both_histories_and_blocks_late_create(code):
    service, harness, directory = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "hello", expected_epoch=service.epoch
    )
    await harness.started.wait()
    with pytest.raises(CodeConflictError):
        await service.delete_session(session.id, session.revision)
    assert harness.connections[0].deleted == []
    await harness.connections[0].finish("turn-1")
    detail = await service.get_detail(session.id)
    await service.delete_session(session.id, detail.session.revision)
    await service.wait_idle(session.id)
    assert harness.connections[0].deleted == ["native-1"]
    assert (await service.list_sessions()).sessions == ()
    with pytest.raises(CodeConflictError):
        await service.create_session(session.id, str(directory))


@pytest.mark.asyncio
async def test_restart_preserves_receipt_and_does_not_replay_input(tmp_path):
    harness = FakeHarness()
    database, lock = tmp_path / "code.db", tmp_path / "code.lock"
    first = CodeService(SQLiteCodeStore(database, lock), harness)
    await first.start()
    session = await first.create_session(new_id(), str(tmp_path))
    run = await first.send(
        session.id, new_id(), session.revision, "once", expected_epoch=first.epoch
    )
    await harness.started.wait()
    await first.close()
    second = CodeService(SQLiteCodeStore(database, lock), harness)
    await second.start()
    try:
        repeat = await second.send(
            session.id, run.id, session.revision, "once", expected_epoch=second.epoch
        )
        assert repeat.id == run.id
        assert repeat.status == "interrupted"
        assert len(harness.connections) == 1
        assert harness.connections[0].inputs == [(run.id, "once", "provider/model")]
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_two_answers_claim_one_native_prompt(code):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "approval", expected_epoch=service.epoch
    )
    await harness.started.wait()
    connection = harness.connections[0]
    await connection.prompt(0)
    prompt = (await service.get_detail(session.id)).prompts[0]
    harness.answer_gate.clear()
    results = await asyncio.gather(
        service.answer(session.id, prompt.id, new_id(), {"choice": "accept"}),
        service.answer(session.id, prompt.id, new_id(), {"choice": "decline"}),
        return_exceptions=True,
    )
    assert sum(isinstance(result, CodeConflictError) for result in results) == 1
    await harness.answering.wait()
    harness.answer_gate.set()
    await harness.answered.wait()
    assert len(connection.answers) == 1
    assert (await service.get_detail(session.id)).prompts[0].status == "resolved"


@pytest.mark.asyncio
async def test_prompt_arrival_has_one_durable_position_despite_repeated_native_item(
    code,
):
    service, harness, directory = code
    session = await session_for(code)
    run = await service.send(
        session.id, new_id(), session.revision, "Work", expected_epoch=service.epoch
    )
    await harness.started.wait()
    connection = harness.connections[0]
    await connection.text("turn-1", "shared-tool", "Before approval", kind="tool")
    subscription, _ = await service.subscribe()
    try:
        for request_id in (1, 1, "1"):
            prompt = PromptRequest(
                request_id, "approval", {}, {}, "turn-1", "shared-tool"
            )
            connection.requests[request_id] = prompt
            await connection.sink(
                HarnessEvent(
                    connection.generation,
                    connection.thread_id,
                    "prompt",
                    turn_id="turn-1",
                    prompt=prompt,
                )
            )
        detail = await service.get_detail(session.id)
        entries = [item for item in detail.items if item.kind == "prompt"]
        assert len(entries) == len(detail.prompts) == 2
        assert [item.sequence for item in entries] == [3, 4]
        assert {item.id for item in entries} == {prompt.id for prompt in detail.prompts}
        events = subscription.__aiter__()
        async with asyncio.timeout(2):
            for _ in range(2):
                event = await anext(events)
                if event.event == "item.updated":
                    event = await anext(events)
                assert event.event == "prompt.updated"
                assert event.data["item"]["id"] == event.data["prompt"]["id"]
                assert event.data["runs"][0]["id"] == run.id
        for request_id in (1, "1"):
            await connection.resolve(request_id)
        await connection.text(
            "turn-1", "shared-tool", "Complete output", kind="tool", complete=True
        )
        await connection.text("turn-1", "after", "After approvals", complete=True)
        await connection.finish("turn-1")
        await service.close()
        restored = CodeService(
            SQLiteCodeStore(directory / "code.db", directory / "code.lock"), harness
        )
        await restored.start()
        try:
            saved = await restored.get_detail(session.id)
            assert [item for item in saved.items if item.kind == "prompt"] == entries
            assert [item.sequence for item in saved.items] == [1, 2, 3, 4, 5]
            assert all(prompt.status == "resolved" for prompt in saved.prompts)
            await restored.send(
                session.id,
                new_id(),
                saved.session.revision,
                "Continue",
                expected_epoch=restored.epoch,
            )
            await harness.wait_inputs(2)
            await harness.connections[-1].finish("turn-2")
            recovered = await restored.get_detail(session.id)
            assert [
                item for item in recovered.items if item.kind == "prompt"
            ] == entries
            assert (
                len(
                    [
                        item
                        for item in recovered.items
                        if item.native_item_id == "shared-tool"
                    ]
                )
                == 1
            )
        finally:
            await restored.close()
    finally:
        await subscription.aclose()


@pytest.mark.asyncio
async def test_active_prompt_outside_page_keeps_its_entry_run_and_older_cursor(code):
    service, harness, _ = code
    session = await session_for(code)
    run = await service.send(
        session.id, new_id(), session.revision, "Work", expected_epoch=service.epoch
    )
    await harness.started.wait()
    connection = harness.connections[0]
    await connection.prompt(0, turn_id=None)
    prompt = (await service.get_detail(session.id)).prompts[0]
    for index in range(55):
        await connection.text("turn-1", str(index), f"Output {index}", complete=True)
    await connection.finish("turn-1")
    newest = await service.get_detail(session.id)
    assert newest.next_before is not None
    assert len(newest.items) == 51
    assert newest.items[0].id == prompt.id
    assert newest.items[0].run_id == run.id
    assert newest.active_prompt_ids == (prompt.id,)
    await connection.resolve(0)
    resolved = await service.get_detail(session.id)
    assert resolved.active_prompt_ids == () and resolved.prompts == ()
    assert resolved.next_before == newest.next_before
    older = await service.get_detail(session.id, before=newest.next_before)
    assert older.prompts[0].id == prompt.id and older.prompts[0].status == "resolved"
    by_id = {item.id: item for item in (*older.items, *newest.items)}
    assert sorted(item.sequence for item in by_id.values()) == list(range(1, 58))


@pytest.mark.asyncio
async def test_prompt_during_thread_creation_has_a_saved_run_without_native_ids(code):
    service, harness, _ = code
    session = await session_for(code)
    harness.creation_gate.clear()
    run = await service.send(
        session.id, new_id(), session.revision, "Start", expected_epoch=service.epoch
    )
    await harness.creating.wait()
    connection = harness.connections[0]
    try:
        await connection.prompt(0, turn_id=None)
        detail = await service.get_detail(session.id)
        assert detail.run.native_turn_id is None
        assert detail.items[-1].id == detail.prompts[0].id
        assert detail.items[-1].run_id == run.id
        assert detail.items[-1].sequence == 2
        assert detail.prompts[0].native_turn_id is None
        await connection.resolve(0)
    finally:
        harness.creation_gate.set()


@pytest.mark.asyncio
async def test_native_resolution_while_answer_is_waiting_never_resurrects_prompt(code):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "approval", expected_epoch=service.epoch
    )
    await harness.started.wait()
    connection = harness.connections[0]
    await connection.prompt(0)
    prompt = (await service.get_detail(session.id)).prompts[0]
    harness.answer_gate.clear()
    await service.answer(session.id, prompt.id, new_id(), {"choice": "accept"})
    await harness.answering.wait()
    await connection.resolve(0)
    harness.answer_gate.set()
    await harness.answered.wait()
    assert connection.answers == []
    assert (await service.get_detail(session.id)).prompts[0].status == "resolved"


@pytest.mark.asyncio
async def test_streaming_does_not_invalidate_rename_revision(code):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "hello", expected_epoch=service.epoch
    )
    await harness.started.wait()
    before = await service.get_detail(session.id)
    await harness.connections[0].text("turn-1", "answer", "partial")
    after = await service.get_detail(session.id)
    assert after.version > before.version
    assert after.session.revision == before.session.revision
    await service.update_settings(
        session.id, before.session.revision, {"title": "My code"}
    )
    assert (await service.get_detail(session.id)).session.title == "My code"


@pytest.mark.asyncio
async def test_context_usage_is_an_absolute_deduplicated_session_snapshot(code):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "hello", expected_epoch=service.epoch
    )
    await harness.started.wait()
    connection = harness.connections[0]
    before = await service.get_detail(session.id)
    subscription, _ = await service.subscribe()
    events = subscription.__aiter__()
    try:
        await connection.context_usage("turn-1", 90_000)
        async with asyncio.timeout(2):
            event = await anext(events)
        after = await service.get_detail(session.id)
        assert event.event == "session.updated"
        assert event.data["session"]["context_used_tokens"] == 90_000
        assert after.session.context_used_tokens == 90_000
        assert after.session.revision == before.session.revision
        assert after.session.updated_at == before.session.updated_at
        assert (
            await service._store.get_session(session.id)
        ).context_used_tokens == 90_000

        await connection.context_usage("turn-1", 90_000)
        duplicate = await service.get_detail(session.id)
        assert duplicate.version == after.version

        await connection.context_usage("turn-1", 25_000)
        compacted = await service.get_detail(session.id)
        assert compacted.session.context_used_tokens == 25_000
        assert compacted.session.revision == after.session.revision
        assert compacted.session.updated_at == after.session.updated_at
        await connection.finish("turn-1")
    finally:
        await subscription.aclose()


@pytest.mark.asyncio
async def test_context_usage_is_retained_across_model_changes(code):
    service, harness, _ = code
    harness.configurations["provider/other"] = "other"
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "first", expected_epoch=service.epoch
    )
    await harness.started.wait()
    connection = harness.connections[0]
    await connection.context_usage("turn-1", 40_000)
    await connection.finish("turn-1")

    detail = await service.get_detail(session.id)
    changed = await service.update_settings(
        session.id, detail.session.revision, {"model": "provider/other"}
    )
    assert changed.context_used_tokens == 40_000
    restored = await service.update_settings(
        session.id, changed.revision, {"model": "provider/model"}
    )
    assert restored.context_used_tokens == 40_000
    assert (await service._store.get_session(session.id)).context_used_tokens == 40_000

    await service.send(
        session.id,
        new_id(),
        restored.revision,
        "second",
        expected_epoch=service.epoch,
    )
    await harness.wait_inputs(2)
    await connection.context_usage("turn-2", 7_500)
    current = await service.get_detail(session.id)
    assert current.session.context_used_tokens == 7_500
    await connection.finish("turn-2")


@pytest.mark.asyncio
async def test_resumed_context_usage_is_accepted_after_the_next_run_is_admitted(code):
    service, harness, _ = code
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "first", expected_epoch=service.epoch
    )
    await harness.started.wait()
    original = harness.connections[0]
    await original.context_usage("turn-1", 10_000)
    await original.finish("turn-1")

    session = (await service.get_detail(session.id)).session
    harness.configurations[harness.model] = "replacement"
    harness.start_gate.clear()
    harness.started.clear()
    harness.submitted.clear()
    await service.send(
        session.id, new_id(), session.revision, "second", expected_epoch=service.epoch
    )
    await harness.submitted.wait()
    resumed = harness.connections[1]
    assert original.closed
    assert resumed.resumed == [session.native_thread_id]

    await resumed.context_usage("turn-1", 11_000)
    await resumed.context_usage("turn-1", 12_000, generation=original.generation)
    await resumed.context_usage("turn-1", 13_000, thread_id="another-thread")
    replayed = await service.get_detail(session.id)
    assert replayed.session.context_used_tokens == 11_000
    assert (await service._store.get_session(session.id)).context_used_tokens == 11_000

    harness.start_gate.set()
    await harness.started.wait()
    await resumed.context_usage("turn-2", 20_000)
    current = await service.get_detail(session.id)
    assert current.session.context_used_tokens == 20_000
    await resumed.finish("turn-2")


@pytest.mark.asyncio
async def test_catalog_replacement_is_limited_to_selected_entry_and_session(code):
    service, harness, _ = code
    harness.configurations["provider/other"] = "other-1"
    session = await session_for(code)
    await service.send(
        session.id, new_id(), session.revision, "first", expected_epoch=service.epoch
    )
    await harness.started.wait()
    original = harness.connections[0]
    await original.finish("turn-1")
    harness.model = "provider/other"
    detail = await service.get_detail(session.id)
    selected = await service.update_settings(
        session.id, detail.session.revision, {"model": harness.model}
    )
    await service.send(
        session.id,
        new_id(),
        selected.revision,
        "second",
        expected_epoch=service.epoch,
    )
    await harness.wait_inputs(2)
    await original.finish("turn-2")
    assert len(harness.connections) == 1
    harness.configurations["unrelated/model"] = "new"
    harness.configurations["provider/other"] = "changed"
    detail = await service.get_detail(session.id)
    await service.send(
        session.id,
        new_id(),
        detail.session.revision,
        "third",
        expected_epoch=service.epoch,
    )
    await harness.wait_inputs(3)
    assert len(harness.connections) == 2
    assert original.closed
    assert harness.connections[1].resumed == ["native-1"]


@pytest.mark.asyncio
async def test_cancelled_http_admission_still_commits_and_executes_once(tmp_path):
    committed = asyncio.Event()
    release = asyncio.Event()

    class GatedStore(SQLiteCodeStore):
        async def admit_run(self, session, run, item, expected_revision):
            result = await super().admit_run(session, run, item, expected_revision)
            committed.set()
            await release.wait()
            return result

    harness = FakeHarness()
    service = CodeService(
        GatedStore(tmp_path / "code.db", tmp_path / "code.lock"), harness
    )
    await service.start()
    try:
        session = await service.create_session(new_id(), str(tmp_path))
        operation_id = new_id()
        http = asyncio.create_task(
            service.send(
                session.id,
                operation_id,
                session.revision,
                "once",
                expected_epoch=service.epoch,
            )
        )
        await committed.wait()
        http.cancel()
        with pytest.raises(asyncio.CancelledError):
            await http
        release.set()
        await harness.started.wait()
        repeat = await service.send(
            session.id,
            operation_id,
            session.revision,
            "once",
            expected_epoch=service.epoch,
        )
        assert repeat.id == operation_id
        assert len(harness.connections[0].inputs) == 1
    finally:
        release.set()
        await service.close()
