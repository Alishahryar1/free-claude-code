import pytest

from free_claude_code.application.code_sessions.models import (
    CodeConflictError,
    CodeItem,
    CodeRun,
    CodeSession,
    HarnessEvent,
    ItemUpdate,
    NativeThread,
    NativeTurn,
    PromptRequest,
)
from free_claude_code.application.code_sessions.state import (
    SessionProgress,
    SessionState,
)


def running_state():
    session = CodeSession(
        id="session", cwd="project", model="provider/model", native_thread_id="native"
    )
    run = CodeRun(
        id="run",
        session_id=session.id,
        ordinal=1,
        text="hello",
        model=session.model,
        status="running",
        submission_started=True,
        native_turn_id="turn",
    )
    return SessionState(session, run, (), (), (run,))


def test_admission_installs_only_the_acknowledged_records():
    session = CodeSession(id="session", cwd="project", model="provider/model")
    state = SessionState(session, None, (), (), ())
    candidate, run, item = state.prepare_admission(
        "send", "hello\nworld", session.model, None, "config"
    )
    assert (
        candidate.title == "hello world" and candidate.revision == session.revision + 1
    )
    assert state.session == session and state.run is None
    assert not state.items and not state.runs

    admitted = run.model_copy(update={"ordinal": 7})
    state.accept_admission(candidate, admitted, item)
    assert state.run == admitted and state.runs[admitted.id] == admitted
    assert state.items[item.id] == item and item.sequence == 1
    assert state.busy and not state.pending_items
    with pytest.raises(CodeConflictError, match="busy"):
        state.prepare_admission("another", "next", session.model, None, "config")


def test_terminal_bundle_preserves_null_turn_prompt_and_requires_buffer_acknowledgment():
    state = running_state()
    for request_id, turn_id in ((1, "turn"), ("1", None)):
        progress = state.prepare_prompt(
            PromptRequest(request_id, "approval", {}, {}, turn_id), "generation"
        )
        state.apply_progress(progress)
    before = state.session, state.run, tuple(state.prompts.values())
    item, changed = state.stage_item(
        ItemUpdate("turn", "answer", "text", text="partial"), state.run
    )
    assert changed

    finish = state.prepare_finish("completed")
    assert finish.items == (item,)
    assert len(finish.prompts) == 1 and finish.prompts[0].request_id == 1
    assert finish.prompts[0].status == "expired"
    assert (state.session, state.run, tuple(state.prompts.values())) == before
    assert state.pending_items == (item,)

    state.apply_progress(finish)
    assert state.run.status == "completed" and not state.busy
    assert state.pending and state.pending_items == (item,)
    assert (
        next(
            prompt for prompt in state.prompts.values() if prompt.request_id == "1"
        ).status
        == "pending"
    )
    state.acknowledge_flush()
    assert not state.pending_items and state.dirty_characters == 0
    with pytest.raises(CodeConflictError, match="busy"):
        state.check_can_send()


def test_recovery_updates_historical_run_without_replacing_current_run():
    session = CodeSession(id="session", cwd="project", model="provider/model")
    old = CodeRun(
        id="old",
        session_id=session.id,
        ordinal=1,
        text="old input",
        model=session.model,
        status="failed",
        submission_started=True,
        error="Connection lost",
        finished_at=100,
    )
    current = CodeRun(
        id="new",
        session_id=session.id,
        ordinal=2,
        text="new input",
        model=session.model,
    )
    state = SessionState(session, current, (), (), (old, current))
    native = NativeThread(
        "native",
        (
            NativeTurn(
                "old-turn",
                (ItemUpdate("old-turn", "user", "user", client_id="old"),),
                error="Native failed",
                error_details={"code": "native-error"},
            ),
        ),
    )
    ((turn, recovered),) = state.match_history(native)
    assert turn.id == recovered.native_turn_id == "old-turn"
    assert recovered.status == "failed" and recovered.finished_at == 100
    assert recovered.error == "Connection lost"
    assert recovered.error_details == {"code": "native-error"}
    assert state.runs[old.id] == old
    state.apply_progress(SessionProgress(session, run=recovered))
    assert state.run == current and state.runs[old.id] == recovered


def test_historical_merge_retains_transcript_identity_and_completed_content():
    base = running_state()
    saved = CodeItem(
        id="entry",
        session_id=base.session.id,
        run_id=base.run.id,
        sequence=12,
        native_turn_id="turn",
        native_item_id="answer",
        kind="text",
        text="complete output",
        detail="captured details",
        complete=True,
        raw={"nested": {"captured": "keep"}},
    )
    state = SessionState(base.session, base.run, (saved,), (), (base.run,))
    update = ItemUpdate(
        "turn",
        "answer",
        "text",
        text="short",
        raw={"nested": {"captured": "short", "new": "fill"}},
    )
    assert state.run is not None
    merged, changed = state.stage_item(update, state.run, historical=True)
    assert changed and merged.id == saved.id and merged.sequence == 12
    assert (
        merged.complete and merged.text == saved.text and merged.detail == saved.detail
    )
    assert merged.raw == {"nested": {"captured": "keep", "new": "fill"}}
    assert saved.raw == {"nested": {"captured": "keep"}}
    assert state.dirty_characters == 0 and state.pending_items == (merged,)


def test_review_liveness_can_change_without_changing_its_item_or_run():
    state = running_state()
    update = ItemUpdate(
        "child-turn", "review", "subagent_auto_review", text="Reviewing"
    )
    item, changed = state.stage_review(update, "first")
    assert changed and item.run_id == state.run.id
    state.apply_progress(SessionProgress(state.session, items=state.pending_items))
    state.acknowledge_flush()
    assert state.clear_review_liveness()

    state.apply_progress(state.prepare_finish("completed"))
    candidate, run, message = state.prepare_admission(
        "second", "next", state.session.model, None, "config"
    )
    state.accept_admission(candidate, run.model_copy(update={"ordinal": 2}), message)
    repeated, changed = state.stage_review(update, "second")
    assert repeated == item and not changed
    assert state.active_review_ids("second") == (item.id,)
    assert state.active_review_ids("first") == ()
    assert state.pending_items == (item,)


def test_only_new_turn_started_can_bind_an_unacknowledged_submission():
    base = running_state()
    current = CodeRun(
        id="new",
        session_id=base.session.id,
        ordinal=2,
        text="next",
        model=base.session.model,
        submission_started=True,
    )
    state = SessionState(base.session, current, (), (), (base.run, current))
    assert not state.matches_turn(
        HarnessEvent("generation", "native", "turn_started", turn_id="turn")
    )
    assert not state.matches_turn(
        HarnessEvent("generation", "native", "item", turn_id="new-turn")
    )
    assert state.matches_turn(
        HarnessEvent("generation", "native", "turn_started", turn_id="new-turn")
    )
    state.apply_progress(state.bind_turn("new-turn"))
    assert state.matches_turn(
        HarnessEvent("generation", "native", "item", turn_id="new-turn")
    )
    assert not state.accepts_event(HarnessEvent("generation", "other-thread", "notice"))
