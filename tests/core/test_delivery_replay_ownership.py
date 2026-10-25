"""Completed source state belongs only to publicly visible attempts."""

from dataclasses import replace

from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.stream_delivery import StreamDeliveryState
from free_claude_code.core.stream_recovery import DeliveredPrefix


def snapshot(state):
    value = state.prefix
    assert value is not None
    return value


def record(text="same thought"):
    return ReplayRecord(
        ReplayOrigin("provider", "chat", "endpoint", "account", "model"),
        {"reasoning_content": text},
    )


def test_visible_records_survive_but_invisible_attempt_records_do_not():
    state = StreamDeliveryState()
    state.bind_prefix(lambda: DeliveredPrefix("answer", "", False, True, False, None))
    first = record()
    state.begin_attempt()
    state.bind_replay_source(lambda: (first,))
    assert snapshot(state).replay_records == ()
    state.release_content()
    assert snapshot(state).replay_records == (first,)
    state.begin_attempt()
    state.bind_replay_source(lambda: (record("discard me"),))
    assert snapshot(state).replay_records == (first,)
    state.begin_attempt()
    state.bind_replay_source(lambda: (record(),))
    state.release_content()
    assert snapshot(state).replay_records == (first, first)
    state.bind_prefix(None)
    assert state.committed_replay == () and state._replay_source is None


def test_native_replay_and_public_text_share_the_retention_budget(monkeypatch):
    saved = record()
    limit = len(encode_replay(saved)) + 1
    monkeypatch.setattr(
        "free_claude_code.core.stream_delivery.RECOVERY_RETENTION_BYTES", limit
    )
    state = StreamDeliveryState()
    prefix = DeliveredPrefix("answer", "", False, True, False, None, retained_bytes=2)
    state.bind_prefix(lambda: prefix)
    state.begin_attempt()
    state.bind_replay_source(lambda: (saved,))
    state.release_content()
    assert not snapshot(state).eligible
    assert snapshot(state).unsafe_reason == "retention_limit"
    assert snapshot(state).replay_records == ()
    prefix = replace(prefix, retained_bytes=0)
    assert snapshot(state).eligible
    state.begin_attempt()
    assert state.committed_replay == (saved,)
