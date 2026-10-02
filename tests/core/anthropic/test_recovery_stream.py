"""One public lifecycle, completed-call publication, and resumable checkpoints."""

from dataclasses import replace
from weakref import WeakKeyDictionary

from free_claude_code.core.anthropic.native_stream import NativeMessagesStreamState
from free_claude_code.core.anthropic.recovery_stream import MessagesRecoveryWriter
from free_claude_code.core.history_replay import ReplayOrigin
from free_claude_code.core.stream_events import DecodedStreamEvent, StreamEvent

ORIGIN = ReplayOrigin("first", "messages", "https://first.test", "key", "model")
SOURCES: WeakKeyDictionary[MessagesRecoveryWriter, NativeMessagesStreamState] = (
    WeakKeyDictionary()
)


def _feed(
    writer: MessagesRecoveryWriter, kind: str, **body: object
) -> list[StreamEvent]:
    event = StreamEvent(kind, {"type": kind, **body})
    if kind == "message_start":
        SOURCES[writer] = NativeMessagesStreamState(permissive=True)
    source = SOURCES[writer]
    completed = source.accept(kind, event.payload)
    event = replace(event, item_completion=completed.completion if completed else None)
    return writer.feed(
        DecodedStreamEvent(
            ORIGIN,
            event,
            (event,),
            native_reasoning_pending=source.native_reasoning_pending,
        )
    )


def _start(
    writer: MessagesRecoveryWriter, identity: str = "upstream"
) -> list[StreamEvent]:
    writer.begin_attempt()
    return _feed(
        writer,
        "message_start",
        message={"id": identity, "model": "provider", "content": []},
    )


def _text(
    writer: MessagesRecoveryWriter, value: str, index: int = 0
) -> list[StreamEvent]:
    return [
        *_feed(
            writer,
            "content_block_start",
            index=index,
            content_block={"type": "text", "text": ""},
        ),
        *_feed(
            writer,
            "content_block_delta",
            index=index,
            delta={"type": "text_delta", "text": value},
        ),
    ]


def _tool(
    writer: MessagesRecoveryWriter, index: int, identity: str
) -> list[StreamEvent]:
    return _feed(
        writer,
        "content_block_start",
        index=index,
        content_block={"type": "tool_use", "id": identity, "name": "read", "input": {}},
    )


def test_unknown_message_state_is_preserved_and_blocks_continuation():
    writer = MessagesRecoveryWriter(model="public", input_tokens=12, native=True)
    writer.begin_attempt()
    frames = _feed(
        writer,
        "message_start",
        message={"id": "original", "content": [], "future_state": {"handle": "opaque"}},
    )
    assert frames[0].payload["message"]["future_state"] == {"handle": "opaque"}
    assert writer.checkpoint.blocked_reason is not None


def test_repeated_failures_keep_one_identity_and_the_latest_committed_text() -> None:
    writer = MessagesRecoveryWriter(model="public", input_tokens=12)
    emitted = _start(writer, "original")
    for value in ("First ", "second ", "third"):
        emitted += _text(writer, value)
        emitted += writer.interrupt()
        if value != "third":
            emitted += _start(writer, "replacement")
    _feed(
        writer,
        "message_delta",
        delta={"stop_reason": "end_turn"},
        usage={"input_tokens": 2000, "output_tokens": 6},
    )
    _feed(writer, "message_stop")
    emitted += writer.finish()

    starts = [event for event in emitted if event.kind == "message_start"]
    assert len(starts) == 1
    assert starts[0].payload["message"]["id"] == "original"
    assert starts[0].payload["message"]["model"] == "public"
    assert writer.checkpoint.text == "First second third"
    assert [
        event.payload["index"]
        for event in emitted
        if event.kind == "content_block_start"
    ] == [0, 1, 2]
    assert sum(event.kind == "message_stop" for event in emitted) == 1
    assert emitted[-2].payload["usage"]["input_tokens"] == 12


def test_a_large_pending_tool_is_discarded_without_exposing_its_identity() -> None:
    writer = MessagesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    _text(writer, "Before")
    _feed(writer, "content_block_stop", index=0)
    assert not _tool(writer, 1, "unpublished")
    assert not _feed(
        writer,
        "content_block_delta",
        index=1,
        delta={"type": "input_json_delta", "partial_json": '{"value":"' + "x" * 70000},
    )
    assert not writer.interrupt()
    assert not writer.checkpoint.published_tools
    assert writer.checkpoint.text == "Before"
    _start(writer)
    emitted = _text(writer, "After")
    assert all("unpublished" not in event.serialize() for event in emitted)
    assert writer.checkpoint.text == "BeforeAfter"


def test_each_complete_call_is_released_without_waiting_for_the_turn() -> None:
    writer = MessagesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    assert not _tool(writer, 0, "first")
    assert not _feed(
        writer,
        "content_block_delta",
        index=0,
        delta={"type": "input_json_delta", "partial_json": "{}"},
    )
    complete = _feed(writer, "content_block_stop", index=0)
    assert [event.kind for event in complete] == [
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    ]
    assert writer.checkpoint.published_tools
    assert not _tool(writer, 1, "unfinished")
    terminal = writer.finish(salvage=True)
    assert terminal[-2].payload["delta"]["stop_reason"] == "tool_use"
    assert "unfinished" not in "".join(event.serialize() for event in terminal)


def test_interleaved_calls_release_in_start_order() -> None:
    writer = MessagesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    _tool(writer, 0, "first")
    _tool(writer, 1, "second")
    assert not _feed(writer, "content_block_stop", index=1)
    released = _feed(writer, "content_block_stop", index=0)
    assert [
        event.payload["content_block"]["id"]
        for event in released
        if event.kind == "content_block_start"
    ] == ["first", "second"]
    assert [event.payload["index"] for event in released] == [0, 0, 1, 1]


def test_incomplete_native_thinking_prevents_resume_or_tool_salvage() -> None:
    writer = MessagesRecoveryWriter(model="public", input_tokens=12, native=True)
    _start(writer)
    _feed(
        writer,
        "content_block_start",
        index=0,
        content_block={"type": "thinking", "thinking": "private"},
    )
    assert writer.checkpoint.blocked_reason is not None
    assert not writer.interrupt()


def test_normal_token_exhaustion_keeps_the_real_stop_reason() -> None:
    writer = MessagesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    _tool(writer, 0, "complete")
    _feed(writer, "content_block_stop", index=0)
    _feed(
        writer,
        "message_delta",
        delta={"stop_reason": "max_tokens"},
        usage={"input_tokens": 17, "output_tokens": 32000},
    )
    _feed(writer, "message_stop")
    terminal = writer.finish()
    assert terminal[-2].payload["delta"]["stop_reason"] == "max_tokens"
    assert terminal[-2].payload["usage"] == {"input_tokens": 17, "output_tokens": 32000}


def test_complete_prefix_replay_does_not_add_text_to_checkpoint() -> None:
    writer = MessagesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    _text(writer, "The beginning. ")
    writer.interrupt()
    revision = writer.revision
    _start(writer)
    _text(writer, "The begin")
    _feed(
        writer,
        "content_block_delta",
        index=0,
        delta={"type": "text_delta", "text": "ning. "},
    )
    writer.interrupt()
    assert writer.revision == revision
    assert writer.checkpoint.text == "The beginning. "


def test_unknown_extension_on_a_known_block_prevents_recovery():
    writer = MessagesRecoveryWriter(model="public", input_tokens=12, native=True)
    _start(writer)
    events = _feed(
        writer,
        "content_block_start",
        index=0,
        content_block={
            "type": "text",
            "text": "Hello",
            "future_reference": {"id": "remote"},
        },
    )
    assert events[0].payload["content_block"]["future_reference"] == {"id": "remote"}
    assert writer.checkpoint.blocked_reason is not None


def test_recovered_usage_excludes_opaque_signatures_and_foreign_cache_counts():
    usages = []
    for signature in ("short", "opaque" * 10000):
        writer = MessagesRecoveryWriter(model="public", input_tokens=12)
        _start(writer)
        _feed(
            writer,
            "content_block_start",
            index=0,
            content_block={
                "type": "thinking",
                "thinking": "Plan.",
                "signature": signature,
            },
        )
        _feed(writer, "content_block_stop", index=0)
        writer.interrupt()
        _start(writer, "replacement")
        _text(writer, "Answer.")
        _feed(
            writer,
            "message_delta",
            delta={"stop_reason": "end_turn"},
            usage={
                "input_tokens": 900,
                "output_tokens": 20,
                "cache_read_input_tokens": 800,
            },
        )
        _feed(writer, "message_stop")
        usages.append(writer.finish()[-2].payload["usage"])
    assert usages[0] == usages[1]
    assert usages[0]["input_tokens"] == 12
    assert "cache_read_input_tokens" not in usages[0]
