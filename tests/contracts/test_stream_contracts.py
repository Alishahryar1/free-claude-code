"""Stream/SSE contract tests. Strict transcript *ordering* is covered here for
the public Messages writer; for integration ordering, add messaging or API
integration tests.
"""

from collections.abc import Iterable

from free_claude_code.core.anthropic import (
    ContentType,
    ThinkTagParser,
)
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    event_names,
    parse_sse_text,
    text_content,
    thinking_content,
)
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.chat_observations import ChatChange, ChatStreamUsage
from free_claude_code.core.stream_events import StreamEvent
from tests.protocol_stream_support import ChatSourceHarness
from tests.stream_helpers import serialize_events


def test_interleaved_thinking_text_blocks_are_valid() -> None:
    events = _parse_builder_events(
        _interleaved_thinking_text_events(
            ("first thought", "first answer", "second thought", "final answer")
        )
    )
    assert_anthropic_stream_contract(events)
    assert event_names(events).count("content_block_start") == 4
    assert thinking_content(events) == "first thoughtsecond thought"
    assert text_content(events) == "first answerfinal answer"


def test_split_think_tags_preserve_text_and_thinking() -> None:
    events = _parse_builder_events(
        _events_from_text_chunks(["before <thi", "nk>hidden", "</think> after"])
    )
    assert_anthropic_stream_contract(events)
    assert thinking_content(events) == "hidden"
    assert text_content(events) == "before  after"


def test_mixed_reasoning_content_and_think_tags_keep_order() -> None:
    builder = ChatSourceHarness(model="contract-model", input_tokens=0)
    chunks = builder.start_events()
    chunks.extend(builder.ensure_reasoning_block())
    chunks.append(builder.emit_reasoning_delta("reasoning field"))
    chunks.extend(
        _changes_from_text_chunks([" visible <think>tagged</think> done"], builder)
    )
    chunks.extend(builder.close_all_blocks())
    chunks.extend(
        builder.finish_success(stop_reason="end_turn", usage=ChatStreamUsage(0, 10))
    )

    events = parse_sse_text(serialize_events(builder.project(chunks)))
    assert_anthropic_stream_contract(events)
    assert thinking_content(events) == "reasoning fieldtagged"
    assert text_content(events) == " visible  done"


def test_redacted_thinking_block_start_stop_is_valid() -> None:
    """Native redacted_thinking uses start/stop only (no deltas)."""
    chunks = [
        format_sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_r",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "m",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            },
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "redacted_thinking", "data": "opaque"},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]
    events = parse_sse_text(serialize_events(chunks))
    assert_anthropic_stream_contract(events)


def test_enable_thinking_false_suppresses_reasoning_only() -> None:
    events = _parse_builder_events(
        _events_from_text_chunks(
            ["hello <think>secret</think> world"], enable_thinking=False
        )
    )
    assert_anthropic_stream_contract(events)
    assert "secret" not in thinking_content(events)
    assert text_content(events) == "hello  world"


def _interleaved_thinking_text_events(
    parts: tuple[str, str, str, str],
) -> Iterable[str]:
    builder = ChatSourceHarness(model="contract-model", input_tokens=0)
    changes = [
        *builder.start_events(),
        *builder.ensure_reasoning_block(),
        builder.emit_reasoning_delta(parts[0]),
        *builder.ensure_text_block(),
        builder.emit_text_delta(parts[1]),
        *builder.ensure_reasoning_block(),
        builder.emit_reasoning_delta(parts[2]),
        *builder.ensure_text_block(),
        builder.emit_text_delta(parts[3]),
        *builder.finish_success(stop_reason="end_turn", usage=ChatStreamUsage(0, 20)),
    ]
    yield from builder.project(changes)


def _events_from_text_chunks(
    chunks: list[str], *, enable_thinking: bool = True
) -> list[StreamEvent]:
    source = ChatSourceHarness(model="contract-model", input_tokens=0)
    changes = [
        *source.start_events(),
        *_changes_from_text_chunks(chunks, source, enable_thinking=enable_thinking),
        *source.finish_success(stop_reason="end_turn", usage=ChatStreamUsage(0, 20)),
    ]
    return source.project(changes)


def _changes_from_text_chunks(
    chunks: list[str], source: ChatSourceHarness, *, enable_thinking: bool = True
) -> list[ChatChange]:
    parser = ThinkTagParser()
    out: list[ChatChange] = []
    for chunk in chunks:
        out.extend(_emit_parser_parts(source, parser.feed(chunk), enable_thinking))
    remaining = parser.flush()
    if remaining is not None:
        out.extend(_emit_parser_parts(source, [remaining], enable_thinking))
    return out


def _emit_parser_parts(
    builder: ChatSourceHarness,
    parts: Iterable,
    enable_thinking: bool,
) -> list[ChatChange]:
    out: list[ChatChange] = []
    for part in parts:
        if part.type == ContentType.THINKING:
            if enable_thinking:
                out.extend(builder.ensure_reasoning_block())
                out.append(builder.emit_reasoning_delta(part.content))
            continue
        out.extend(builder.ensure_text_block())
        out.append(builder.emit_text_delta(part.content))
    return out


def _parse_builder_events(chunks: Iterable[StreamEvent | str]):
    return parse_sse_text(serialize_events(chunks))
