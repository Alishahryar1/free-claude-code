"""Messaging-specific assertions built on neutral Anthropic stream contracts."""

from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    has_tool_use,
    parse_sse_text,
)
from free_claude_code.core.chat_observations import ChatStreamUsage
from free_claude_code.messaging.event_parser import parse_cli_event
from free_claude_code.messaging.transcript import RenderCtx, TranscriptBuffer
from tests.protocol_stream_support import ChatSourceHarness
from tests.stream_helpers import serialize_events


def test_thinking_tool_text_and_transcript_order_contract() -> None:
    builder = ChatSourceHarness(model="contract-model", input_tokens=0)
    changes = [
        *builder.start_events(),
        *builder.ensure_reasoning_block(),
        builder.emit_reasoning_delta("inspect first"),
        *builder.close_content_blocks(),
        builder.start_tool_block(0, "toolu_1", "Read"),
        *builder.emit_tool_delta(0, '{"file":"README.md"}'),
        *builder.stop_tool_block(0),
        *builder.ensure_text_block(),
        builder.emit_text_delta("done"),
        *builder.finish_success(stop_reason="end_turn", usage=ChatStreamUsage(0, 20)),
    ]
    chunks = builder.project(changes)

    events = parse_sse_text(serialize_events(chunks))
    assert_anthropic_stream_contract(events)
    assert has_tool_use(events)

    transcript = TranscriptBuffer()
    for event in events:
        for parsed in parse_cli_event(event.data):
            transcript.apply(parsed)
    rendered = transcript.render(_render_ctx(), limit_chars=3900, status=None)
    assert (
        rendered.find("inspect first")
        < rendered.find("Tool call:")
        < rendered.find("done")
    )


def _render_ctx() -> RenderCtx:
    return RenderCtx(
        bold=lambda s: f"*{s}*",
        code_inline=lambda s: f"`{s}`",
        escape_code=lambda s: s,
        escape_text=lambda s: s,
        render_markdown=lambda s: s,
    )
