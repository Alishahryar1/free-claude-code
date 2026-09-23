"""Messaging-specific assertions built on neutral Anthropic stream contracts."""

from free_claude_code.core.anthropic import AnthropicStreamLedger
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    has_tool_use,
    parse_sse_text,
)
from free_claude_code.messaging.event_parser import parse_cli_event
from free_claude_code.messaging.transcript import RenderCtx, TranscriptBuffer


def test_thinking_tool_text_and_transcript_order_contract() -> None:
    builder = AnthropicStreamLedger("msg_contract", "contract-model")
    chunks = [builder.message_start()]
    chunks.extend(builder.ensure_thinking_block())
    chunks.append(builder.emit_thinking_delta("inspect first"))
    chunks.extend(builder.close_content_blocks())
    tool_block_index = builder.blocks.allocate_index()
    chunks.append(
        builder.content_block_start(
            tool_block_index, "tool_use", id="toolu_1", name="Read"
        )
    )
    chunks.append(
        builder.content_block_delta(
            tool_block_index, "input_json_delta", '{"file":"README.md"}'
        )
    )
    chunks.append(builder.content_block_stop(tool_block_index))
    chunks.extend(builder.ensure_text_block())
    chunks.append(builder.emit_text_delta("done"))
    chunks.extend(builder.close_all_blocks())
    chunks.append(builder.message_delta("end_turn", 20))
    chunks.append(builder.message_stop())

    events = parse_sse_text("".join(chunks))
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
