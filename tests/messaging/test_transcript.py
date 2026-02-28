from unittest.mock import patch

from messaging.rendering.telegram_markdown import (
    escape_md_v2,
    escape_md_v2_code,
    mdv2_bold,
    mdv2_code_inline,
    render_markdown_to_mdv2,
)
from messaging.transcript import RenderCtx, TranscriptBuffer


def _ctx() -> RenderCtx:
    return RenderCtx(
        bold=mdv2_bold,
        code_inline=mdv2_code_inline,
        escape_code=escape_md_v2_code,
        escape_text=escape_md_v2,
        render_markdown=render_markdown_to_mdv2,
        thinking_tail_max=1000,
        tool_input_tail_max=1200,
        tool_output_tail_max=1600,
        text_tail_max=2000,
    )


def test_transcript_order_thinking_tool_text():
    t = TranscriptBuffer()
    t.apply({"type": "thinking_chunk", "text": "think1"})
    t.apply({"type": "tool_use", "id": "tool_1", "name": "ls", "input": {"path": "."}})
    t.apply({"type": "text_chunk", "text": "done"})

    out = t.render(_ctx(), limit_chars=3900, status=None)
    assert out.find("think1") < out.find("Tool call:") < out.find("done")


def test_transcript_subagent_suppresses_thinking_and_text_inside():
    t = TranscriptBuffer()

    # Enter subagent context (Task tool call).
    t.apply(
        {
            "type": "tool_use",
            "id": "task_1",
            "name": "Task",
            "input": {"description": "Fix bug"},
        }
    )

    # These should be suppressed while inside subagent context.
    t.apply({"type": "thinking_delta", "index": -1, "text": "secret"})
    t.apply({"type": "text_chunk", "text": "visible?"})

    # Tool activity should still show.
    t.apply({"type": "tool_use", "id": "tool_2", "name": "ls", "input": {"path": "."}})
    t.apply({"type": "tool_result", "tool_use_id": "tool_2", "content": "x"})

    # Close subagent context (Task tool result).
    t.apply({"type": "tool_result", "tool_use_id": "task_1", "content": "done"})

    # Now text should show again.
    t.apply({"type": "text_chunk", "text": "after"})

    out = t.render(_ctx(), limit_chars=3900, status=None)
    assert "Subagent:" in out
    assert "secret" not in out
    assert "visible?" not in out
    # Only the current tool call should be shown (not the full history).
    assert out.count("Tool call:") == 1
    assert "\n  🛠" in out or out.startswith("  🛠") or "  🛠" in out
    assert "Tools used:" in out
    assert "Tool calls:" in out
    assert "after" in out


def test_transcript_subagent_closes_on_whitespace_tool_ids():
    t = TranscriptBuffer()

    # Provider emitted a Task tool_use id with leading whitespace.
    t.apply(
        {
            "type": "tool_use",
            "id": " functions.Task:0",
            "name": "Task",
            "input": {"description": "Outer"},
        }
    )

    # Task completes, but tool_result references a trimmed id (or vice versa).
    t.apply(
        {"type": "tool_result", "tool_use_id": "functions.Task:0", "content": "done"}
    )

    # Next Task should be top-level, not nested under the previous subagent.
    t.apply(
        {
            "type": "tool_use",
            "id": "functions.Task:1",
            "name": "Task",
            "input": {"description": "Next"},
        }
    )

    out = t.render(_ctx(), limit_chars=3900, status=None)
    assert out.count("Subagent:") == 2
    # If nesting is incorrect, the second subagent line will be indented under the first.
    assert "\n  🤖 *Subagent:* `Next`" not in out


def test_transcript_subagent_closes_on_task_result_id_suffix_match():
    t = TranscriptBuffer()
    t.apply(
        {
            "type": "tool_use",
            "id": "task_1",
            "name": "Task",
            "input": {"description": "Outer"},
        }
    )
    t.apply({"type": "tool_result", "tool_use_id": "task_1_result", "content": "done"})
    t.apply(
        {
            "type": "tool_use",
            "id": "task_2",
            "name": "Task",
            "input": {"description": "Next"},
        }
    )

    out = t.render(_ctx(), limit_chars=3900, status=None)
    assert out.count("Subagent:") == 2
    assert "\n  🤖 *Subagent:* `Next`" not in out


def test_transcript_unmatched_non_task_tool_result_does_not_pop_subagent():
    t = TranscriptBuffer()
    t.apply(
        {
            "type": "tool_use",
            "id": "task_1",
            "name": "Task",
            "input": {"description": "Outer"},
        }
    )
    t.apply({"type": "tool_result", "tool_use_id": "totally_unrelated", "content": "x"})

    assert t._subagent_stack == ["task_1"]


def test_transcript_sequential_tasks_mismatched_results_no_depth_drift():
    t = TranscriptBuffer()
    t.apply(
        {
            "type": "tool_use",
            "id": "task_1",
            "name": "Task",
            "input": {"description": "A"},
        }
    )
    t.apply({"type": "tool_result", "tool_use_id": "task_1_result", "content": "done"})
    t.apply(
        {
            "type": "tool_use",
            "id": "task_2",
            "name": "Task",
            "input": {"description": "B"},
        }
    )
    t.apply({"type": "tool_result", "tool_use_id": "task_2_result", "content": "done"})
    t.apply(
        {
            "type": "tool_use",
            "id": "task_3",
            "name": "Task",
            "input": {"description": "C"},
        }
    )

    out = t.render(_ctx(), limit_chars=3900, status=None)
    assert "🤖 *Subagent:* `A`\n  🤖 *Subagent:* `B`" not in out
    assert "\n  🤖 *Subagent:* `C`" not in out
    assert t._subagent_stack == ["task_3"]


def test_transcript_synthetic_task_start_closes_on_functions_task_result_id():
    t = TranscriptBuffer()
    t.apply(
        {
            "type": "tool_use_start",
            "index": 0,
            "id": "",
            "name": "Task",
            "input": {"description": "Outer"},
        }
    )
    t.apply({"type": "tool_result", "tool_use_id": "functions.Task:0", "content": "x"})
    t.apply(
        {
            "type": "tool_use_start",
            "index": 1,
            "id": "",
            "name": "Task",
            "input": {"description": "Next"},
        }
    )

    out = t.render(_ctx(), limit_chars=3900, status=None)
    assert out.count("Subagent:") == 2
    assert "\n  🤖 *Subagent:* `Next`" not in out


def test_transcript_synthetic_task_not_closed_by_unknown_non_task_result_id():
    t = TranscriptBuffer()
    t.apply(
        {
            "type": "tool_use_start",
            "index": 0,
            "id": "",
            "name": "Task",
            "input": {"description": "Outer"},
        }
    )
    t.apply({"type": "tool_result", "tool_use_id": "call_deadbeef", "content": "x"})

    assert t._subagent_stack == ["__task_1"]


def test_transcript_overlapping_tasks_are_flat_not_nested():
    t = TranscriptBuffer()
    t.apply(
        {
            "type": "tool_use",
            "id": "task_a",
            "name": "Task",
            "input": {"description": "A"},
        }
    )
    t.apply(
        {
            "type": "tool_use",
            "id": "task_b",
            "name": "Task",
            "input": {"description": "B"},
        }
    )
    t.apply({"type": "tool_result", "tool_use_id": "task_b", "content": "done"})
    t.apply({"type": "tool_result", "tool_use_id": "task_a", "content": "done"})

    out = t.render(_ctx(), limit_chars=3900, status=None)
    assert "🤖 *Subagent:* `A`" in out
    assert "🤖 *Subagent:* `B`" in out
    assert out.find("🤖 *Subagent:* `A`") < out.find("🤖 *Subagent:* `B`")
    assert "\n  🤖 *Subagent:* `B`" not in out


def test_transcript_truncates_by_dropping_oldest_segments():
    t = TranscriptBuffer()

    # Create many segments by opening/closing distinct text blocks.
    for i in range(60):
        t.apply({"type": "text_start", "index": i})
        t.apply(
            {"type": "text_delta", "index": i, "text": f"segment_{i} " + ("x" * 120)}
        )
        t.apply({"type": "block_stop", "index": i})

    out = t.render(_ctx(), limit_chars=600, status="status")
    assert escape_md_v2("... (truncated)") in out
    # We keep the tail and drop the oldest segments when truncating.
    assert escape_md_v2("segment_59") in out
    assert escape_md_v2("segment_0") not in out


def test_transcript_render_many_segments_completes_quickly():
    """Render with 200+ segments exercises O(n) truncation (deque popleft)."""
    t = TranscriptBuffer()
    for i in range(200):
        t.apply({"type": "text_start", "index": i})
        t.apply({"type": "text_delta", "index": i, "text": f"seg_{i} " + ("y" * 80)})
        t.apply({"type": "block_stop", "index": i})

    out = t.render(_ctx(), limit_chars=500, status="ok")
    assert escape_md_v2("... (truncated)") in out
    assert "199" in out  # last segment (MarkdownV2 escapes underscores)
    assert "seg_0 " not in out  # oldest segment dropped


def test_transcript_reused_index_closes_previous_open_block():
    t = TranscriptBuffer()
    # Open a text block at index 0, but never close it.
    t.apply({"type": "text_start", "index": 0})
    t.apply({"type": "text_delta", "index": 0, "text": "a"})
    # Provider reuses index 0 for a new tool block without a stop.
    t.apply(
        {"type": "tool_use_start", "index": 0, "id": "t1", "name": "ls", "input": {}}
    )
    # Old open text should have been closed.
    assert 0 not in t._open_text_by_index
    assert 0 in t._open_tools_by_index


def test_transcript_render_segment_exception_skipped():
    """When a segment's render() raises, that segment is skipped and rest is rendered."""
    t = TranscriptBuffer()
    t.apply({"type": "thinking_chunk", "text": "before"})
    t.apply({"type": "text_chunk", "text": "middle"})
    t.apply({"type": "text_chunk", "text": "after"})

    bad_segment = t._segments[1]

    def _raising_render(self, ctx):
        raise ValueError("render failed")

    with patch.object(bad_segment, "render", _raising_render):
        out = t.render(_ctx(), limit_chars=3900, status=None)
    assert "before" in out
    assert "after" in out
    assert "middle" not in out


def test_transcript_render_status_only_exceeds_limit():
    """When all segments dropped, status-only output; long status returned as-is."""
    t = TranscriptBuffer()
    t.apply({"type": "text_chunk", "text": "x" * 5000})

    long_status = "A" * 500
    msg = t.render(_ctx(), limit_chars=100, status=long_status)
    assert "... (truncated)" in msg or long_status in msg


def test_transcript_truncation_preserves_last_segment_tail():
    """When all segments exceed limit, preserve tail of last segment (not just marker+status)."""
    t = TranscriptBuffer()
    t.apply({"type": "thinking_chunk", "text": "Thinking..."})
    t.apply(
        {"type": "text_chunk", "text": "The actual output content here" + "x" * 500}
    )

    msg = t.render(_ctx(), limit_chars=100, status="✅ *Complete*")
    # Must include actual content (tail of last segment), not only "... (truncated)\n✅ *Complete*"
    assert escape_md_v2("... (truncated)") in msg
    assert "✅ *Complete*" in msg
    assert "actual output" in msg or "content" in msg or "x" in msg
