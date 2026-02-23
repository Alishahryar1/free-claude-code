from unittest.mock import MagicMock

import pytest

from messaging.rendering.telegram_markdown import (
    escape_md_v2,
    escape_md_v2_code,
    mdv2_bold,
    mdv2_code_inline,
    render_markdown_to_mdv2,
)
from messaging.transcript import RenderCtx, TranscriptBuffer


@pytest.fixture
def handler():
    platform = MagicMock()
    cli = MagicMock()
    store = MagicMock()
    # Kept for backwards test structure; transcript rendering is now separate.
    return (platform, cli, store)


def _ctx() -> RenderCtx:
    return RenderCtx(
        bold=mdv2_bold,
        code_inline=mdv2_code_inline,
        escape_code=escape_md_v2_code,
        escape_text=escape_md_v2,
        render_markdown=render_markdown_to_mdv2,
    )


def test_transcript_structure_and_order(handler):
    """Verify ordered transcript rendering (thinking/tool/subagent/text/error/status)."""
    status = "✅ *Complete*"
    t = TranscriptBuffer()

    # Apply in a deliberate sequence.
    t.apply({"type": "thinking_chunk", "text": "Thinking process..."})
    t.apply(
        {"type": "tool_use", "id": "t1", "name": "list_files", "input": {"path": "."}}
    )

    # Subagent marker (Task tool).
    t.apply(
        {
            "type": "tool_use",
            "id": "task1",
            "name": "Task",
            "input": {"description": "Searching codebase..."},
        }
    )
    t.apply(
        {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "x.py"}}
    )
    t.apply({"type": "tool_result", "tool_use_id": "task1", "content": "done"})

    t.apply({"type": "text_chunk", "text": "Here is the file content."})
    t.apply({"type": "error", "message": "Some error happened"})

    msg = t.render(_ctx(), limit_chars=3900, status=status)

    print(f"Generated Message:\n{msg}")

    # Check existence
    assert "Thinking process..." in msg
    assert "list_files" in msg
    assert "read_file" in msg
    assert "Searching codebase..." in msg
    assert escape_md_v2("Here is the file content.") in msg
    assert "Some error happened" in msg
    assert "✅ *Complete*" in msg

    # Check headers/markers used in the transcript.
    assert "💭 *Thinking*" in msg
    assert "🛠 *Tool call:*" in msg
    assert "🤖 *Subagent:*" in msg
    assert "⚠️ *Error:*" in msg

    # Check Order: Thinking -> Tool call -> Subagent -> Content -> Errors -> Status
    p_thinking = msg.find("Thinking process...")
    p_tool_call = msg.find("🛠 *Tool call:*")
    p_subagent = msg.find("🤖 *Subagent:*")
    p_content = msg.find(escape_md_v2("Here is the file content."))
    p_errors = msg.find("⚠️ *Error:*")
    p_status = msg.find("✅ *Complete*")

    assert p_thinking < p_tool_call, "Thinking should come before tool calls"
    assert p_tool_call < p_subagent, "Tool calls should come before subagent marker"
    assert p_subagent < p_content, "Subagent should come before Content"
    assert p_content < p_errors, "Content should come before Errors"
    assert p_errors < p_status, "Errors should come before Status"


def test_transcript_simple(handler):
    """Verify simple transcript with just text + status."""
    t = TranscriptBuffer()
    t.apply({"type": "text_chunk", "text": "Simple message."})
    msg = t.render(_ctx(), limit_chars=3900, status="Ready")

    assert escape_md_v2("Simple message.") in msg
    assert "Ready" in msg
    assert "💭" not in msg
    assert "🛠" not in msg


def test_subagents_formatting(handler):
    """Verify subagent formatting (Task tool)."""
    t = TranscriptBuffer()
    t.apply(
        {
            "type": "tool_use",
            "id": "task_1",
            "name": "Task",
            "input": {"description": "Task 1"},
        }
    )
    t.apply({"type": "tool_result", "tool_use_id": "task_1", "content": "done"})
    t.apply(
        {
            "type": "tool_use",
            "id": "task_2",
            "name": "Task",
            "input": {"description": "Task 2"},
        }
    )

    msg = t.render(_ctx(), limit_chars=3900, status=None)

    assert "🤖 *Subagent:* `Task 1`" in msg
    assert "🤖 *Subagent:* `Task 2`" in msg
