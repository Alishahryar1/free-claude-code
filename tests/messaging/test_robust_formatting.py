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
    return (platform, cli, store)


def _ctx() -> RenderCtx:
    return RenderCtx(
        bold=mdv2_bold,
        code_inline=mdv2_code_inline,
        escape_code=escape_md_v2_code,
        escape_text=escape_md_v2,
        render_markdown=render_markdown_to_mdv2,
    )


def test_truncation_closes_code_blocks(handler):
    """Verify that truncation correctly closes open code blocks."""
    t = TranscriptBuffer()
    t.apply(
        {
            "type": "thinking_chunk",
            "text": "Starting some long thinking process that will definitely cause truncation later on...",
        }
    )
    t.apply(
        {
            "type": "text_chunk",
            "text": "```python\ndef very_long_function():\n    # " + ("A" * 4000),
        }
    )

    msg = t.render(_ctx(), limit_chars=3900, status="✅ *Complete*")

    # The backtick count must be even to be a valid block.
    assert msg.count("```") % 2 == 0
    assert msg.endswith("```") or "✅ *Complete*" in msg.split("```")[-1]


def test_truncation_preserves_status(handler):
    """Verify that status is still appended after truncation."""
    status = "READY_STATUS"
    t = TranscriptBuffer()
    t.apply({"type": "thinking_chunk", "text": "Thinking..."})
    t.apply({"type": "text_chunk", "text": "A" * 5000})
    msg = t.render(_ctx(), limit_chars=3900, status=status)

    assert status in msg


def test_empty_components_with_status(handler):
    """Verify message building with just a status."""
    status = "Simple Status"
    t = TranscriptBuffer()
    msg = t.render(_ctx(), limit_chars=3900, status=status)
    assert msg == "\n\nSimple Status"


def test_render_markdown_unclosed_markdown():
    """Malformed markdown (e.g. unclosed *) does not crash and produces acceptable output."""
    from messaging.rendering.telegram_markdown import render_markdown_to_mdv2

    md = "*bold without close"
    out = render_markdown_to_mdv2(md)
    assert out is not None
    assert "bold" in out


def test_escape_md_v2_unicode_emoji():
    """Unicode and emoji pass through correctly (no special char escaping needed)."""
    from messaging.rendering.telegram_markdown import escape_md_v2, escape_md_v2_code

    text = "Hello 世界 🎉 café"
    assert escape_md_v2(text) == text
    assert escape_md_v2_code(text) == text
