"""Preserve fenced code before messaging Markdown rendering."""

import pytest

from free_claude_code.messaging.rendering.discord_markdown import (
    render_markdown_to_discord,
)
from free_claude_code.messaging.rendering.markdown_tables import normalize_gfm_tables
from free_claude_code.messaging.rendering.telegram_markdown import (
    render_markdown_to_mdv2,
)


@pytest.mark.parametrize(
    "opening,inside,closing",
    [
        ("~~~", "example", "~~~"),
        ("~~~~python", "~~~\nexample", "~~~~"),
        ("````python", "```\nexample", "````"),
        ("```", "~~~\nexample", "```"),
        ("```", "```not-a-closing-fence\nexample", "```"),
        ("~~~", "~~~\u00a0\nexample", "~~~"),
        ("   ~~~", "example", "   ~~~~~"),
        ("~~~", "example", ""),
    ],
)
def test_preserves_table_like_content_in_fences(opening, inside, closing):
    text = f"{opening}\n{inside}\n| A | B |\n| --- | --- |"
    if closing:
        text += f"\n{closing}"
    assert normalize_gfm_tables(text) == text


@pytest.mark.parametrize("fence", ["```", "~~~", "~~~\t"])
def test_table_after_closed_fence_is_normalized(fence):
    text = f"{fence}\nexample\n{fence}\n| A | B |\n| --- | --- |"
    assert normalize_gfm_tables(text) == text.replace(
        f"{fence}\n| A", f"{fence}\n\n| A"
    )


@pytest.mark.parametrize(
    "renderer", [render_markdown_to_discord, render_markdown_to_mdv2]
)
@pytest.mark.parametrize("fence,inside", [("~~~", "example"), ("````", "```\nexample")])
def test_renderers_preserve_code_line_spacing(renderer, fence, inside):
    text = f"{fence}\n{inside}\n| A | B |\n| --- | --- |\n{fence}"
    rendered = renderer(text)
    assert "example\n| A | B |\n| --- | --- |" in rendered
    assert "example\n\n| A | B |" not in rendered
