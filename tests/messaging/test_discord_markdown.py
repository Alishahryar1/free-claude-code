"""Tests for messaging/rendering/discord_markdown.py."""

from messaging.rendering.discord_markdown import (
    _is_gfm_table_header_line,
    _normalize_gfm_tables,
    discord_bold,
    discord_code_inline,
    escape_discord,
    escape_discord_code,
    format_status,
    format_status_discord,
    render_markdown_to_discord,
)


class TestEscapeDiscord:
    """Tests for escape_discord."""

    def test_empty_string(self):
        assert escape_discord("") == ""

    def test_plain_text_unchanged(self):
        assert escape_discord("hello world") == "hello world"

    def test_special_chars_escaped(self):
        for ch in "\\*_`~|>":
            assert escape_discord(ch) == f"\\{ch}"

    def test_mixed_special_and_plain(self):
        assert escape_discord("a*b_c") == "a\\*b\\_c"

    def test_unicode_preserved(self):
        assert escape_discord("café 日本語") == "café 日本語"


class TestEscapeDiscordCode:
    """Tests for escape_discord_code."""

    def test_empty_string(self):
        assert escape_discord_code("") == ""

    def test_backslash_escaped(self):
        assert escape_discord_code("\\") == "\\\\"

    def test_backtick_escaped(self):
        assert escape_discord_code("`") == "\\`"

    def test_both_escaped(self):
        assert escape_discord_code("`\\") == "\\`\\\\"


class TestDiscordBold:
    """Tests for discord_bold."""

    def test_simple(self):
        assert discord_bold("hello") == "**hello**"

    def test_escapes_inner(self):
        assert discord_bold("a*b") == "**a\\*b**"


class TestDiscordCodeInline:
    """Tests for discord_code_inline."""

    def test_simple(self):
        assert discord_code_inline("x") == "`x`"

    def test_escapes_backtick(self):
        assert discord_code_inline("`") == "`\\``"


class TestFormatStatusDiscord:
    """Tests for format_status_discord."""

    def test_label_only(self):
        assert format_status_discord("Running") == "**Running**"

    def test_label_with_suffix(self):
        # Parentheses not in DISCORD_SPECIAL, so unchanged
        assert (
            format_status_discord("Queued", "(position 2)") == "**Queued** (position 2)"
        )


class TestFormatStatus:
    """Tests for format_status."""

    def test_label_only(self):
        assert format_status("🔄", "Running") == "🔄 **Running**"

    def test_label_with_suffix(self):
        assert format_status("⏳", "Waiting", "5/10") == "⏳ **Waiting** 5/10"


class TestIsGfmTableHeaderLine:
    """Tests for _is_gfm_table_header_line."""

    def test_no_pipe_returns_false(self):
        assert _is_gfm_table_header_line("hello world") is False

    def test_separator_only_returns_false(self):
        assert _is_gfm_table_header_line("|---|") is False
        assert _is_gfm_table_header_line("|:---|:---|") is False

    def test_valid_header(self):
        assert _is_gfm_table_header_line("| A | B |") is True
        assert _is_gfm_table_header_line("A | B") is True

    def test_single_column_returns_false(self):
        assert _is_gfm_table_header_line("| A |") is False


class TestNormalizeGfmTables:
    """Tests for _normalize_gfm_tables."""

    def test_single_line_unchanged(self):
        assert _normalize_gfm_tables("hello") == "hello"

    def test_two_lines_no_table_unchanged(self):
        assert _normalize_gfm_tables("a\nb") == "a\nb"

    def test_table_gets_blank_line_before(self):
        text = "para\n| A | B |\n|---|\n| 1 | 2 |"
        result = _normalize_gfm_tables(text)
        assert "para" in result
        assert "| A | B |" in result

    def test_table_inside_fence_unchanged(self):
        text = "```\n| A | B |\n|---|\n```"
        result = _normalize_gfm_tables(text)
        assert result == text


class TestRenderMarkdownToDiscord:
    """Tests for render_markdown_to_discord."""

    def test_empty_string(self):
        assert render_markdown_to_discord("") == ""

    def test_plain_paragraph(self):
        assert "hello" in render_markdown_to_discord("hello")

    def test_headings(self):
        result = render_markdown_to_discord("# Title\n## Sub")
        assert "Title" in result
        assert "Sub" in result

    def test_bold_italic(self):
        result = render_markdown_to_discord("**bold** *italic*")
        assert "bold" in result
        assert "italic" in result

    def test_strikethrough(self):
        result = render_markdown_to_discord("~~strike~~")
        assert "strike" in result

    def test_inline_code(self):
        result = render_markdown_to_discord("use `code` here")
        assert "`" in result
        assert "code" in result

    def test_code_block(self):
        result = render_markdown_to_discord("```\nprint(1)\n```")
        assert "print(1)" in result
        assert "```" in result

    def test_blockquote(self):
        result = render_markdown_to_discord("> quote")
        assert "quote" in result

    def test_bullet_list(self):
        result = render_markdown_to_discord("- a\n- b")
        assert "a" in result
        assert "b" in result

    def test_ordered_list(self):
        result = render_markdown_to_discord("1. first\n2. second")
        assert "first" in result
        assert "second" in result

    def test_link(self):
        result = render_markdown_to_discord("[text](https://example.com)")
        assert "text" in result
        assert "https://example.com" in result

    def test_image_with_alt(self):
        result = render_markdown_to_discord("![alt](https://img.png)")
        assert "alt" in result
        assert "https://img.png" in result

    def test_image_without_alt(self):
        result = render_markdown_to_discord("![](https://img.png)")
        assert "https://img.png" in result

    def test_gfm_table(self):
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = render_markdown_to_discord(text)
        assert "A" in result
        assert "B" in result
        assert "1" in result
        assert "2" in result
