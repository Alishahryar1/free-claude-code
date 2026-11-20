"""Shared Markdown table pre-normalization for platform renderers."""

import re

from markdown_it import MarkdownIt

_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_CODE_BLOCK_PARSER = MarkdownIt("commonmark", {"html": False, "breaks": False})


def _is_gfm_table_header_line(line: str) -> bool:
    """Return whether a line looks like a GFM table header."""
    if "|" not in line:
        return False
    if _TABLE_SEP_RE.match(line):
        return False
    parts = [part.strip() for part in line.strip().strip("|").split("|")]
    return len([part for part in parts if part]) >= 2


def normalize_gfm_tables(text: str) -> str:
    """Insert blank lines before detected tables outside Markdown code blocks."""
    lines = text.splitlines()
    if len(lines) < 2:
        return text

    out_lines: list[str] = []
    # Use the renderers' Markdown grammar: list indentation and indented code
    # cannot be recognized reliably from an isolated fence line. Source maps
    # must use the same normalized lines that this function emits.
    code_ranges = iter(
        (token.map[0], token.map[1])
        for token in _CODE_BLOCK_PARSER.parse("\n".join(lines))
        if token.type in ("fence", "code_block") and token.map is not None
    )
    code_range = next(code_ranges, None)

    for idx, line in enumerate(lines):
        while code_range is not None and idx >= code_range[1]:
            code_range = next(code_ranges, None)
        if code_range is not None and idx >= code_range[0]:
            out_lines.append(line)
            continue

        if (
            idx + 1 < len(lines)
            and _is_gfm_table_header_line(line)
            and _TABLE_SEP_RE.match(lines[idx + 1])
            and out_lines
            and out_lines[-1].strip() != ""
        ):
            indent_match = re.match(r"^(\s*)", line)
            out_lines.append(indent_match.group(1) if indent_match else "")

        out_lines.append(line)

    return "\n".join(out_lines)
