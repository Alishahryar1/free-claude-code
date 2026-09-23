"""Shared Markdown table pre-normalization for platform renderers."""

import re

_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_FENCE_RE = re.compile(r"^\s*```")


def _is_gfm_table_header_line(line: str) -> bool:
    """Return whether a line looks like a GFM table header."""
    if "|" not in line:
        return False
    if _TABLE_SEP_RE.match(line):
        return False
    parts = [part.strip() for part in line.strip().strip("|").split("|")]
    return len([part for part in parts if part]) >= 2


def normalize_gfm_tables(text: str) -> str:
    """Insert blank lines before detected tables outside fenced code blocks."""
    lines = text.splitlines()
    if len(lines) < 2:
        return text

    out_lines: list[str] = []
    in_fence = False

    for idx, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out_lines.append(line)
            continue

        if (
            not in_fence
            and idx + 1 < len(lines)
            and _is_gfm_table_header_line(line)
            and _TABLE_SEP_RE.match(lines[idx + 1])
            and out_lines
            and out_lines[-1].strip() != ""
        ):
            indent_match = re.match(r"^(\s*)", line)
            out_lines.append(indent_match.group(1) if indent_match else "")

        out_lines.append(line)

    return "\n".join(out_lines)
