"""Shared Markdown table pre-normalization for platform renderers."""

import re

_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


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
    fence_marker = ""
    fence_length = 0

    for idx, line in enumerate(lines):
        fence_match = _FENCE_RE.match(line)
        if fence_marker:
            if fence_match:
                marker, suffix = fence_match.groups()
                if (
                    marker[0] == fence_marker
                    and len(marker) >= fence_length
                    and not suffix.strip(" \t")
                ):
                    fence_marker = ""
            out_lines.append(line)
            continue
        if fence_match:
            marker, suffix = fence_match.groups()
            # Backtick fence info strings cannot themselves contain backticks.
            if marker[0] == "~" or "`" not in suffix:
                fence_marker = marker[0]
                fence_length = len(marker)
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
