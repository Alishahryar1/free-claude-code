"""Telegram MarkdownV2 utilities.

Renders common Markdown into Telegram MarkdownV2 format.
Used by the message handler and Telegram platform adapter.
"""

import re
from typing import List, Optional

from markdown_it import MarkdownIt

MDV2_SPECIAL_CHARS = set("\\_*[]()~`>#+-=|{}.!")
MDV2_LINK_ESCAPE = set("\\)")

_MD = MarkdownIt("commonmark", {"html": False, "breaks": False})
_MD.enable("strikethrough")
_MD.enable("table")

_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_FENCE_RE = re.compile(r"^\s*```")


def _is_gfm_table_header_line(line: str) -> bool:
    """Check if line is a GFM table header (pipe-delimited, not separator)."""
    if "|" not in line:
        return False
    if _TABLE_SEP_RE.match(line):
        return False
    stripped = line.strip()
    parts = [p.strip() for p in stripped.strip("|").split("|")]
    parts = [p for p in parts if p != ""]
    return len(parts) >= 2


def _normalize_gfm_tables(text: str) -> str:
    """
    Many LLMs emit tables immediately after a paragraph line (no blank line).
    Markdown-it will treat that as a softbreak within the paragraph, so the
    table extension won't trigger. Insert a blank line before detected tables.

    We only do this outside fenced code blocks.
    """
    lines = text.splitlines()
    if len(lines) < 2:
        return text

    out_lines: List[str] = []
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
        ):
            if out_lines and out_lines[-1].strip() != "":
                m = re.match(r"^(\s*)", line)
                indent = m.group(1) if m else ""
                out_lines.append(indent)

        out_lines.append(line)

    return "\n".join(out_lines)


def escape_md_v2(text: str) -> str:
    """Escape text for Telegram MarkdownV2."""
    return "".join(f"\\{ch}" if ch in MDV2_SPECIAL_CHARS else ch for ch in text)


def escape_md_v2_code(text: str) -> str:
    """Escape text for Telegram MarkdownV2 code spans/blocks."""
    return text.replace("\\", "\\\\").replace("`", "\\`")


def escape_md_v2_link_url(text: str) -> str:
    """Escape URL for Telegram MarkdownV2 link destination."""
    return "".join(f"\\{ch}" if ch in MDV2_LINK_ESCAPE else ch for ch in text)


def mdv2_bold(text: str) -> str:
    """Format text as bold in MarkdownV2."""
    return f"*{escape_md_v2(text)}*"


def mdv2_code_inline(text: str) -> str:
    """Format text as inline code in MarkdownV2."""
    return f"`{escape_md_v2_code(text)}`"


def format_status(emoji: str, label: str, suffix: Optional[str] = None) -> str:
    """Format a status message with emoji and optional suffix."""
    base = f"{emoji} {mdv2_bold(label)}"
    if suffix:
        return f"{base} {escape_md_v2(suffix)}"
    return base


def render_markdown_to_mdv2(text: str) -> str:
    """Render common Markdown into Telegram MarkdownV2."""
    if not text:
        return ""

    text = _normalize_gfm_tables(text)
    tokens = _MD.parse(text)

    def render_inline_table_plain(children) -> str:
        out: List[str] = []
        for tok in children:
            if tok.type == "text":
                out.append(tok.content)
            elif tok.type == "code_inline":
                out.append(tok.content)
            elif tok.type in {"softbreak", "hardbreak"}:
                out.append(" ")
            elif tok.type == "image":
                if tok.content:
                    out.append(tok.content)
        return "".join(out)

    def render_inline_plain(children) -> str:
        out: List[str] = []
        for tok in children:
            if tok.type == "text":
                out.append(escape_md_v2(tok.content))
            elif tok.type == "code_inline":
                out.append(escape_md_v2(tok.content))
            elif tok.type in {"softbreak", "hardbreak"}:
                out.append("\n")
        return "".join(out)

    def render_inline(children) -> str:
        out: List[str] = []
        i = 0
        while i < len(children):
            tok = children[i]
            t = tok.type
            if t == "text":
                out.append(escape_md_v2(tok.content))
            elif t in {"softbreak", "hardbreak"}:
                out.append("\n")
            elif t == "em_open":
                out.append("_")
            elif t == "em_close":
                out.append("_")
            elif t == "strong_open":
                out.append("*")
            elif t == "strong_close":
                out.append("*")
            elif t == "s_open":
                out.append("~")
            elif t == "s_close":
                out.append("~")
            elif t == "code_inline":
                out.append(f"`{escape_md_v2_code(tok.content)}`")
            elif t == "link_open":
                href = ""
                if tok.attrs:
                    if isinstance(tok.attrs, dict):
                        href = tok.attrs.get("href", "")
                    else:
                        for key, val in tok.attrs:
                            if key == "href":
                                href = val
                                break
                inner_tokens = []
                i += 1
                while i < len(children) and children[i].type != "link_close":
                    inner_tokens.append(children[i])
                    i += 1
                link_text = ""
                for child in inner_tokens:
                    if child.type == "text":
                        link_text += child.content
                    elif child.type == "code_inline":
                        link_text += child.content
                out.append(
                    f"[{escape_md_v2(link_text)}]({escape_md_v2_link_url(href)})"
                )
            elif t == "image":
                href = ""
                alt = tok.content or ""
                if tok.attrs:
                    if isinstance(tok.attrs, dict):
                        href = tok.attrs.get("src", "")
                    else:
                        for key, val in tok.attrs:
                            if key == "src":
                                href = val
                                break
                if alt:
                    out.append(f"{escape_md_v2(alt)} ({escape_md_v2_link_url(href)})")
                else:
                    out.append(escape_md_v2_link_url(href))
            else:
                out.append(escape_md_v2(tok.content or ""))
            i += 1
        return "".join(out)

    out: List[str] = []
    list_stack: List[dict] = []
    pending_prefix: Optional[str] = None
    blockquote_level = 0
    in_heading = False

    def apply_blockquote(val: str) -> str:
        if blockquote_level <= 0:
            return val
        prefix = "> " * blockquote_level
        return prefix + val.replace("\n", "\n" + prefix)

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        t = tok.type
        if t == "paragraph_open":
            pass
        elif t == "paragraph_close":
            out.append("\n")
        elif t == "heading_open":
            in_heading = True
        elif t == "heading_close":
            in_heading = False
            out.append("\n")
        elif t == "bullet_list_open":
            list_stack.append({"type": "bullet", "index": 1})
        elif t == "bullet_list_close":
            if list_stack:
                list_stack.pop()
            out.append("\n")
        elif t == "ordered_list_open":
            start = 1
            if tok.attrs:
                if isinstance(tok.attrs, dict):
                    val = tok.attrs.get("start")
                    if val is not None:
                        try:
                            start = int(val)
                        except TypeError, ValueError:
                            start = 1
                else:
                    for key, val in tok.attrs:
                        if key == "start":
                            try:
                                start = int(val)
                            except TypeError, ValueError:
                                start = 1
                            break
            list_stack.append({"type": "ordered", "index": start})
        elif t == "ordered_list_close":
            if list_stack:
                list_stack.pop()
            out.append("\n")
        elif t == "list_item_open":
            if list_stack:
                top = list_stack[-1]
                if top["type"] == "bullet":
                    pending_prefix = "\\- "
                else:
                    pending_prefix = f"{top['index']}\\."
                    top["index"] += 1
                    pending_prefix += " "
        elif t == "list_item_close":
            out.append("\n")
        elif t == "blockquote_open":
            blockquote_level += 1
        elif t == "blockquote_close":
            blockquote_level = max(0, blockquote_level - 1)
            out.append("\n")
        elif t == "table_open":
            if pending_prefix:
                out.append(apply_blockquote(pending_prefix.rstrip()))
                out.append("\n")
                pending_prefix = None

            rows: List[List[str]] = []
            row_is_header: List[bool] = []

            j = i + 1
            in_thead = False
            in_row = False
            current_row: List[str] = []
            current_row_header = False

            in_cell = False
            cell_parts: List[str] = []

            while j < len(tokens):
                tt = tokens[j].type
                if tt == "thead_open":
                    in_thead = True
                elif tt == "thead_close":
                    in_thead = False
                elif tt == "tr_open":
                    in_row = True
                    current_row = []
                    current_row_header = in_thead
                elif tt in {"th_open", "td_open"}:
                    in_cell = True
                    cell_parts = []
                elif tt == "inline" and in_cell:
                    cell_parts.append(
                        render_inline_table_plain(tokens[j].children or [])
                    )
                elif tt in {"th_close", "td_close"} and in_cell:
                    cell = " ".join(cell_parts).strip()
                    current_row.append(cell)
                    in_cell = False
                    cell_parts = []
                elif tt == "tr_close" and in_row:
                    rows.append(current_row)
                    row_is_header.append(bool(current_row_header))
                    in_row = False
                elif tt == "table_close":
                    break
                j += 1

            if rows:
                col_count = max((len(r) for r in rows), default=0)
                norm_rows: List[List[str]] = []
                for r in rows:
                    if len(r) < col_count:
                        r = r + [""] * (col_count - len(r))
                    norm_rows.append(r)

                widths: List[int] = []
                for c in range(col_count):
                    w = max((len(r[c]) for r in norm_rows), default=0)
                    widths.append(max(w, 3))

                def fmt_row(r: List[str]) -> str:
                    cells = [r[c].ljust(widths[c]) for c in range(col_count)]
                    return "| " + " | ".join(cells) + " |"

                def fmt_sep() -> str:
                    cells = ["-" * widths[c] for c in range(col_count)]
                    return "| " + " | ".join(cells) + " |"

                last_header_idx = -1
                for idx, is_h in enumerate(row_is_header):
                    if is_h:
                        last_header_idx = idx

                lines: List[str] = []
                for idx, r in enumerate(norm_rows):
                    lines.append(fmt_row(r))
                    if idx == last_header_idx:
                        lines.append(fmt_sep())

                table_text = "\n".join(lines).rstrip()
                out.append(f"```\n{escape_md_v2_code(table_text)}\n```")
                out.append("\n")

            i = j + 1
            continue
        elif t in {"code_block", "fence"}:
            code = escape_md_v2_code(tok.content.rstrip("\n"))
            out.append(f"```\n{code}\n```")
            out.append("\n")
        elif t == "inline":
            rendered = render_inline(tok.children or [])
            if in_heading:
                rendered = f"*{render_inline_plain(tok.children or [])}*"
            if pending_prefix:
                rendered = pending_prefix + rendered
                pending_prefix = None
            rendered = apply_blockquote(rendered)
            out.append(rendered)
        else:
            if tok.content:
                out.append(escape_md_v2(tok.content))
        i += 1

    return "".join(out).rstrip()


__all__ = [
    "escape_md_v2",
    "escape_md_v2_code",
    "escape_md_v2_link_url",
    "mdv2_bold",
    "mdv2_code_inline",
    "format_status",
    "render_markdown_to_mdv2",
]
