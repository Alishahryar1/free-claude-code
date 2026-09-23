"""Discord markdown utilities.

Discord uses standard markdown: **bold**, *italic*, `code`, ```code block```.
Used by the message handler and Discord platform adapter.
"""

from markdown_it import MarkdownIt

from .markdown_tables import normalize_gfm_tables

# Discord escapes: \ * _ ` ~ | >
DISCORD_SPECIAL = set("\\*_`~|>")

_MD = MarkdownIt("commonmark", {"html": False, "breaks": False})
_MD.enable("strikethrough")
_MD.enable("table")


def escape_discord(text: str) -> str:
    """Escape text for Discord markdown (bold, italic, etc.)."""
    return "".join(f"\\{ch}" if ch in DISCORD_SPECIAL else ch for ch in text)


def escape_discord_code(text: str) -> str:
    """Escape text for Discord code spans/blocks."""
    return text.replace("\\", "\\\\").replace("`", "\\`")


def discord_bold(text: str) -> str:
    """Format text as bold in Discord (uses **)."""
    return f"**{escape_discord(text)}**"


def discord_code_inline(text: str) -> str:
    """Format text as inline code in Discord."""
    return f"`{escape_discord_code(text)}`"


def format_status_discord(label: str, suffix: str | None = None) -> str:
    """Format a status message for Discord (label in bold, optional suffix)."""
    base = discord_bold(label)
    if suffix:
        return f"{base} {escape_discord(suffix)}"
    return base


def format_status(emoji: str, label: str, suffix: str | None = None) -> str:
    """Format a status message with emoji for Discord (matches Telegram API)."""
    base = f"{emoji} {discord_bold(label)}"
    if suffix:
        return f"{base} {escape_discord(suffix)}"
    return base


def render_markdown_to_discord(text: str) -> str:
    """Render common Markdown into Discord-compatible format."""
    if not text:
        return ""

    text = normalize_gfm_tables(text)
    tokens = _MD.parse(text)

    def render_inline_table_plain(children) -> str:
        out: list[str] = []
        for tok in children:
            if tok.type == "text" or tok.type == "code_inline":
                out.append(tok.content)
            elif tok.type in {"softbreak", "hardbreak"}:
                out.append(" ")
            elif tok.type == "image" and tok.content:
                out.append(tok.content)
        return "".join(out)

    def render_inline(children) -> str:
        out: list[str] = []
        i = 0
        while i < len(children):
            tok = children[i]
            t = tok.type
            if t == "text":
                out.append(escape_discord(tok.content))
            elif t in {"softbreak", "hardbreak"}:
                out.append("\n")
            elif t == "em_open" or t == "em_close":
                out.append("*")
            elif t == "strong_open" or t == "strong_close":
                out.append("**")
            elif t == "s_open" or t == "s_close":
                out.append("~~")
            elif t == "code_inline":
                out.append(f"`{escape_discord_code(tok.content)}`")
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
                    if child.type == "text" or child.type == "code_inline":
                        link_text += child.content
                out.append(f"[{escape_discord(link_text)}]({href})")
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
                    out.append(f"{escape_discord(alt)} ({href})")
                else:
                    out.append(href)
            else:
                out.append(escape_discord(tok.content or ""))
            i += 1
        return "".join(out)

    out: list[str] = []
    list_stack: list[dict] = []
    pending_prefix: str | None = None
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
                    pending_prefix = "- "
                else:
                    pending_prefix = f"{top['index']}. "
                    top["index"] += 1
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

            rows: list[list[str]] = []
            row_is_header: list[bool] = []

            j = i + 1
            in_thead = False
            in_row = False
            current_row: list[str] = []
            current_row_header = False

            in_cell = False
            cell_parts: list[str] = []

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
                norm_rows: list[list[str]] = []
                for r in rows:
                    if len(r) < col_count:
                        r = r + [""] * (col_count - len(r))
                    norm_rows.append(r)

                widths: list[int] = []
                for c in range(col_count):
                    w = max((len(r[c]) for r in norm_rows), default=0)
                    widths.append(max(w, 3))

                def fmt_row(
                    r: list[str], _w: list[int] = widths, _c: int = col_count
                ) -> str:
                    cells = [r[c].ljust(_w[c]) for c in range(_c)]
                    return "| " + " | ".join(cells) + " |"

                def fmt_sep(_w: list[int] = widths, _c: int = col_count) -> str:
                    cells = ["-" * _w[c] for c in range(_c)]
                    return "| " + " | ".join(cells) + " |"

                last_header_idx = -1
                for idx, is_h in enumerate(row_is_header):
                    if is_h:
                        last_header_idx = idx

                lines: list[str] = []
                for idx, r in enumerate(norm_rows):
                    lines.append(fmt_row(r))
                    if idx == last_header_idx:
                        lines.append(fmt_sep())

                table_text = "\n".join(lines).rstrip()
                out.append(f"```\n{escape_discord_code(table_text)}\n```")
                out.append("\n")

            i = j + 1
            continue
        elif t in {"code_block", "fence"}:
            code = escape_discord_code(tok.content.rstrip("\n"))
            out.append(f"```\n{code}\n```")
            out.append("\n")
        elif t == "inline":
            rendered = render_inline(tok.children or [])
            if in_heading:
                rendered = f"**{render_inline(tok.children or [])}**"
            if pending_prefix:
                rendered = pending_prefix + rendered
                pending_prefix = None
            rendered = apply_blockquote(rendered)
            out.append(rendered)
        else:
            if tok.content:
                out.append(escape_discord(tok.content))
        i += 1

    return "".join(out).rstrip()


__all__ = [
    "discord_bold",
    "discord_code_inline",
    "escape_discord",
    "escape_discord_code",
    "format_status",
    "format_status_discord",
    "render_markdown_to_discord",
]
