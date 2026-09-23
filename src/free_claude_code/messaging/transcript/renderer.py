"""Render and truncate ordered transcript segments."""

from collections import deque
from collections.abc import Iterable

from .context import RenderCtx
from .segments import Segment


def render_segments(
    segments: Iterable[Segment],
    ctx: RenderCtx,
    *,
    limit_chars: int,
    status: str | None,
) -> str:
    rendered: list[str] = []
    for segment in segments:
        try:
            output = segment.render(ctx)
        except Exception:
            continue
        if output:
            rendered.append(output)

    status_text = f"\n\n{status}" if status else ""
    prefix_marker = ctx.escape_text("... (truncated)\n")

    def _join(parts: Iterable[str], add_marker: bool) -> str:
        body = "\n".join(parts)
        if add_marker and body:
            body = prefix_marker + body
        return body + status_text if (body or status_text) else status_text

    candidate = _join(rendered, add_marker=False)
    if len(candidate) <= limit_chars:
        return candidate

    parts: deque[str] = deque(rendered)
    dropped = False
    last_part: str | None = None
    while parts:
        candidate = _join(parts, add_marker=True)
        if len(candidate) <= limit_chars:
            return candidate
        last_part = parts.popleft()
        dropped = True

    if dropped and last_part:
        budget = limit_chars - len(prefix_marker) - len(status_text)
        if budget > 20:
            tail = (
                "..." + last_part[-(budget - 3) :]
                if len(last_part) > budget
                else last_part
            )
            candidate = prefix_marker + tail + status_text
            if len(candidate) <= limit_chars:
                return candidate

    if dropped:
        minimal = prefix_marker + status_text.lstrip("\n")
        if len(minimal) <= limit_chars:
            return minimal
    return status or ""
