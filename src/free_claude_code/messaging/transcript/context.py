"""Rendering context used by transcript segments."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class RenderCtx:
    bold: Callable[[str], str]
    code_inline: Callable[[str], str]
    escape_code: Callable[[str], str]
    escape_text: Callable[[str], str]
    render_markdown: Callable[[str], str]

    thinking_tail_max: int | None = 1000
    tool_input_tail_max: int | None = 1200
    tool_output_tail_max: int | None = 1600
    text_tail_max: int | None = 2000
