"""Platform rendering profiles for messaging transcripts and status text."""

from collections.abc import Callable
from dataclasses import dataclass

from free_claude_code.messaging.rendering.discord_markdown import (
    discord_bold,
    discord_code_inline,
    escape_discord,
    escape_discord_code,
    render_markdown_to_discord,
)
from free_claude_code.messaging.rendering.discord_markdown import (
    format_status as format_status_discord,
)
from free_claude_code.messaging.rendering.telegram_markdown import (
    escape_md_v2,
    escape_md_v2_code,
    mdv2_bold,
    mdv2_code_inline,
    render_markdown_to_mdv2,
)
from free_claude_code.messaging.rendering.telegram_markdown import (
    format_status as format_status_telegram,
)
from free_claude_code.messaging.transcript import RenderCtx


@dataclass(frozen=True, slots=True)
class RenderingProfile:
    format_status: Callable[[str, str, str | None], str]
    parse_mode: str | None
    render_ctx: RenderCtx
    limit_chars: int


def build_rendering_profile(platform_name: str) -> RenderingProfile:
    """Return rendering rules for a messaging platform."""
    is_discord = platform_name == "discord"
    return RenderingProfile(
        format_status=format_status_discord if is_discord else format_status_telegram,
        parse_mode=None if is_discord else "MarkdownV2",
        render_ctx=RenderCtx(
            bold=discord_bold if is_discord else mdv2_bold,
            code_inline=discord_code_inline if is_discord else mdv2_code_inline,
            escape_code=escape_discord_code if is_discord else escape_md_v2_code,
            escape_text=escape_discord if is_discord else escape_md_v2,
            render_markdown=render_markdown_to_discord
            if is_discord
            else render_markdown_to_mdv2,
        ),
        limit_chars=1900 if is_discord else 3900,
    )
