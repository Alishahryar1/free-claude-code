"""Safe Markdown rendering for locally stored model-authored prose."""

from collections.abc import Sequence
from html import escape
from urllib.parse import urlsplit

from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.renderer import RendererHTML
from markdown_it.token import Token
from markdown_it.utils import EnvType, OptionsDict


def _safe_http_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return value


class _ProseRenderer(RendererHTML):
    def link_open(
        self,
        tokens: Sequence[Token],
        idx: int,
        options: OptionsDict,
        env: EnvType,
    ) -> str:
        token = tokens[idx]
        href = _safe_http_url(token.attrGet("href"))
        if href is None:
            token.attrSet("href", "#")
            token.attrSet("aria-disabled", "true")
        else:
            token.attrSet("href", href)
            token.attrSet("target", "_blank")
            token.attrSet("rel", "noopener noreferrer")
        return self.renderToken(tokens, idx, options, env)

    def image(
        self,
        tokens: Sequence[Token],
        idx: int,
        options: OptionsDict,
        env: EnvType,
    ) -> str:
        del self, options, env
        token = tokens[idx]
        label = token.content.strip() or "Image"
        source = _safe_http_url(token.attrGet("src"))
        if source is None:
            return escape(label)
        return (
            f'<a href="{escape(source, quote=True)}" target="_blank" '
            f'rel="noopener noreferrer">{escapeHtml(label)}</a>'
        )


_MARKDOWN = MarkdownIt(
    "commonmark",
    {"html": False, "linkify": False},
    renderer_cls=_ProseRenderer,
)


def render_markdown(value: str) -> str:
    """Render model text without raw HTML or active remote images."""

    return _MARKDOWN.render(value)
