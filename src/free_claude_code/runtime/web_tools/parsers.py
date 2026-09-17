"""HTML parsing for web_search / web_fetch."""

import html
from dataclasses import replace
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

from free_claude_code.core.web_tools import WebSearchResult


class SearchResultParser(HTMLParser):
    """DuckDuckGo lite HTML: extract result links and titles."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[WebSearchResult] = []
        self._href: str | None = None
        self._title_parts: list[str] = []
        self._in_snippet = False
        self._snippet_parts: list[str] = []
        self._snippet_target: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "td" and "result-snippet" in (dict(attrs).get("class") or "").split():
            self._in_snippet = True
            self._snippet_parts = []
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if not href or "uddg=" not in href:
            return
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        uddg = query.get("uddg", [""])[0]
        if not uddg:
            return
        self._href = unquote(uddg)
        self._title_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_snippet:
            self._snippet_parts.append(data)
        if self._href is not None:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._in_snippet:
            self._in_snippet = False
            if self._snippet_target is not None:
                self.results[self._snippet_target] = replace(
                    self.results[self._snippet_target],
                    snippet=" ".join("".join(self._snippet_parts).split()),
                )
        if tag != "a" or self._href is None:
            return
        title = " ".join("".join(self._title_parts).split())
        if title and not any(result.url == self._href for result in self.results):
            self.results.append(
                WebSearchResult(title=html.unescape(title), url=self._href)
            )
        self._snippet_target = next(
            (
                index
                for index, result in enumerate(self.results)
                if result.url == self._href
            ),
            None,
        )
        self._href = None
        self._title_parts = []


class HTMLTextParser(HTMLParser):
    """Strip scripts/styles and collect visible text + title for fetch previews."""

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.text_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        elif not self._skip_depth:
            self.text_parts.append(text)
