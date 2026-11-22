"""Outbound request, parsing, and resource ownership with no network access."""

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from typing import override

import httpx
import pytest
from aiohttp.abc import AbstractResolver

from free_claude_code.core.web_tools import WebSearchResult
from free_claude_code.runtime.web_tools import client as web_client


def _httpx_clients(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
):
    original_client = httpx.AsyncClient
    clients = []

    def construct(
        *, timeout: float, follow_redirects: bool, headers: Mapping[str, str]
    ):
        client = original_client(
            timeout=timeout,
            follow_redirects=follow_redirects,
            headers=headers,
            transport=httpx.MockTransport(handler),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(web_client.httpx, "AsyncClient", construct)
    return clients


@pytest.mark.asyncio
async def test_search_request_parsing_limit_and_client_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handle(request: httpx.Request):
        assert str(request.url).startswith("https://lite.duckduckgo.com/lite/")
        assert request.url.params["q"] == "query & details"
        assert "free-claude-code/" in request.headers["User-Agent"]
        links = [
            f'<a href="/l/?uddg=https%3A%2F%2Fexample.com%2F{index}">Title {index}</a>'
            for index in range(12)
        ]
        links.insert(1, links[0])
        return httpx.Response(200, text="".join(links))

    clients = _httpx_clients(monkeypatch, handle)
    result = await web_client.HTTPWebToolsClient().search("query & details")
    assert result == [
        WebSearchResult(title=f"Title {index}", url=f"https://example.com/{index}")
        for index in range(10)
    ]
    assert len(clients) == 1 and clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during_body", [False, True])
async def test_search_cancellation_closes_client_and_open_response(
    monkeypatch: pytest.MonkeyPatch, cancel_during_body: bool
) -> None:
    entered = asyncio.Event()

    class Body(httpx.AsyncByteStream):
        closed = False

        @override
        async def __aiter__(self):
            entered.set()
            await asyncio.Event().wait()
            yield b""

        @override
        async def aclose(self):
            self.closed = True

    body = Body()

    async def handle(request: object):
        if not cancel_during_body:
            entered.set()
            await asyncio.Event().wait()
        return httpx.Response(200, stream=body)

    clients = _httpx_clients(monkeypatch, handle)
    task = asyncio.create_task(web_client.HTTPWebToolsClient().search("held"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(clients) == 1 and clients[0].is_closed
        assert body.closed is cancel_during_body
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_fetch_cancellation_closes_response_session_and_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from free_claude_code.application.web_tools.ports import WebFetchEgressPolicy

    entered = asyncio.Event()
    response_closed: list[bool] = []
    session_closed: list[bool] = []
    connectors = []
    original_connector = web_client.TCPConnector

    def connector(*, resolver: AbstractResolver, force_close: bool):
        value = original_connector(resolver=resolver, force_close=force_close)
        connectors.append(value)
        return value

    async def chunks(_size):
        entered.set()
        await asyncio.Event().wait()
        yield b""

    @asynccontextmanager
    async def response(url: str, *, allow_redirects: bool):
        assert url == "https://8.8.8.8/"
        assert allow_redirects is False
        try:
            yield SimpleNamespace(
                status=200,
                url=url,
                headers={},
                charset=None,
                raise_for_status=lambda: None,
                content=SimpleNamespace(iter_chunked=chunks),
            )
        finally:
            response_closed.append(True)

    @asynccontextmanager
    async def session(**kwargs: object):
        assert kwargs["connector"] is connectors[-1]
        try:
            yield SimpleNamespace(get=response)
        finally:
            session_closed.append(True)

    monkeypatch.setattr(web_client, "TCPConnector", connector)
    monkeypatch.setattr(web_client, "ClientSession", session)
    task = asyncio.create_task(
        web_client.HTTPWebToolsClient().fetch(
            "https://8.8.8.8/",
            egress=WebFetchEgressPolicy(False, frozenset({"https"})),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert response_closed == [True]
        assert session_closed == [True]
        assert len(connectors) == 1 and connectors[0].closed
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "charset", "body", "expected_title", "expected_data"),
    [
        (
            {"content-type": "text/html"},
            None,
            b"<html><head><title>Test Title</title></head><body><p>Hello world</p></body></html>",
            "Test Title",
            "Hello world",
        ),
        (
            {},
            None,
            b"Plain utf-8 body",
            "https://8.8.8.8/",
            "Plain utf-8 body",
        ),
        (
            {"content-type": "text/plain; charset=iso-8859-1"},
            "iso-8859-1",
            b"caf\xe9",
            "https://8.8.8.8/",
            "café",
        ),
        (
            {"content-type": "text/plain; charset=bogus"},
            "bogus",
            b"fallback text",
            "https://8.8.8.8/",
            "fallback text",
        ),
        (
            {"content-type": "application/pdf"},
            None,
            b"%PDF-1.4\n%mangled\xff\xfe",
            "https://8.8.8.8/",
            "%PDF-1.4\n%mangled\ufffd\ufffd",
        ),
    ],
)
async def test_fetch_charset_and_content_type_handling(
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
    charset: str | None,
    body: bytes,
    expected_title: str,
    expected_data: str,
) -> None:
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from free_claude_code.application.web_tools.ports import WebFetchEgressPolicy

    async def chunks(_size):
        yield body

    def raise_for_unread_encoding():
        if charset:
            return charset
        raise RuntimeError("Cannot compute fallback encoding of a not yet read body")

    @asynccontextmanager
    async def response(url: str, *, allow_redirects: bool):
        assert url == "https://8.8.8.8/"
        assert allow_redirects is False
        yield SimpleNamespace(
            status=200,
            url=url,
            headers=headers,
            charset=charset,
            get_encoding=raise_for_unread_encoding,
            raise_for_status=lambda: None,
            content=SimpleNamespace(iter_chunked=chunks),
        )

    @asynccontextmanager
    async def session(**kwargs: object):
        yield SimpleNamespace(get=response)

    monkeypatch.setattr(web_client, "ClientSession", session)
    result = await web_client.HTTPWebToolsClient().fetch(
        "https://8.8.8.8/",
        egress=WebFetchEgressPolicy(False, frozenset({"https"})),
    )
    assert result.url == "https://8.8.8.8/"
    assert result.title == expected_title
    assert result.media_type == "text/plain"
    assert result.data == expected_data


@pytest.mark.parametrize(
    ("charset", "expected"),
    [
        (None, "utf-8"),
        ("", "utf-8"),
        ("utf-8", "utf-8"),
        ("UTF-8", "utf-8"),
        ("iso-8859-1", "iso8859-1"),
        ("latin-1", "iso8859-1"),
        ("bogus", "utf-8"),
        ("invalid\x00null", "utf-8"),
    ],
)
def test_text_encoding(charset: str | None, expected: str) -> None:
    assert web_client._text_encoding(charset) == expected
