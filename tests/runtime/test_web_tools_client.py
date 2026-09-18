"""Outbound request, parsing, and resource ownership with no network access."""

import asyncio

import httpx
import pytest

from free_claude_code.core.web_tools import WebSearchResult
from free_claude_code.runtime.web_tools import client as web_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,cap,truncated,expected",
    [
        (b"short", 8, False, "short"),
        (b"12345678", 8, False, "12345678"),
        (b"123456789", 8, True, "12345678"),
        (
            b"<p>Start</p><script>" + b"x" * 200 + b"</script><p>needle</p>",
            100,
            True,
            "Start",
        ),
        (b"x" * 24001, 30000, True, "x" * 24000),
    ],
    ids=["below-cap", "exact-cap", "over-cap", "hidden-html", "visible-text"],
)
async def test_fetch_completeness_includes_raw_body_cap(
    monkeypatch, body, cap, truncated, expected
):
    from aiohttp import web

    from free_claude_code.application.web_tools.ports import WebFetchEgressPolicy

    monkeypatch.setattr(web_client.constants, "_MAX_WEB_FETCH_RESPONSE_BYTES", cap)

    async def page(request):
        response = web.StreamResponse(headers={"Content-Type": "text/html"})
        await response.prepare(request)
        for part in (body[:3], body[3:7], body[7:]):
            await response.write(part)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/", page)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        address = runner.addresses[0]
        assert isinstance(address, tuple)
        result = await web_client.HTTPWebToolsClient().fetch(
            f"http://127.0.0.1:{address[1]}/",
            egress=WebFetchEgressPolicy(True, frozenset({"http"})),
        )
        assert result.truncated is truncated
        assert result.data == expected
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("text/html", "<title>Café</title><p>Text to find.</p>".encode()),
        (
            "text/html; charset=not-a-codec",
            "<title>Café</title><p>Text to find.</p>".encode(),
        ),
        (
            "text/html; charset=iso-8859-1",
            "<title>Café</title><p>Text to find.</p>".encode("iso-8859-1"),
        ),
    ],
)
async def test_fetch_decodes_capped_stream_with_optional_charset(content_type, body):
    from aiohttp import web

    from free_claude_code.application.web_tools.ports import WebFetchEgressPolicy

    async def page(request):
        return web.Response(body=body, headers={"content-type": content_type})

    app = web.Application()
    app.router.add_get("/", page)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        address = runner.addresses[0]
        assert isinstance(address, tuple)
        result = await web_client.HTTPWebToolsClient().fetch(
            f"http://127.0.0.1:{address[1]}/",
            egress=WebFetchEgressPolicy(True, frozenset({"http"})),
        )
        assert result.title == "Café"
        assert "Text to find." in result.data
        assert not result.truncated
    finally:
        await runner.cleanup()


def _httpx_clients(monkeypatch, handler):
    original_client = httpx.AsyncClient
    clients = []

    def construct(**kwargs):
        client = original_client(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(web_client.httpx, "AsyncClient", construct)
    return clients


@pytest.mark.asyncio
async def test_search_request_parsing_limit_and_client_closure(monkeypatch):
    def handle(request):
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
    monkeypatch, cancel_during_body
):
    entered = asyncio.Event()

    class Body(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            entered.set()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            self.closed = True

    body = Body()

    async def handle(request):
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
async def test_fetch_cancellation_closes_response_session_and_connector(monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from free_claude_code.application.web_tools.ports import WebFetchEgressPolicy

    entered = asyncio.Event()
    response_closed, session_closed = [], []
    connectors = []
    original_connector = web_client.TCPConnector

    def connector(**kwargs):
        value = original_connector(**kwargs)
        connectors.append(value)
        return value

    async def read(_size):
        entered.set()
        await asyncio.Event().wait()
        return b""

    @asynccontextmanager
    async def response(url, *, allow_redirects):
        assert url == "https://8.8.8.8/"
        assert allow_redirects is False
        try:
            yield SimpleNamespace(
                status=200,
                url=url,
                headers={},
                charset="utf-8",
                raise_for_status=lambda: None,
                content=SimpleNamespace(read=read),
            )
        finally:
            response_closed.append(True)

    @asynccontextmanager
    async def session(**kwargs):
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
