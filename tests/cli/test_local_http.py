"""Local FCC transport must never escape through an outbound proxy."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from free_claude_code.cli.launchers.common import preflight_proxy


@contextmanager
def _status_server(status_code: int) -> Iterator[tuple[str, list[str]]]:
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            self.send_response(status_code)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_proxy_preflight_connects_directly_when_http_proxy_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _status_server(502) as (forward_proxy_url, forward_proxy_hits):
        monkeypatch.setenv("HTTP_PROXY", forward_proxy_url)
        monkeypatch.setenv("http_proxy", forward_proxy_url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)

        with _status_server(200) as (fcc_url, fcc_hits):
            assert preflight_proxy(fcc_url) is None

    assert fcc_hits == ["/health"]
    assert forward_proxy_hits == []
