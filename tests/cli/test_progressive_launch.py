import asyncio
import errno
import json
import socket
import threading
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from urllib.error import URLError

import pytest
import uvicorn
from fastapi import FastAPI

from free_claude_code.cli import commands
from free_claude_code.cli.launchers import common
from free_claude_code.cli.server_socket import ServerSockets
from free_claude_code.cli.uvicorn_server import RuntimeServer
from free_claude_code.config.settings import Settings


def test_listeners_remain_exclusive_until_owner_closes():
    with ServerSockets.reserve("127.0.0.1", 0) as owner:
        port = owner.sockets[0].getsockname()[1]
        with pytest.raises(OSError):
            ServerSockets.reserve("127.0.0.1", port)
    with ServerSockets.reserve("127.0.0.1", port) as replacement:
        assert replacement.sockets[0].getsockname()[1] == port


def test_partial_address_failure_closes_every_reserved_socket(monkeypatch):
    from free_claude_code.cli import server_socket

    listeners = [MagicMock(), MagicMock()]
    listeners[1].bind.side_effect = OSError(errno.EADDRINUSE, "busy")
    addresses = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 12345)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 12345, 0, 0)),
    ]
    monkeypatch.setattr(server_socket.socket, "getaddrinfo", lambda *args: addresses)
    monkeypatch.setattr(
        server_socket.socket, "socket", MagicMock(side_effect=listeners)
    )
    with pytest.raises(OSError):
        ServerSockets.reserve("localhost", 12345)
    for listener in listeners:
        listener.close.assert_called_once()
    listeners[0].listen.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["lifespan", "quit", "normal"])
async def test_uvicorn_readiness_and_early_exit_close_runtime(failure):
    ready, close = MagicMock(), AsyncMock(return_value=True)
    began = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_app):
        if failure == "lifespan":
            raise RuntimeError("controlled startup failure")
        if failure == "quit":
            server.should_exit = True
        yield

    app = FastAPI(lifespan=lifespan)
    with ServerSockets.reserve("127.0.0.1", 0) as owner:
        port = owner.sockets[0].getsockname()[1]

        def started():
            ready()
            server.should_exit = True

        server = RuntimeServer(
            uvicorn.Config(app, log_config=None, lifespan="on"),
            begin_shutdown=began.set,
            on_started=started,
            close_runtime=close,
        )
        async with asyncio.timeout(3):
            if failure == "lifespan":
                with pytest.raises(SystemExit) as exited:
                    await server.serve(owner.sockets)
                assert exited.value.code == 3
            else:
                await server.serve(owner.sockets)
        assert began.is_set()
        close.assert_awaited_once()
        assert ready.call_count == (1 if failure == "normal" else 0)
    with ServerSockets.reserve("127.0.0.1", port):
        pass


@pytest.mark.parametrize("valid", [False, True])
def test_existing_server_must_identify_itself_as_fcc(monkeypatch, valid):
    settings = Settings()
    payload = {"unrelated": "server"}
    if valid:
        payload = {
            "instance_id": "a" * 32,
            "status": "running",
            "host": settings.host,
            "port": settings.port,
            "provider_status": [],
            "cached_models": {},
        }
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    monkeypatch.setattr(
        commands, "open_local_request", MagicMock(return_value=response)
    )
    browser = MagicMock(return_value=True)
    monkeypatch.setattr(commands.webbrowser, "open", browser)
    assert commands.open_admin_when_ready(settings) is valid
    assert browser.call_count == int(valid)


def test_external_probe_is_cancelled_before_any_request(monkeypatch):
    stop = threading.Event()
    stop.set()
    request = MagicMock()
    monkeypatch.setattr(commands, "open_local_request", request)
    assert not commands.open_admin_when_ready(Settings(), stop_event=stop)
    request.assert_not_called()


def test_launcher_retries_starting_http_but_not_refused_socket(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value.status = 200
    request = MagicMock(side_effect=[TimeoutError("starting"), response])
    monkeypatch.setattr(common, "open_local_request", request)
    assert common.preflight_proxy("http://127.0.0.1:12345") is None
    assert request.call_count == 2
    request.reset_mock(side_effect=True)
    request.side_effect = URLError(ConnectionRefusedError("refused"))
    assert common.preflight_proxy("http://127.0.0.1:12345") == "refused"
    assert request.call_count == 1


def test_launcher_http_wait_has_a_finite_budget(monkeypatch):
    monkeypatch.setattr(common.time, "monotonic", MagicMock(side_effect=[0, 0, 31]))
    request = MagicMock(side_effect=TimeoutError("still starting"))
    monkeypatch.setattr(common, "open_local_request", request)
    assert common.preflight_proxy("http://127.0.0.1:12345") == "still starting"
    assert request.call_count == 1
