"""Uvicorn adapter imported only by the server worker."""

import logging
import socket
from collections.abc import Awaitable, Callable
from copy import deepcopy
from urllib.parse import parse_qsl, urlencode

import uvicorn

from free_claude_code.config.logging_config import InterceptHandler


class QueryCredentialFilter(logging.Filter):
    """Mask query API keys before either Uvicorn output handler sees them."""

    def filter(self, record: logging.LogRecord) -> bool:
        if (
            record.name == "uvicorn.access"
            and isinstance(record.args, tuple)
            and len(record.args) == 5
        ):
            address, method, target, version, status = record.args
            if isinstance(target, str) and "?" in target:
                path, _, query = target.partition("?")
                pairs = parse_qsl(query, keep_blank_values=True)
                if any(key == "key" for key, _ in pairs):
                    query = urlencode(
                        [
                            (key, "[REDACTED]" if key == "key" else value)
                            for key, value in pairs
                        ]
                    )
                    record.args = (address, method, f"{path}?{query}", version, status)
        return True


def uvicorn_log_config(*, console: bool) -> dict[str, object]:
    """Send Uvicorn records to FCC's file and optional standard console handlers."""
    config = deepcopy(uvicorn.config.LOGGING_CONFIG)
    config["filters"] = {"query_credentials": {"()": QueryCredentialFilter}}
    config["loggers"]["uvicorn.access"]["filters"] = ["query_credentials"]
    if not console:
        config["formatters"] = {}
        config["handlers"] = {}
    config["handlers"]["fcc"] = {"()": InterceptHandler}
    for name in ("uvicorn", "uvicorn.access"):
        handlers = config["loggers"][name]["handlers"]
        config["loggers"][name]["handlers"] = [*handlers, "fcc"] if console else ["fcc"]
    return config


class RuntimeServer(uvicorn.Server):
    """Notify the runtime before Uvicorn waits for long-lived HTTP responses."""

    def __init__(
        self,
        config: uvicorn.Config,
        *,
        begin_shutdown: Callable[[], None],
        on_started: Callable[[], None],
        close_runtime: Callable[[], Awaitable[bool]],
    ) -> None:
        super().__init__(config)
        self._begin_shutdown = begin_shutdown
        self._on_started = on_started
        self._close_runtime = close_runtime

    async def serve(self, sockets: list[socket.socket] | None = None) -> None:
        try:
            await super().serve(sockets)
        finally:
            # Uvicorn can return after startup without sending lifespan shutdown,
            # for example when Quit arrives during its startup handshake.
            self._begin_shutdown()
            await self._close_runtime()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets)
        if self.started and not self.should_exit:
            self._on_started()

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        self._begin_shutdown()
        await super().shutdown(sockets)
