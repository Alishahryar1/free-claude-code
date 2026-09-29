"""Uvicorn adapter imported only by the server worker."""

import socket
from collections.abc import Awaitable, Callable
from copy import deepcopy

import uvicorn

from free_claude_code.config.logging_config import InterceptHandler


def uvicorn_log_config(*, console: bool) -> dict[str, object]:
    """Send Uvicorn records to FCC's file and optional standard console handlers."""
    config = deepcopy(uvicorn.config.LOGGING_CONFIG)
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
