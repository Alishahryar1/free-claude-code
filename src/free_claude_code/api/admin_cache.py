"""Admin response cache policy."""

from fastapi import Response
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_ADMIN_CACHE_CONTROL = "no-store"


class AdminNoStoreMiddleware:
    """Prevent browsers from retaining responses from the admin surface."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        path = scope.get("path", "")
        if scope["type"] != "http" or not _is_admin_path(path):
            await self._app(scope, receive, send)
            return

        async def send_without_cache(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = dict(message)
                raw_headers = list(message.get("headers", ()))
                _set_no_store(MutableHeaders(raw=raw_headers))
                message["headers"] = raw_headers
            await send(message)

        await self._app(scope, receive, send_without_cache)


def attach_admin_no_store(response: Response, *, path: str) -> None:
    """Attach the policy when an outer server-error boundary bypasses middleware."""
    if _is_admin_path(path):
        _set_no_store(response.headers)


def _is_admin_path(path: str) -> bool:
    return path == "/admin" or path.startswith("/admin/")


def _set_no_store(headers: MutableHeaders) -> None:
    headers["Cache-Control"] = _ADMIN_CACHE_CONTROL
