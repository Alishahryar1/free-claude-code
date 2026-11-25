"""Typed synthetic upstream failures shared by key-rotation tests."""

import json

import httpx


class UpstreamError(RuntimeError):
    """A raw upstream failure with a status and a real structured HTTP response."""

    status_code: int | None
    response: httpx.Response | None

    def __init__(
        self,
        message: str = "upstream failure",
        *,
        status: int | None = None,
        headers: dict[str, str] | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status
        self.response = (
            httpx.Response(status, headers=headers or {}, text=body or "")
            if status is not None
            else None
        )
        if status is None and headers:
            self.response = httpx.Response(200, headers=headers)


def http_error(
    status: int,
    *,
    code: str | None = None,
    message: str = "",
    headers: dict[str, str] | None = None,
    kind: str = "error",
) -> UpstreamError:
    """An upstream failure carrying a structured JSON error body."""
    error: dict[str, str] = {"message": message, "type": kind}
    if code:
        error["code"] = code
    return UpstreamError(
        message or f"HTTP {status}",
        status=status,
        headers=headers,
        body=json.dumps({"error": error}),
    )
