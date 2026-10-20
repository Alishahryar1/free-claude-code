"""Direct HTTP transport for FCC-local traffic."""

from http.client import HTTPResponse
from urllib.request import ProxyHandler, Request, build_opener

_DIRECT_OPENER = build_opener(ProxyHandler({}))


def open_local_request(request: Request, *, timeout: float) -> HTTPResponse:
    """Open an FCC-local request without consulting machine proxy settings."""

    return _DIRECT_OPENER.open(request, timeout=timeout)
