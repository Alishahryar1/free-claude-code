"""Outbound HTTP for web_search / web_fetch (client, body caps, logging)."""

import asyncio
import socket
from collections.abc import AsyncIterator
from urllib.parse import urljoin, urlparse

import aiohttp
import httpx
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from aiohttp.abc import AbstractResolver, ResolveResult
from loguru import logger

from . import constants
from .constants import (
    _MAX_FETCH_CHARS,
    _MAX_SEARCH_RESULTS,
    _REDIRECT_RESPONSE_BODY_CAP_BYTES,
    _REQUEST_TIMEOUT_S,
    _WEB_FETCH_REDIRECT_STATUSES,
    _WEB_TOOL_HTTP_HEADERS,
)
from .egress import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
    get_validated_stream_addrinfos_for_egress,
)
from .parsers import HTMLTextParser, SearchResultParser


def _safe_public_host_for_logs(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host[:253]


def _log_web_tool_failure(
    tool_name: str,
    error: BaseException,
    *,
    fetch_url: str | None = None,
) -> None:
    exc_type = type(error).__name__
    if isinstance(error, WebFetchEgressViolation):
        host = _safe_public_host_for_logs(fetch_url) if fetch_url else ""
        logger.warning(
            "web_tool_egress_rejected tool={} exc_type={} host={!r}",
            tool_name,
            exc_type,
            host,
        )
        return
    if tool_name == "web_fetch" and fetch_url:
        logger.warning(
            "web_tool_failure tool={} exc_type={} host={!r}",
            tool_name,
            exc_type,
            _safe_public_host_for_logs(fetch_url),
        )
    else:
        logger.warning("web_tool_failure tool={} exc_type={}", tool_name, exc_type)


def _web_tool_client_error_summary(
    tool_name: str,
    error: BaseException,
    *,
    verbose: bool,
) -> str:
    if verbose:
        return f"{tool_name} failed: {type(error).__name__}"
    return "Web tool request failed."


async def _iter_response_body_under_cap(
    response: httpx.Response, max_bytes: int
) -> AsyncIterator[bytes]:
    if max_bytes <= 0:
        return
    received = 0
    async for chunk in response.aiter_bytes(chunk_size=65_536):
        if received >= max_bytes:
            break
        remaining = max_bytes - received
        if len(chunk) <= remaining:
            received += len(chunk)
            yield chunk
            if received >= max_bytes:
                break
        else:
            yield chunk[:remaining]
            break


async def _drain_response_body_capped(response: httpx.Response, max_bytes: int) -> None:
    async for _ in _iter_response_body_under_cap(response, max_bytes):
        pass


async def _read_response_body_capped(response: httpx.Response, max_bytes: int) -> bytes:
    return b"".join(
        [piece async for piece in _iter_response_body_under_cap(response, max_bytes)]
    )


_NUMERIC_RESOLVE_FLAGS = socket.AI_NUMERICHOST | socket.AI_NUMERICSERV
_NAME_RESOLVE_FLAGS = socket.NI_NUMERICHOST | socket.NI_NUMERICSERV


def getaddrinfo_rows_to_resolve_results(
    host: str, addrinfos: list[tuple]
) -> list[ResolveResult]:
    """Map :func:`socket.getaddrinfo` rows to aiohttp :class:`ResolveResult` (ThreadedResolver logic)."""
    out: list[ResolveResult] = []
    for family, _type, proto, _canon, sockaddr in addrinfos:
        if family == socket.AF_INET6:
            if len(sockaddr) < 3:
                continue
            if sockaddr[3]:
                resolved_host, port = socket.getnameinfo(sockaddr, _NAME_RESOLVE_FLAGS)
            else:
                resolved_host, port = sockaddr[:2]
        else:
            assert family == socket.AF_INET, family
            resolved_host, port = sockaddr[0], sockaddr[1]
            resolved_host = str(resolved_host)
            port = int(port)
        out.append(
            ResolveResult(
                hostname=host,
                host=resolved_host,
                port=int(port),
                family=family,
                proto=proto,
                flags=_NUMERIC_RESOLVE_FLAGS,
            )
        )
    return out


class _PinnedEgressStaticResolver(AbstractResolver):
    """Return only pre-validated :class:`ResolveResult` for the outbound request."""

    def __init__(self, results: list[ResolveResult]) -> None:
        self._results = results

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_INET
    ) -> list[ResolveResult]:
        return self._results

    async def close(self) -> None:  # pragma: no cover - aiohttp contract
        return


async def _read_aiohttp_body_capped(
    response: aiohttp.ClientResponse, max_bytes: int
) -> bytes:
    received = 0
    parts: list[bytes] = []
    async for chunk in response.content.iter_chunked(65_536):
        if received >= max_bytes:
            break
        remaining = max_bytes - received
        if len(chunk) <= remaining:
            received += len(chunk)
            parts.append(chunk)
        else:
            parts.append(chunk[:remaining])
            break
    return b"".join(parts)


async def _drain_aiohttp_body_capped(
    response: aiohttp.ClientResponse, max_bytes: int
) -> None:
    if max_bytes <= 0:
        return
    received = 0
    async for chunk in response.content.iter_chunked(65_536):
        received += len(chunk)
        if received >= max_bytes:
            break


async def _run_web_search(query: str) -> list[dict[str, str]]:
    async with (
        httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT_S,
            follow_redirects=True,
            headers=_WEB_TOOL_HTTP_HEADERS,
        ) as client,
        client.stream(
            "GET",
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
        ) as response,
    ):
        response.raise_for_status()
        body_bytes = await _read_response_body_capped(
            response, constants._MAX_WEB_FETCH_RESPONSE_BYTES
        )
    text = body_bytes.decode("utf-8", errors="replace")
    parser = SearchResultParser()
    parser.feed(text)
    return parser.results[:_MAX_SEARCH_RESULTS]


async def _run_web_fetch(url: str, egress: WebFetchEgressPolicy) -> dict[str, str]:
    """Fetch URL with manual redirects; each hop is DNS-pinned to validated addresses."""
    current_url = url
    redirect_hops = 0
    timeout = ClientTimeout(total=_REQUEST_TIMEOUT_S)

    while True:
        addr_infos = await asyncio.to_thread(
            get_validated_stream_addrinfos_for_egress, current_url, egress
        )
        host = urlparse(current_url).hostname or ""
        results = getaddrinfo_rows_to_resolve_results(host, addr_infos)
        resolver = _PinnedEgressStaticResolver(results)
        connector = TCPConnector(
            resolver=resolver,
            force_close=True,
        )
        try:
            async with (
                ClientSession(
                    timeout=timeout,
                    headers=_WEB_TOOL_HTTP_HEADERS,
                    connector=connector,
                ) as session,
                session.get(current_url, allow_redirects=False) as response,
            ):
                if response.status in _WEB_FETCH_REDIRECT_STATUSES:
                    await _drain_aiohttp_body_capped(
                        response, _REDIRECT_RESPONSE_BODY_CAP_BYTES
                    )
                    if redirect_hops >= constants._MAX_WEB_FETCH_REDIRECTS:
                        raise WebFetchEgressViolation(
                            "web_fetch exceeded maximum redirects "
                            f"({constants._MAX_WEB_FETCH_REDIRECTS})"
                        )
                    location = response.headers.get("location")
                    if not location or not location.strip():
                        raise WebFetchEgressViolation(
                            "web_fetch redirect response missing Location header"
                        )
                    current_url = urljoin(str(response.url), location.strip())
                    redirect_hops += 1
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "text/plain")
                final_url = str(response.url)
                encoding = response.charset or "utf-8"
                body_bytes = await _read_aiohttp_body_capped(
                    response, constants._MAX_WEB_FETCH_RESPONSE_BYTES
                )
        finally:
            await connector.close()

        break

    text = body_bytes.decode(encoding, errors="replace")
    title = final_url
    data = text
    if "html" in content_type.lower():
        parser = HTMLTextParser()
        parser.feed(text)
        title = parser.title or final_url
        data = "\n".join(parser.text_parts)
    return {
        "url": final_url,
        "title": title,
        "media_type": "text/plain",
        "data": data[:_MAX_FETCH_CHARS],
    }
