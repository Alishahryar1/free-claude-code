"""Egress policy for user-controlled web_fetch URLs (SSRF guard)."""

import ipaddress
import socket
from urllib.parse import urlparse

from free_claude_code.application.web_tools.ports import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
)
from free_claude_code.core.web_domains import domain_matches


def _port_for_url(parsed) -> int:
    if parsed.port is not None:
        return parsed.port
    return 443 if (parsed.scheme or "").lower() == "https" else 80


def _stream_getaddrinfo_or_raise(host: str, port: int) -> list[tuple]:
    try:
        return socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except OSError as exc:
        raise WebFetchEgressViolation(
            f"Could not resolve host {host!r}: {exc}"
        ) from exc


def get_validated_stream_addrinfos_for_egress(
    url: str, policy: WebFetchEgressPolicy
) -> list[tuple]:
    """Resolve and validate a URL for web_fetch, returning getaddrinfo rows for pinning.

    Each HTTP connect pins to only these `getaddrinfo` results so a malicious DNS
    server cannot rebind to a disallowed address between resolution and the TCP
    connect (used by :meth:`runtime.web_tools.client.HTTPWebToolsClient.fetch`).
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in policy.allowed_schemes:
        raise WebFetchEgressViolation(
            f"URL scheme {scheme!r} is not allowed for web_fetch"
        )

    host = parsed.hostname
    if host is None or host == "":
        raise WebFetchEgressViolation("web_fetch URL must include a host")

    if policy.allowed_domains and not any(
        domain_matches(host, domain) for domain in policy.allowed_domains
    ):
        raise WebFetchEgressViolation("web_fetch URL is outside the allowed domains")

    port = _port_for_url(parsed)

    if policy.allow_private_network_targets:
        return _stream_getaddrinfo_or_raise(host, port)

    host_lower = host.lower()
    if host_lower == "localhost" or host_lower.endswith(".localhost"):
        raise WebFetchEgressViolation("localhost targets are not allowed for web_fetch")
    if host_lower.endswith(".local"):
        raise WebFetchEgressViolation(".local hostnames are not allowed for web_fetch")

    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        parsed_ip = None

    if parsed_ip is not None:
        if not parsed_ip.is_global:
            raise WebFetchEgressViolation(
                f"Non-public IP host {host!r} is not allowed for web_fetch"
            )
        return _stream_getaddrinfo_or_raise(host, port)

    infos = _stream_getaddrinfo_or_raise(host, port)
    for *_, sockaddr in infos:
        addr = sockaddr[0]
        try:
            resolved = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if not resolved.is_global:
            raise WebFetchEgressViolation(
                f"Host {host!r} resolves to a non-public address ({resolved})"
            )
    return infos


def enforce_web_fetch_egress(url: str, policy: WebFetchEgressPolicy) -> None:
    """Validate ``url`` (scheme, host, and resolved addresses) for web_fetch."""
    get_validated_stream_addrinfos_for_egress(url, policy)
