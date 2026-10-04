"""Egress policy for user-controlled web_fetch URLs (SSRF guard)."""

import ipaddress
import socket
import unicodedata
from dataclasses import dataclass
from urllib.parse import ParseResult, urlparse

import idna


@dataclass(frozen=True, slots=True)
class WebFetchEgressPolicy:
    """Egress rules for user-influenced web_fetch URLs."""

    allow_private_network_targets: bool
    allowed_schemes: frozenset[str]
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for option in ("allowed_domains", "blocked_domains"):
            object.__setattr__(
                self,
                option,
                tuple(normalize_web_domain(raw) for raw in getattr(self, option)),
            )


class WebFetchEgressViolation(ValueError):
    """Raised when a web_fetch URL is rejected by egress policy (SSRF guard)."""


def web_fetch_allowed_scheme_set(raw_schemes: str) -> frozenset[str]:
    """Return normalized schemes allowed for web_fetch."""

    return frozenset(
        part.strip().lower() for part in raw_schemes.split(",") if part.strip()
    )


def normalize_web_domain(raw: str) -> str:
    """Validate a plain domain option and return its lowercase IDNA hostname.

    Schemes, ports, paths, credentials and wildcard patterns are not domains.
    A single trailing DNS root dot is accepted and removed.
    """
    if not raw or any(
        char.isspace()
        or unicodedata.category(char).startswith("C")
        or char in ":/\\@*?#%[]"
        for char in raw
    ):
        raise ValueError("Web domain options must be plain hostnames")
    try:
        normalized = idna.uts46_remap(raw, std3_rules=True).removesuffix(".")
        if normalized.endswith("."):
            raise ValueError("Web domain options must not contain empty labels")
        return idna.encode(normalized, std3_rules=True).decode("ascii").lower()
    except idna.IDNAError as exc:
        raise ValueError("Web domain options must be valid hostnames") from exc


def _normalize_web_host(host: str) -> str:
    """Normalize DNS hosts and preserve literal IPv6 support for existing fetches."""
    if ":" in host:
        if "%" in host:
            raise ValueError("IPv6 zone identifiers are not allowed")
        return str(ipaddress.IPv6Address(host))
    return normalize_web_domain(host)


def _normalized_domain_matches(host: str, domain: str) -> bool:
    if host == domain:
        return True
    for name in (host, domain):
        try:
            ipaddress.ip_address(name)
        except ValueError:
            continue
        return False
    return host.endswith(f".{domain}")


def domain_matches(host: str, domain: str) -> bool:
    """Match an exact domain or its subdomain, never a lookalike suffix."""
    try:
        host = _normalize_web_host(host)
        domain = normalize_web_domain(domain)
    except ValueError:
        return False
    return _normalized_domain_matches(host, domain)


def _host_allowed_by_domains(
    host: str, allowed_domains: tuple[str, ...], blocked_domains: tuple[str, ...]
) -> bool:
    return not any(
        _normalized_domain_matches(host, domain) for domain in blocked_domains
    ) and (
        not allowed_domains
        or any(_normalized_domain_matches(host, domain) for domain in allowed_domains)
    )


def _parse_web_url(url: str) -> tuple[ParseResult, str, int]:
    # urlparse silently strips some controls; reject them before parsing.
    if any(ord(char) <= 32 or ord(char) == 127 or char == "\\" for char in url):
        raise WebFetchEgressViolation("web_fetch URL contains invalid characters")
    try:
        parsed = urlparse(url)
        if parsed.username is not None or parsed.password is not None:
            raise WebFetchEgressViolation("Credentials are not allowed in web URLs")
        if not parsed.hostname:
            raise WebFetchEgressViolation("web_fetch URL must include a host")
        host = _normalize_web_host(parsed.hostname)
        port = parsed.port
        if parsed.netloc.endswith(":") or port == 0:
            raise WebFetchEgressViolation("web_fetch URL must include a valid port")
    except ValueError as exc:
        raise WebFetchEgressViolation("web_fetch URL is invalid") from exc
    return (
        parsed,
        host,
        port if port is not None else (443 if parsed.scheme == "https" else 80),
    )


def web_url_allowed_by_domains(
    url: str,
    allowed_domains: tuple[str, ...] = (),
    blocked_domains: tuple[str, ...] = (),
) -> bool:
    """Filter HTTP(S) result URLs by domain without making a DNS request.

    Malformed URLs or domain options fail closed; blocking takes precedence.
    Fetches must still apply their separate resolved-address egress checks.
    """
    try:
        parsed, host, _ = _parse_web_url(url)
        allowed_domains = tuple(normalize_web_domain(raw) for raw in allowed_domains)
        blocked_domains = tuple(normalize_web_domain(raw) for raw in blocked_domains)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and _host_allowed_by_domains(
        host, allowed_domains, blocked_domains
    )


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
    connect (used by :func:`api.web_tools.outbound._run_web_fetch`).
    """
    parsed, host, port = _parse_web_url(url)
    scheme = parsed.scheme.lower()
    if scheme not in policy.allowed_schemes:
        raise WebFetchEgressViolation(
            f"URL scheme {scheme!r} is not allowed for web_fetch"
        )

    if not _host_allowed_by_domains(
        host, policy.allowed_domains, policy.blocked_domains
    ):
        raise WebFetchEgressViolation(
            f"Host {host!r} is not allowed by web_fetch domain constraints"
        )

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
