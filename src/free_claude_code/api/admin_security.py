"""Shared local-only security boundary for Admin product surfaces."""

import ipaddress
from urllib.parse import urlsplit

from fastapi import HTTPException, Request


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _origin_is_local(origin: str | None) -> bool:
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
        and _is_loopback_host(parsed.hostname)
    )


def _authority_is_local(authority: str | None) -> bool:
    if not authority:
        return False
    try:
        parsed = urlsplit(f"//{authority}")
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.username is None
        and parsed.password is None
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
        and _is_loopback_host(parsed.hostname)
    )


def _origin_matches_host(origin: str | None, host: str | None) -> bool:
    """Return True when the Origin matches the request Host (same-origin check).

    Absent Origin (direct navigation / non-browser clients) is always allowed.
    """
    if not origin:
        return True
    if not host:
        return False
    try:
        parsed = urlsplit(origin)
        _ = parsed.port
    except ValueError:
        return False
    hostname = parsed.hostname or ""
    origin_authority = f"{hostname}:{parsed.port}" if parsed.port else hostname
    return origin_authority == host.lower().strip()


def _admin_allow_remote(request: Request) -> bool:
    """Read the admin_allow_remote setting from app state if available."""
    try:
        services = request.app.state.services
        return bool(services.requests.current_settings().admin_allow_remote)
    except AttributeError:
        return False


def require_loopback_admin(request: Request) -> None:
    """Allow Admin access only from the local machine.

    When FCC_ADMIN_ALLOW_REMOTE is enabled in settings, remote LAN/WAN
    access is permitted.  The IP/host checks are relaxed, but the Origin
    header is still validated against the request Host to block cross-site
    requests from malicious pages on the same network.

    Note: enabling remote access does **not** add authentication to the Admin
    UI.  Restrict access at the network level (firewall / trusted-IP allowlist)
    or place the service behind an authenticated reverse proxy.
    """
    if _admin_allow_remote(request):
        origin = request.headers.get("origin")
        host = request.headers.get("host")
        if not _origin_matches_host(origin, host):
            raise HTTPException(
                status_code=403,
                detail="Admin UI cross-origin request blocked",
            )
        return

    client_host = request.client.host if request.client else None
    if not _is_loopback_host(client_host):
        raise HTTPException(status_code=403, detail="Admin UI is local-only")
    if not _authority_is_local(request.headers.get("host")):
        raise HTTPException(status_code=403, detail="Admin UI is local-only")
    if not _origin_is_local(request.headers.get("origin")):
        raise HTTPException(status_code=403, detail="Admin UI is local-only")
