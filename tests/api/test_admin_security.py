"""Unit tests for admin_security.py — loopback guard and DNS-rebinding protection."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from free_claude_code.api.admin_security import (
    _host_is_ip_address,
    _origin_is_local,
    _origin_matches_host,
    require_loopback_admin,
)


# ---------------------------------------------------------------------------
# _host_is_ip_address
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "127.0.0.1:8082",
        "10.0.0.5",
        "10.0.0.5:8082",
        "192.168.1.100",
        "192.168.1.100:8082",
        "0.0.0.0",
        "[::1]",
        "[::1]:8082",
        "::1",
        "localhost",
        "localhost:8082",
    ],
)
def test_host_is_ip_address_accepts_ip_and_localhost(host):
    assert _host_is_ip_address(host) is True


@pytest.mark.parametrize(
    "host",
    [
        "evil.com",
        "evil.com:8082",
        "attacker.com",
        "fcc-server.local",
        "fcc-server.local:8082",
        "my-nas",
        "my-nas:8082",
        "",
        None,
    ],
)
def test_host_is_ip_address_rejects_hostname(host):
    assert _host_is_ip_address(host) is False


# ---------------------------------------------------------------------------
# Helpers for require_loopback_admin tests
# ---------------------------------------------------------------------------


def _loopback_request(*, origin: str | None = None) -> MagicMock:
    """Simulate a loopback request (admin_allow_remote=False)."""
    request = MagicMock()
    request.client.host = "127.0.0.1"
    headers = {"host": "127.0.0.1:8082"}
    if origin is not None:
        headers["origin"] = origin
    request.headers.get = lambda key, default=None: headers.get(key, default)
    settings = MagicMock()
    settings.admin_allow_remote = False
    request.app.state.services.requests.current_settings.return_value = settings
    return request


def _remote_request(
    *, host: str, origin: str | None = None, client_ip: str = "10.0.0.100"
) -> MagicMock:
    """Simulate a LAN request (admin_allow_remote=True)."""
    request = MagicMock()
    request.client.host = client_ip
    headers = {"host": host}
    if origin is not None:
        headers["origin"] = origin
    request.headers.get = lambda key, default=None: headers.get(key, default)
    settings = MagicMock()
    settings.admin_allow_remote = True
    request.app.state.services.requests.current_settings.return_value = settings
    return request


# ---------------------------------------------------------------------------
# Loopback mode (admin_allow_remote=False) — existing behaviour unchanged
# ---------------------------------------------------------------------------


def test_loopback_mode_allows_local_client():
    require_loopback_admin(_loopback_request())


def test_loopback_mode_allows_local_client_with_matching_loopback_origin():
    require_loopback_admin(_loopback_request(origin="http://127.0.0.1:8082"))


def test_loopback_mode_blocks_remote_client():
    request = _remote_request(host="10.0.0.5:8082")
    # admin_allow_remote is True in _remote_request; override to False
    request.app.state.services.requests.current_settings.return_value.admin_allow_remote = (
        False
    )
    request.client.host = "10.0.0.100"
    with pytest.raises(HTTPException) as exc:
        require_loopback_admin(request)
    assert exc.value.status_code == 403


def test_loopback_mode_blocks_external_web_origin():
    with pytest.raises(HTTPException) as exc:
        require_loopback_admin(_loopback_request(origin="https://example.com"))
    assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# Remote mode (admin_allow_remote=True) — DNS-rebinding protection
# ---------------------------------------------------------------------------


def test_remote_mode_allows_ip_host_without_origin():
    """Direct navigation / curl with no Origin header."""
    require_loopback_admin(_remote_request(host="10.0.0.5:8082"))


def test_remote_mode_allows_ip_host_with_matching_origin():
    """Browser accessing Admin UI via LAN IP — same-origin."""
    require_loopback_admin(
        _remote_request(host="10.0.0.5:8082", origin="http://10.0.0.5:8082")
    )


def test_remote_mode_allows_localhost_host_without_origin():
    require_loopback_admin(_remote_request(host="localhost:8082"))


def test_remote_mode_allows_loopback_ip_host():
    require_loopback_admin(_remote_request(host="127.0.0.1:8082"))


def test_remote_mode_blocks_dns_rebind_hostname_with_matching_origin():
    """Core DNS-rebinding scenario: attacker.com resolves to FCC's IP.

    Browser sends Origin: http://attacker.com and Host: attacker.com — both
    match, but attacker.com is not an IP address so the request is rejected.
    """
    with pytest.raises(HTTPException) as exc:
        require_loopback_admin(
            _remote_request(
                host="attacker.com:8082",
                origin="http://attacker.com:8082",
            )
        )
    assert exc.value.status_code == 403
    assert "DNS-rebinding" in exc.value.detail


def test_remote_mode_blocks_hostname_host_without_origin():
    """Hostname in Host with no Origin is still rejected in remote mode."""
    with pytest.raises(HTTPException) as exc:
        require_loopback_admin(_remote_request(host="fcc-server.local:8082"))
    assert exc.value.status_code == 403
    assert "DNS-rebinding" in exc.value.detail


def test_remote_mode_blocks_cross_origin_csrf():
    """Regular CSRF: different-origin request to Admin API."""
    with pytest.raises(HTTPException) as exc:
        require_loopback_admin(
            _remote_request(host="10.0.0.5:8082", origin="http://evil.com")
        )
    assert exc.value.status_code == 403


def test_remote_mode_blocks_cross_origin_csrf_with_port():
    with pytest.raises(HTTPException) as exc:
        require_loopback_admin(
            _remote_request(host="10.0.0.5:8082", origin="http://evil.com:9000")
        )
    assert exc.value.status_code == 403


def test_remote_mode_ipv6_ip_host_allowed():
    """IPv6 LAN address in Host is accepted."""
    require_loopback_admin(_remote_request(host="[fd00::1]:8082"))


def test_remote_mode_ipv6_dns_rebind_blocked():
    """IPv6 DNS rebinding attempt via hostname is blocked."""
    with pytest.raises(HTTPException) as exc:
        require_loopback_admin(
            _remote_request(
                host="attacker.com:8082",
                origin="http://attacker.com:8082",
                client_ip="fd00::1",
            )
        )
    assert exc.value.status_code == 403
