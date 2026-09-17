"""CodeBuddy OAuth device authorization flow.

Ground truth: the reverse-engineered protocol used by the official CLI,
implemented with plain HTTP calls against ``/v2/plugin/auth/...``. One
httpx client is shared across the flow so upstream-set cookies persist
between the state, token, and account steps.
"""

import time
from dataclasses import dataclass
from typing import Any

import httpx2

from .site import CodeBuddySite, default_site

DEVICE_POLL_INTERVAL_SECONDS = 3.0
LOGIN_LIFETIME_SECONDS = 15 * 60


class CodeBuddyLoginError(RuntimeError):
    """An interactive authorization flow could not complete."""


@dataclass(frozen=True, slots=True, repr=False)
class DeviceAuthorization:
    """State and browser URL shown to the customer for one device login."""

    state: str
    verification_url: str
    expires_at: int


@dataclass(frozen=True, slots=True, repr=False)
class DeviceTokens:
    """Token material issued for one authorized device state."""

    access_token: str
    refresh_token: str
    expires_at: int | None
    domain: str | None


@dataclass(frozen=True, slots=True, repr=False)
class DeviceAccountProfile:
    """Identity claims used to build the upstream account headers."""

    uid: str | None
    enterprise_id: str | None
    nickname: str | None


def _headers(site: CodeBuddySite) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": site.origin,
        "Referer": f"{site.origin}/",
        "User-Agent": site.user_agent,
    }


def _decode_envelope(payload: Any, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("code") != 0:
        raise CodeBuddyLoginError(f"CodeBuddy {context} rejected the request.")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise CodeBuddyLoginError(f"CodeBuddy {context} returned no data.")
    return data


async def start_device_authorization(
    client: httpx2.AsyncClient,
    site: CodeBuddySite | None = None,
) -> DeviceAuthorization:
    """Request the state and browser URL for one device login."""

    site = site or default_site()
    response = await client.post(
        f"{site.api_base}/v2/plugin/auth/state?platform=CLI",
        json={},
        headers=_headers(site),
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise CodeBuddyLoginError(
            f"CodeBuddy authorization state response was not JSON (HTTP {response.status_code})."
        ) from exc
    if response.status_code >= 400:
        raise CodeBuddyLoginError(
            f"CodeBuddy authorization state failed (HTTP {response.status_code})."
        )
    data = _decode_envelope(payload, "authorization state")
    state = data.get("state")
    auth_url = data.get("authUrl")
    if not isinstance(state, str) or not state:
        raise CodeBuddyLoginError("CodeBuddy returned an empty authorization state.")
    if not isinstance(auth_url, str) or not auth_url:
        raise CodeBuddyLoginError("CodeBuddy returned no authorization URL.")
    return DeviceAuthorization(
        state=state,
        verification_url=auth_url,
        expires_at=int(time.time()) + LOGIN_LIFETIME_SECONDS,
    )


def _is_terminal_poll_status(status_code: int) -> bool:
    """Return whether one device-poll status is a terminal client rejection."""

    return 400 <= status_code < 500


def _poll_rejection(status_code: int, payload: Any) -> str:
    """Build the customer-facing reason one device poll was rejected."""

    detail = None
    if isinstance(payload, dict):
        for key in ("msg", "message", "error"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                detail = value.strip()
                break
    if detail is None:
        return f"CodeBuddy sign-in was rejected (HTTP {status_code})."
    return f"CodeBuddy sign-in was rejected: {detail}"


async def poll_device_tokens(
    client: httpx2.AsyncClient,
    authorization: DeviceAuthorization,
    site: CodeBuddySite | None = None,
) -> DeviceTokens | None:
    """Poll once; return None while the customer has not finished in the browser.

    Pending is the upstream's ``code`` 0 envelope without token material.
    A client error or a nonzero envelope is terminal (expired or unknown
    device state) and raises so the flow stops instead of waiting out the
    authorization deadline. Server errors and unreadable bodies stay
    pending, because the browser step can still complete.
    """

    site = site or default_site()
    response = await client.get(
        f"{site.api_base}/v2/plugin/auth/token",
        params={"state": authorization.state},
        headers=_headers(site),
    )
    try:
        payload = response.json()
    except ValueError:
        if _is_terminal_poll_status(response.status_code):
            raise CodeBuddyLoginError(
                f"CodeBuddy sign-in was rejected (HTTP {response.status_code})."
            ) from None
        return None
    if _is_terminal_poll_status(response.status_code):
        raise CodeBuddyLoginError(_poll_rejection(response.status_code, payload))
    if response.status_code >= 400 or not isinstance(payload, dict):
        return None
    if payload.get("code") != 0:
        raise CodeBuddyLoginError(_poll_rejection(response.status_code, payload))
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    access_token = data.get("accessToken")
    if not isinstance(access_token, str) or not access_token:
        return None
    expires_in = data.get("expiresIn")
    expires_at = (
        int(time.time()) + int(expires_in)
        if isinstance(expires_in, int | float) and not isinstance(expires_in, bool)
        else None
    )
    refresh_token = data.get("refreshToken")
    domain = data.get("domain")
    return DeviceTokens(
        access_token=access_token,
        refresh_token=refresh_token if isinstance(refresh_token, str) else "",
        expires_at=expires_at,
        domain=domain if isinstance(domain, str) else None,
    )


async def fetch_account_profile(
    client: httpx2.AsyncClient,
    authorization: DeviceAuthorization,
    tokens: DeviceTokens,
    site: CodeBuddySite | None = None,
) -> DeviceAccountProfile | None:
    """Best-effort account identity; absence must not fail the login."""

    site = site or default_site()
    headers = _headers(site)
    headers["Authorization"] = f"Bearer {tokens.access_token}"
    try:
        response = await client.get(
            f"{site.api_base}/v2/plugin/login/account",
            params={"state": authorization.state},
            headers=headers,
        )
        payload = response.json()
    except httpx2.HTTPError, ValueError:
        return None
    if response.status_code >= 400 or not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    uid = data.get("uid")
    enterprise_id = data.get("enterpriseId")
    nickname = data.get("nickname")
    return DeviceAccountProfile(
        uid=str(uid) if uid is not None else None,
        enterprise_id=str(enterprise_id) if enterprise_id is not None else None,
        nickname=nickname if isinstance(nickname, str) else None,
    )
