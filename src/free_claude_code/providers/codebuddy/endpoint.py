"""Request-scoped CodeBuddy endpoint snapshots and upstream headers."""

import secrets
from collections.abc import Mapping
from uuid import uuid4

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.endpoint_types import HttpEndpoint

from .auth import CodeBuddyAccess, CodeBuddyAuthManager, CodeBuddyReconnectRequired
from .site import CodeBuddySite, default_site


def common_headers(site: CodeBuddySite) -> dict[str, str]:
    """Return headers shared by every upstream request."""

    return {
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": site.origin,
        "Referer": f"{site.origin}/",
        "User-Agent": site.user_agent,
    }


def chat_headers(site: CodeBuddySite, access: CodeBuddyAccess) -> dict[str, str]:
    """Return the chat request headers, including the account identity."""

    headers = common_headers(site)
    headers["Accept"] = "text/event-stream"
    headers["Authorization"] = f"Bearer {access.access_token}"
    headers["X-Request-ID"] = secrets.token_hex(16)
    headers["X-Request-Trace-Id"] = str(uuid4())
    headers["X-Product"] = site.product
    if access.uid:
        headers["X-User-Id"] = access.uid
    else:
        headers["X-No-User-Id"] = "1"
    if access.enterprise_id:
        headers["X-Enterprise-Id"] = access.enterprise_id
    else:
        headers["X-No-Enterprise-Id"] = "1"
    if access.domain:
        headers["X-Domain"] = access.domain
    else:
        headers["X-No-Department-Info"] = "1"
    return headers


class CodeBuddyEndpointContext:
    """Borrow credentials and recover the token rejected by this request."""

    def __init__(
        self,
        auth: CodeBuddyAuthManager,
        *,
        site: CodeBuddySite | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._auth = auth
        self._site = site or default_site()
        self._headers = dict(headers or {})
        self._access: CodeBuddyAccess | None = None

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        try:
            if force_refresh:
                if self._access is None:
                    raise RuntimeError(
                        "CodeBuddy recovery requires a rejected credential."
                    )
                access = await self._auth.recover_unauthorized(
                    self._access.access_token
                )
            else:
                access = await self._auth.access()
        except CodeBuddyReconnectRequired as error:
            raise ExecutionFailure(
                FailureKind.AUTHENTICATION, 401, str(error), False
            ) from error
        self._access = access
        return HttpEndpoint(
            self._site.chat_base_url,
            {**chat_headers(self._site, access), **self._headers},
            account_id=access.uid,
        )
