"""Request-scoped subscription credentials for the Codex backend."""

from collections.abc import Mapping

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.endpoint_types import HttpEndpoint

from .auth import OpenAIAccess, OpenAIAuthManager, OpenAIReconnectRequired


class CodexEndpointContext:
    """Borrow credentials and recover the token rejected by this request."""

    def __init__(
        self,
        auth: OpenAIAuthManager,
        *,
        base_url: str,
        headers: Mapping[str, str],
        session_id: str | None = None,
    ) -> None:
        self._auth = auth
        self._base_url = base_url
        self._headers = dict(headers)
        if session_id is not None:
            self._headers.update(
                {"session_id": session_id, "Accept": "text/event-stream"}
            )
        self._access: OpenAIAccess | None = None

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        try:
            if force_refresh:
                if self._access is None:
                    raise RuntimeError("Codex recovery requires a rejected credential.")
                access = await self._auth.recover_unauthorized(
                    self._access.access_token
                )
            else:
                access = await self._auth.access()
        except OpenAIReconnectRequired as error:
            raise ExecutionFailure(
                FailureKind.AUTHENTICATION, 401, str(error), False
            ) from error
        self._access = access
        headers = {
            **self._headers,
            "Authorization": f"Bearer {access.access_token}",
            "ChatGPT-Account-ID": access.account_id,
        }
        if access.fedramp:
            headers["X-OpenAI-Fedramp"] = "true"
        return HttpEndpoint(self._base_url, headers, account_id=access.account_id)
