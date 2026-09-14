"""Request-scoped endpoint snapshots borrowed by provider transports."""

import httpx
import httpx2
from openai import AsyncOpenAI, Omit

from free_claude_code.providers.admission import (
    ProviderAttempt,
    ProviderCorrectionAction,
    ProviderExecution,
)
from free_claude_code.providers.endpoint_types import EndpointContext, HttpEndpoint
from free_claude_code.providers.failure_policy import provider_authentication_status


class _BorrowedTransport(httpx2.AsyncBaseTransport):
    """Share connections while the provider generation retains pool ownership."""

    def __init__(self, transport: httpx2.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        return await self._transport.handle_async_request(request)


class RequestEndpoint:
    """Resolve one request's credentials and permit one refresh before commitment."""

    def __init__(
        self,
        context: EndpointContext,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._context = context
        self._transport = transport
        self._http: httpx2.AsyncClient | None = None
        self._refreshed = False
        self._refresh_pending = False
        self._committed = False
        self._omit_authorization = False
        self.snapshot: HttpEndpoint | None = None

    def commit(self) -> None:
        self._committed = True

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    async def openai_client(self, client: AsyncOpenAI) -> AsyncOpenAI:
        endpoint = await self._context.endpoint(force_refresh=self._refresh_pending)
        self.snapshot = endpoint
        self._refresh_pending = False
        if self._http is None:
            self._http = httpx2.AsyncClient(
                transport=_BorrowedTransport(self._transport)
                if self._transport is not None
                else None,
                follow_redirects=False,
            )
        # An endpoint is authoritative for every attempt, including after refresh.
        self._http.cookies.clear()
        headers = dict(httpx.Headers(endpoint.headers).items())
        self._omit_authorization = (
            endpoint.api_key is None and "authorization" not in headers
        )
        # SDK defaults are merged as a case-sensitive mapping before HTTP parsing.
        # Match their spelling so an endpoint replaces, rather than duplicates, them.
        sdk_header_names = {"authorization": "Authorization"}
        for name in client.default_headers:
            sdk_header_names.setdefault(name.lower(), name)
        headers = {
            sdk_header_names.get(name, name): value for name, value in headers.items()
        }

        async def credential() -> str:
            # A callable also overrides an inherited key with an empty credential.
            return endpoint.api_key or ""

        view = client.with_options(
            api_key=credential,
            base_url=endpoint.base_url,
            set_default_headers=headers,
            set_default_query={},
            http_client=self._http,
            max_retries=0,
        )
        # SDK copy(None) inherits these values. Clear only this owned view.
        view.organization = None
        view.project = None
        view.admin_api_key = None
        return view

    def openai_headers(self) -> dict[str, str | Omit]:
        return {"Authorization": Omit()} if self._omit_authorization else {}

    async def retry_authentication(
        self, error: Exception, attempt: ProviderAttempt, execution: ProviderExecution
    ) -> bool:
        if self._refreshed or self._committed:
            return False
        status = provider_authentication_status(error)
        if status not in {401, 403}:
            return False
        allowed = (
            execution.can_attempt
            if attempt.accepted
            else await attempt.correct(error) is ProviderCorrectionAction.RETRY
        )
        if allowed:
            self._refreshed = self._refresh_pending = True
        return allowed
