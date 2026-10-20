"""Bind resolved endpoints to isolated OpenAI SDK request clients."""

import httpx
import httpx2
from openai import AsyncOpenAI, Omit

from free_claude_code.providers.endpoint_types import HttpEndpoint


class _BorrowedTransport(httpx2.AsyncBaseTransport):
    """Share connections while the provider generation retains pool ownership."""

    def __init__(self, transport: httpx2.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        return await self._transport.handle_async_request(request)


class OpenAIRequestClient:
    """Own the request HTTP client while borrowing the base SDK client and pool."""

    def __init__(self, transport: httpx2.AsyncBaseTransport | None = None) -> None:
        self._transport = transport
        self._http: httpx2.AsyncClient | None = None
        self._omit_authorization = False

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    def for_endpoint(self, client: AsyncOpenAI, endpoint: HttpEndpoint) -> AsyncOpenAI:
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
