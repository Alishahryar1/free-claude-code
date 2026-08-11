"""SDK client construction for Chat provider resource owners."""

from collections.abc import Awaitable, Callable, Mapping

import httpx2
from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

from free_claude_code.providers.base import ProviderConfig

OpenAIAsyncCredentialProvider = Callable[[], Awaitable[str]]


async def _drop_authorization_header(request: httpx2.Request) -> None:
    """Strip the SDK's placeholder bearer token from keyless requests."""
    request.headers.pop("Authorization", None)


def create_chat_client(
    config: ProviderConfig,
    *,
    base_url: str,
    provider_name: str,
    default_headers: Mapping[str, str] | None = None,
    api_key_provider: OpenAIAsyncCredentialProvider | None = None,
    credential_optional: bool = False,
) -> AsyncOpenAI:
    """Create a provider-owned SDK client with FCC's existing HTTP policy."""
    # Keyless endpoints (credential_optional profiles such as `custom`) must
    # not send any Authorization header. The OpenAI SDK requires a non-empty
    # api_key and would otherwise emit a placeholder bearer token on the
    # wire, so strip the header right before each request.
    keyless = api_key_provider is None and not config.api_key
    if keyless and not credential_optional:
        raise ValueError(f"{provider_name} requires an API key or credential provider")
    timeout = httpx2.Timeout(
        config.http_read_timeout,
        connect=config.http_connect_timeout,
        read=config.http_read_timeout,
        write=config.http_write_timeout,
    )
    http_client = None
    if config.proxy or keyless:
        http_client = DefaultAsyncHttpx2Client(
            proxy=config.proxy,
            timeout=timeout,
            event_hooks={"request": [_drop_authorization_header]} if keyless else None,
        )
    return AsyncOpenAI(
        # The SDK rejects a missing api_key outright; keyless profiles pass
        # a placeholder that _drop_authorization_header removes.
        api_key=api_key_provider or config.api_key or "no-api-key",
        base_url=base_url,
        max_retries=0,
        default_headers=default_headers,
        timeout=timeout,
        http_client=http_client,
    )
