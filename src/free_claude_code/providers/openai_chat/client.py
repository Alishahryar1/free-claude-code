"""SDK client construction for Chat provider resource owners."""

from collections.abc import Awaitable, Callable, Mapping

import httpx2
from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

from free_claude_code.providers.base import ProviderConfig

OpenAIAsyncCredentialProvider = Callable[[], Awaitable[str]]


def create_chat_client(
    config: ProviderConfig,
    *,
    base_url: str,
    provider_name: str,
    default_headers: Mapping[str, str] | None = None,
    api_key_provider: OpenAIAsyncCredentialProvider | None = None,
) -> AsyncOpenAI:
    """Create a provider-owned SDK client with FCC's existing HTTP policy."""
    if config.api_key is None and api_key_provider is None:
        raise ValueError(f"{provider_name} requires an API key or credential provider")
    timeout = httpx2.Timeout(
        config.http_read_timeout,
        connect=config.http_connect_timeout,
        read=config.http_read_timeout,
        write=config.http_write_timeout,
    )
    http_client = None
    if config.proxy:
        http_client = DefaultAsyncHttpx2Client(proxy=config.proxy, timeout=timeout)
    return AsyncOpenAI(
        api_key=api_key_provider or config.api_key,
        base_url=base_url,
        max_retries=0,
        default_headers=default_headers,
        timeout=timeout,
        http_client=http_client,
    )
