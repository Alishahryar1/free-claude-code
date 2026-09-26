"""SDK client construction for Chat provider resource owners."""

import re
from collections.abc import Awaitable, Callable, Mapping

import httpx2
from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

from free_claude_code.providers.base import ProviderConfig

OpenAIAsyncCredentialProvider = Callable[[], Awaitable[str]]

_LOCALHOST_HOST_RE = re.compile(
    r"^(?P<scheme>https?)://(?P<host>localhost|\.localhost)(?P<port>:\d+)?(?P<path>/|$)"
)


def _normalize_localhost_base_url(base_url: str) -> str:
    """Route localhost base URLs over IPv4 loopback.

    ``localhost`` resolves to ``::1`` first on hosts with IPv6 enabled; when the
    IPv6 loopback NAT is broken (e.g. missing Hyper-V HNS/vmcompute services) the
    connection is reset before any HTTP payload is exchanged and model discovery
    retries with exponential backoff before reporting a generic failure. A local
    endpoint nearly always listens on 127.0.0.1, so normalize to IPv4 explicitly.
    """
    match = _LOCALHOST_HOST_RE.match(base_url)
    if match is None:
        return base_url
    prefix = (
        f"{match.group('scheme')}://127.0.0.1{match.group('port') or ''}{match.group('path')}"
    )
    return prefix + base_url[match.end() :]


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
        base_url=_normalize_localhost_base_url(base_url),
        max_retries=0,
        default_headers=default_headers,
        timeout=timeout,
        http_client=http_client,
    )
