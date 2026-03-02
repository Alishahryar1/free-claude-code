"""Dependency injection for FastAPI."""

from fastapi import HTTPException
from loguru import logger

from config.settings import Settings
from config.settings import get_settings as _get_settings
from providers.base import BaseProvider, ProviderConfig
from providers.common import get_user_facing_error_message
from providers.exceptions import AuthenticationError
from providers.lmstudio import LMStudioProvider
from providers.nvidia_nim import NVIDIA_NIM_BASE_URL, NvidiaNimProvider
from providers.open_router import OPENROUTER_BASE_URL, OpenRouterProvider

# Provider registry: keyed by provider type string, lazily populated
_providers: dict[str, BaseProvider] = {}
def get_settings() -> Settings:
    """Get application settings via dependency injection."""
    return _get_settings()
def _create_provider_for_type(provider_type: str, settings: Settings) -> BaseProvider:
    """Construct and return a new provider instance for the given provider type."""
    if provider_type == "nvidia_nim":
        if not settings.nvidia_nim_api_key or not settings.nvidia_nim_api_key.strip():
            raise AuthenticationError(
                "NVIDIA_NIM_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://build.nvidia.com/settings/api-keys"
            )
        config = ProviderConfig(
            api_key=settings.nvidia_nim_api_key,
            base_url=NVIDIA_NIM_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return NvidiaNimProvider(config, nim_settings=settings.nim)
    if provider_type == "open_router":
        if not settings.open_router_api_key or not settings.open_router_api_key.strip():
            raise AuthenticationError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://openrouter.ai/keys"
            )
        config = ProviderConfig(
            api_key=settings.open_router_api_key,
            base_url=OPENROUTER_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return OpenRouterProvider(config)
    if provider_type == "lmstudio":
        config = ProviderConfig(
            api_key="lm-studio",
            base_url=settings.lm_studio_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return LMStudioProvider(config)
    logger.error(
        "Unknown provider_type: '{}'. Supported: 'nvidia_nim', 'open_router', 'lmstudio'",
        provider_type,
    )
    raise ValueError(
        f"Unknown provider_type: '{provider_type}'. "
        f"Supported: 'nvidia_nim', 'open_router', 'lmstudio'"
    )
def get_provider_for_type(provider_type: str) -> BaseProvider:
    """Get or create a provider for the given provider type.

    Providers are cached in the registry and reused across requests.
    """
    if provider_type not in _providers:
        try:
            _providers[provider_type] = _create_provider_for_type(
                provider_type, get_settings()
            )
        except AuthenticationError as e:
            raise HTTPException(
                status_code=503, detail=get_user_facing_error_message(e)
            ) from e
        logger.info("Provider initialized: {}", provider_type)
    return _providers[provider_type]
def get_provider() -> BaseProvider:
    """Get or create the default provider (based on MODEL env var).

    Backward-compatible convenience for health/root endpoints and tests.
    """
    return get_provider_for_type(get_settings().provider_type)
async def cleanup_provider():
    """Cleanup all provider resources."""
    global _providers
    for provider in _providers.values():
        await provider.cleanup()
    _providers = {}
    logger.debug("Provider cleanup completed")
