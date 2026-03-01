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

# Global provider instance (singleton)
_provider: BaseProvider | None = None


def get_settings() -> Settings:
    """Get application settings via dependency injection."""
    return _get_settings()


def _create_provider(settings: Settings) -> BaseProvider:
    """Construct and return a new provider instance from settings."""
    if settings.provider_type == "nvidia_nim":
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
        provider = NvidiaNimProvider(config, nim_settings=settings.nim)
    elif settings.provider_type == "open_router":
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
        provider = OpenRouterProvider(config)
    elif settings.provider_type == "lmstudio":
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
        provider = LMStudioProvider(config)
    else:
        logger.error(
            "Unknown provider_type: '{}'. Supported: 'nvidia_nim', 'open_router', 'lmstudio'",
            settings.provider_type,
        )
        raise ValueError(
            f"Unknown provider_type: '{settings.provider_type}'. "
            f"Supported: 'nvidia_nim', 'open_router', 'lmstudio'"
        )
    logger.info("Provider initialized: {}", settings.provider_type)
    return provider


def get_provider() -> BaseProvider:
    """Get or create the provider instance based on settings.provider_type."""
    global _provider
    if _provider is None:
        try:
            _provider = _create_provider(get_settings())
        except AuthenticationError as e:
            raise HTTPException(
                status_code=503, detail=get_user_facing_error_message(e)
            ) from e
    return _provider


async def cleanup_provider():
    """Cleanup provider resources."""
    global _provider
    if _provider:
        await _provider.cleanup()
    _provider = None
    logger.debug("Provider cleanup completed")
