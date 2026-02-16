"""Dependency injection for FastAPI."""

from typing import Optional

from loguru import logger

from config.settings import Settings, get_settings as _get_settings, NVIDIA_NIM_BASE_URL
from providers.base import BaseProvider, ProviderConfig


# Global provider instance (singleton)
_provider: Optional[BaseProvider] = None


def get_settings() -> Settings:
    """Get application settings via dependency injection."""
    return _get_settings()


def get_provider() -> BaseProvider:
    """Get or create the provider instance based on settings.provider_type."""
    global _provider
    if _provider is None:
        settings = get_settings()

        if settings.provider_type == "nvidia_nim":
            from providers.nvidia_nim import NvidiaNimProvider

            config = ProviderConfig(
                api_key=settings.nvidia_nim_api_key,
                base_url=NVIDIA_NIM_BASE_URL,
                rate_limit=settings.provider_rate_limit,
                rate_window=settings.provider_rate_window,
                nim_settings=settings.nim,
                http_read_timeout=settings.http_read_timeout,
                http_write_timeout=settings.http_write_timeout,
                http_connect_timeout=settings.http_connect_timeout,
            )
            _provider = NvidiaNimProvider(config)
            logger.info("Provider initialized: %s", settings.provider_type)
        elif settings.provider_type == "open_router":
            from providers.open_router import OpenRouterProvider

            config = ProviderConfig(
                api_key=settings.open_router_api_key,
                base_url="https://openrouter.ai/api/v1",
                rate_limit=settings.provider_rate_limit,
                rate_window=settings.provider_rate_window,
                nim_settings=settings.nim,
                http_read_timeout=settings.http_read_timeout,
                http_write_timeout=settings.http_write_timeout,
                http_connect_timeout=settings.http_connect_timeout,
            )
            _provider = OpenRouterProvider(config)
            logger.info("Provider initialized: %s", settings.provider_type)
        elif settings.provider_type == "lmstudio":
            from providers.lmstudio import LMStudioProvider

            config = ProviderConfig(
                api_key="lm-studio",
                base_url=settings.lm_studio_base_url,
                rate_limit=settings.provider_rate_limit,
                rate_window=settings.provider_rate_window,
                nim_settings=settings.nim,
                http_read_timeout=settings.http_read_timeout,
                http_write_timeout=settings.http_write_timeout,
                http_connect_timeout=settings.http_connect_timeout,
            )
            _provider = LMStudioProvider(config)
            logger.info("Provider initialized: %s", settings.provider_type)
        else:
            logger.error(
                "Unknown provider_type: '%s'. Supported: 'nvidia_nim', 'open_router', 'lmstudio'",
                settings.provider_type,
            )
            raise ValueError(
                f"Unknown provider_type: '{settings.provider_type}'. "
                f"Supported: 'nvidia_nim', 'open_router', 'lmstudio'"
            )
    return _provider


async def cleanup_provider():
    """Cleanup provider resources."""
    global _provider
    if _provider:
        client = getattr(_provider, "_client", None)
        if client and hasattr(client, "aclose"):
            await client.aclose()
    _provider = None
    logger.debug("Provider cleanup completed")
