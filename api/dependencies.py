"""Dependency injection for FastAPI."""

import logging
from typing import Optional

from config.settings import (
    Settings,
    get_settings as _get_settings,
    NVIDIA_NIM_BASE_URL,
)
from providers.base import BaseProvider, ProviderConfig

logger = logging.getLogger(__name__)

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
                rate_limit=settings.nvidia_nim_rate_limit,
                rate_window=settings.nvidia_nim_rate_window,
                nim_settings=settings.nim,
            )
            _provider = NvidiaNimProvider(config)
            logger.info("Provider initialized: %s", settings.provider_type)
        elif settings.provider_type == "openrouter":
            from providers.openrouter import OpenRouterProvider

            config = ProviderConfig(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                rate_limit=settings.openrouter_rate_limit,
                rate_window=settings.openrouter_rate_window,
                nim_settings=settings.nim,
            )
            _provider = OpenRouterProvider(config)
            logger.info("Provider initialized: %s", settings.provider_type)
        else:
            logger.error(
                "Unknown provider_type: '%s'. Supported: 'nvidia_nim', 'openrouter'",
                settings.provider_type,
            )
            raise ValueError(
                f"Unknown provider_type: '{settings.provider_type}'. "
                f"Supported: 'nvidia_nim', 'openrouter'"
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
