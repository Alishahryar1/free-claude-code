"""Dependency injection for FastAPI."""

from fastapi import Depends, HTTPException, Request
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


async def verify_auth_token(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """Verify ANTHROPIC_AUTH_TOKEN from request headers if configured.

    - If settings.anthropic_auth_token is empty, authentication is disabled (backward compatible)
    - If configured, clients must provide matching ANTHROPIC_AUTH_TOKEN header
    """
    if not settings.anthropic_auth_token:
        # Auth not configured, allow all requests
        return

    # Get token from headers - support both anthropic-auth-token and authorization (Bearer) formats
    auth_header = request.headers.get("anthropic-auth-token")
    auth_bearer = request.headers.get("authorization")

    # Extract token from Bearer format if present
    token_from_bearer = None
    if auth_bearer and auth_bearer.lower().startswith("bearer "):
        token_from_bearer = auth_bearer[7:]  # Remove "Bearer " prefix

    # Use the token from either header
    received_token = auth_header or token_from_bearer

    # Debug logging: show configured token and received token (masked for security)
    configured_token_masked = settings.anthropic_auth_token[:4] + "..." + settings.anthropic_auth_token[-4:] if len(settings.anthropic_auth_token) > 8 else "***"
    received_masked = "<missing>"
    if received_token:
        received_masked = received_token[:4] + "..." + received_token[-4:] if len(received_token) > 8 else "***"

    logger.debug(
        "Auth check - Configured token: {}, Received token: {}, Source header: {}",
        configured_token_masked,
        received_masked,
        "anthropic-auth-token" if auth_header else ("authorization" if token_from_bearer else "none")
    )

    if not received_token:
        # Dump all request headers for debugging
        logger.warning("Missing ANTHROPIC_AUTH_TOKEN. Expected: {}", configured_token_masked)
        logger.warning("=== FULL REQUEST HEADERS DUMP ===")
        for header_name, header_value in request.headers.items():
            logger.warning("Header: {} = {}", header_name, header_value)
        logger.warning("=== END HEADERS DUMP ===")
        raise HTTPException(
            status_code=401,
            detail={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "Missing ANTHROPIC_AUTH_TOKEN header"
                }
            }
        )

    if received_token != settings.anthropic_auth_token:
        logger.warning(
            "ANTHROPIC_AUTH_TOKEN mismatch. Expected: {}, Got: {}",
            configured_token_masked,
            received_masked
        )
        # Dump all request headers for debugging
        logger.warning("=== FULL REQUEST HEADERS DUMP ===")
        for header_name, header_value in request.headers.items():
            logger.warning("Header: {} = {}", header_name, header_value)
        logger.warning("=== END HEADERS DUMP ===")
        raise HTTPException(
            status_code=401,
            detail={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid ANTHROPIC_AUTH_TOKEN"
                }
            }
        )

    logger.debug("ANTHROPIC_AUTH_TOKEN authentication successful")
