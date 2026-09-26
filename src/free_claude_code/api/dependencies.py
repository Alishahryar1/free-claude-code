"""FastAPI dependencies for the explicit runtime service boundary."""

import secrets

from fastapi import Depends, HTTPException, Request
from loguru import logger

from free_claude_code.application.errors import UnknownProviderError
from free_claude_code.application.ports import ProviderPort, RequestRuntimeLease
from free_claude_code.config.settings import Settings

from .ports import ApiServices


def get_services(request: Request) -> ApiServices:
    """Return the complete services supplied when the app was constructed."""
    return request.app.state.services


def get_settings(services: ApiServices = Depends(get_services)) -> Settings:
    """Return the current request-runtime settings snapshot."""
    return services.requests.current_settings()


async def resolve_provider(
    provider_type: str,
    *,
    lease: RequestRuntimeLease,
) -> ProviderPort:
    """Resolve a provider through one retained generation."""
    should_log_init = not lease.is_provider_cached(provider_type)
    try:
        provider = await lease.resolve_provider(provider_type)
    except UnknownProviderError:
        logger.error(
            "Unknown provider_type: '{}'. Supported: {}",
            provider_type,
            ", ".join(f"'{key}'" for key in lease.settings.provider_ids),
        )
        raise
    if should_log_init:
        logger.info("Provider initialized: {}", provider_type)
    return provider


def require_proxy_auth(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Require the configured proxy token as HTTP bearer authorization."""
    if not settings.proxy_auth_enabled:
        return

    authorization = request.headers.get("authorization")
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing proxy authentication token",
        )

    if not _proxy_token_matches(
        authorization,
        settings.proxy_auth_token,
        require_bearer=True,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid proxy authentication token",
        )


def require_anthropic_proxy_auth(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Require Bearer or Anthropic ``x-api-key`` proxy authentication."""
    if not settings.proxy_auth_enabled:
        return

    authorization = request.headers.get("authorization")
    if authorization is not None:
        if _proxy_token_matches(
            authorization,
            settings.proxy_auth_token,
            require_bearer=True,
        ):
            return
        raise HTTPException(
            status_code=401,
            detail="Invalid proxy authentication token",
        )

    x_api_key = request.headers.get("x-api-key")
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing proxy authentication token",
        )

    if not _proxy_token_matches(
        x_api_key,
        settings.proxy_auth_token,
        require_bearer=False,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid proxy authentication token",
        )


def _proxy_token_matches(
    credential: str,
    configured_token: str,
    *,
    require_bearer: bool,
) -> bool:
    token = credential.strip()
    if require_bearer:
        parts = token.split(maxsplit=1)
        if len(parts) != 2 or parts[0].casefold() != "bearer":
            return False
        token = parts[1].strip()

    return bool(token) and secrets.compare_digest(
        token.encode("utf-8"),
        configured_token.encode("utf-8"),
    )
