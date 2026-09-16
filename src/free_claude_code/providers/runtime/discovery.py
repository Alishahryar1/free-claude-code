"""Provider model-list discovery and background refresh."""

import httpx

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
    ProviderDescriptor,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.model_listing import ModelListResponseError

from .config import has_provider_configuration


def _provider_query_failure_reason(exc: BaseException, settings: Settings) -> str:
    """Return a concise model-list query failure reason for user-facing logs."""
    if isinstance(exc, ModelListResponseError):
        return f"malformed model-list response: {exc.message}"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"query failure: HTTP {exc.response.status_code}"
    if isinstance(exc, ApplicationUnavailableError):
        return f"query failure: {exc.message}"
    if isinstance(exc, ExecutionFailure) and settings.log_api_error_tracebacks:
        return f"query failure: {exc.message}"
    return f"query failure: {type(exc).__name__}"


def referenced_provider_ids(settings: Settings) -> tuple[str, ...]:
    """Return unique provider ids referenced by configured chat models."""
    return tuple(
        dict.fromkeys(ref.provider_id for ref in configured_chat_model_refs(settings))
    )


def model_cache_provider_ids_for_settings(
    settings: Settings,
    connected_provider_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return providers whose model metadata is valid for these settings."""
    configured = tuple(
        provider_id
        for provider_id, descriptor in PROVIDER_CATALOG.items()
        if has_provider_configuration(descriptor, settings)
    )
    available = set(configured) | set(connected_provider_ids)
    return tuple(
        provider_id for provider_id in PROVIDER_CATALOG if provider_id in available
    )


def _self_sufficient_local(descriptor: ProviderDescriptor) -> bool:
    """Return whether a local server can be probed with zero configuration.

    Keyless locals advertising built-in defaults (Ollama, LM Studio,
    llama.cpp) expose their OpenAI-compatible model lists without any user
    settings, so they belong in best-effort discovery (#1296).
    """
    return (
        descriptor.local
        and descriptor.static_credential is not None
        and descriptor.default_base_url is not None
    )


def model_list_provider_ids_for_settings(
    settings: Settings,
    connected_provider_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return providers worth discovering for this process configuration."""
    referenced_ids = referenced_provider_ids(settings)
    included = set(
        model_cache_provider_ids_for_settings(settings, connected_provider_ids)
    )
    for provider_id, descriptor in PROVIDER_CATALOG.items():
        if (
            descriptor.local
            and provider_id not in included
            and (provider_id in referenced_ids or _self_sufficient_local(descriptor))
        ):
            included.add(provider_id)
    return tuple(
        provider_id for provider_id in SUPPORTED_PROVIDER_IDS if provider_id in included
    )
