"""Provider model-list discovery and background refresh."""

import re
import socket
import ssl

import httpx
import httpx2

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.model_listing import ModelListResponseError

from .config import has_provider_configuration

_FAILURE_MESSAGES = {
    "proxy": "The network proxy rejected or failed the connection. Check your proxy settings.",
    "dns": "DNS lookup failed. Check your network and provider or proxy address.",
    "certificate": "TLS certificate verification failed. Check your network's certificate configuration.",
    "tls": "The TLS connection failed. Check your network and proxy settings.",
    "timeout": "The model-list request timed out. Check connectivity and retry.",
    "connection": "Could not connect to the provider. Check your network and proxy settings.",
    "response": "The provider returned an invalid model list.",
    "unavailable": "The provider is unavailable. Check its configuration and access.",
    "unknown": "Could not load models. Check the provider's settings and retry.",
}


def safe_discovery_failure_message(value: object) -> str:
    """Allow only classifier output into shareable diagnostics."""
    if isinstance(value, str) and (
        value in _FAILURE_MESSAGES.values()
        or re.fullmatch(r"The model-list request returned HTTP [1-5][0-9]{2}\.", value)
    ):
        return value
    return _FAILURE_MESSAGES["unknown"]


def _provider_query_failure_reason(exc: BaseException) -> str:
    """Classify the transport cause without copying upstream text or URLs."""
    import openai

    from free_claude_code.providers.failure_policy import ProviderRecoveryExhausted

    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if isinstance(current, ProviderRecoveryExhausted):
            current = current.last_error
        else:
            # HTTPCore hides transport context with `raise exc from None`.
            # It still carries the DNS/TLS cause needed for this classification.
            current = current.__cause__ or current.__context__
    # The SDK often wraps a more useful transport or OS error.
    for error in reversed(chain):
        if isinstance(error, (httpx.ProxyError, httpx2.ProxyError)):
            return _FAILURE_MESSAGES["proxy"]
        if isinstance(error, socket.gaierror):
            return _FAILURE_MESSAGES["dns"]
        if isinstance(error, ssl.SSLCertVerificationError):
            return _FAILURE_MESSAGES["certificate"]
        if isinstance(error, ssl.SSLError):
            return _FAILURE_MESSAGES["tls"]
        if isinstance(
            error,
            (
                TimeoutError,
                httpx.TimeoutException,
                httpx2.TimeoutException,
                openai.APITimeoutError,
            ),
        ):
            return _FAILURE_MESSAGES["timeout"]
        if isinstance(
            error,
            (httpx.HTTPStatusError, httpx2.HTTPStatusError, openai.APIStatusError),
        ):
            return safe_discovery_failure_message(
                f"The model-list request returned HTTP {error.response.status_code}."
            )
        if isinstance(error, ModelListResponseError):
            return _FAILURE_MESSAGES["response"]
        if isinstance(
            error,
            (
                ConnectionError,
                httpx.NetworkError,
                httpx2.NetworkError,
                openai.APIConnectionError,
            ),
        ):
            return _FAILURE_MESSAGES["connection"]
        if isinstance(error, ExecutionFailure) and error.kind is FailureKind.TIMEOUT:
            return _FAILURE_MESSAGES["timeout"]
        if isinstance(error, (ApplicationUnavailableError, ExecutionFailure)):
            return _FAILURE_MESSAGES["unavailable"]
    return _FAILURE_MESSAGES["unknown"]


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
        provider_id
        for provider_id in settings.provider_ids
        if provider_id in available or settings.custom_provider(provider_id) is not None
    )


def model_list_provider_ids_for_settings(
    settings: Settings,
    connected_provider_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return providers worth discovering for this process configuration."""
    return model_cache_provider_ids_for_settings(settings, connected_provider_ids)


def is_optional_local_provider(settings: Settings, provider_id: str) -> bool:
    """Identify default local endpoints the user has not explicitly configured."""
    descriptor = PROVIDER_CATALOG.get(provider_id)
    return (
        descriptor is not None
        and descriptor.local
        and provider_id not in referenced_provider_ids(settings)
        # Validation retains explicitly supplied fields, even default-valued URLs.
        and descriptor.base_url_attr not in settings.model_fields_set
        and not (descriptor.proxy_attr and getattr(settings, descriptor.proxy_attr))
    )


def is_optional_local_failure(
    settings: Settings, provider_id: str, message: str | None
) -> bool:
    """An unused default endpoint being absent is normal during discovery."""
    return is_optional_local_provider(settings, provider_id) and message in {
        _FAILURE_MESSAGES["connection"],
        _FAILURE_MESSAGES["timeout"],
    }
