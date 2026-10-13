"""Provider configuration construction from neutral catalog metadata."""

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config.api_keys import selected_api_key
from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.config.provider_catalog import ProviderDescriptor
from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import ProviderConfig


def string_setting(settings: Settings, attr_name: str | None) -> str | None:
    """Return an optional validated string setting."""
    if attr_name is None:
        return None
    value = getattr(settings, attr_name, None)
    return value if isinstance(value, str) else None


def provider_credential(
    descriptor: ProviderDescriptor, settings: Settings, api_key_number: int = 1
) -> str | None:
    """Return the configured credential for a provider descriptor."""
    if descriptor.credential_attr:
        credential = selected_api_key(
            getattr(settings, descriptor.credential_attr), api_key_number
        )
        if api_key_number != 1 and credential is None:
            raise ApplicationUnavailableError(
                f"{descriptor.display_name} Key {api_key_number} is not configured. "
                "Update the fallback key selection in the Admin UI."
            )
        return credential
    if api_key_number != 1:
        raise ApplicationUnavailableError(
            f"{descriptor.display_name} has no selectable API key list."
        )
    return descriptor.static_credential


def has_provider_configuration(
    descriptor: ProviderDescriptor, settings: Settings
) -> bool:
    """Return whether all provider-defining settings are present."""
    attrs = descriptor.configuration_attrs()
    if attrs:
        return all(
            provider_credential(descriptor, settings)
            if attr == descriptor.credential_attr
            else string_setting(settings, attr)
            for attr in attrs
        )
    return descriptor.static_credential is not None


def require_provider_credential(
    descriptor: ProviderDescriptor, credential: str | None
) -> None:
    """Raise a user-facing configuration error when a required key is missing."""
    if descriptor.credential_env is None:
        return
    if credential:
        return
    message = f"{descriptor.credential_env} is not set. Add it in the Admin UI."
    if descriptor.credential_url:
        message = f"{message} Get a key at {descriptor.credential_url}"
    raise ApplicationUnavailableError(message)


def build_provider_config(
    descriptor: ProviderDescriptor, settings: Settings, *, api_key_number: int = 1
) -> ProviderConfig:
    """Build shared provider configuration for one provider descriptor."""
    credential = provider_credential(descriptor, settings, api_key_number)
    require_provider_credential(descriptor, credential)
    base_url = string_setting(settings, descriptor.base_url_attr)
    resolved_base_url = base_url or descriptor.default_base_url
    if not resolved_base_url:
        if descriptor.base_url_attr is None:
            raise AssertionError(
                f"Provider {descriptor.provider_id!r} has no base URL owner."
            )
        field = Settings.model_fields[descriptor.base_url_attr]
        env_name = field.validation_alias or descriptor.base_url_attr
        raise ApplicationUnavailableError(
            f"{env_name} is not set. Add it in the Admin UI."
        )
    for attr in descriptor.required_settings_attrs:
        if (
            provider_credential(descriptor, settings)
            if attr == descriptor.credential_attr
            else string_setting(settings, attr)
        ) is not None:
            continue
        field = Settings.model_fields[attr]
        env_name = field.validation_alias or attr
        raise ApplicationUnavailableError(
            f"{env_name} is not set. Add it in the Admin UI."
        )
    proxy = string_setting(settings, descriptor.proxy_attr)
    return ProviderConfig(
        api_key=credential,
        base_url=resolved_base_url,
        http_read_timeout=settings.http_read_timeout,
        http_write_timeout=settings.http_write_timeout,
        http_connect_timeout=settings.http_connect_timeout,
        proxy=proxy,
        log_raw_sse_events=settings.log_raw_sse_events,
        log_api_error_tracebacks=settings.log_api_error_tracebacks,
    )


def build_custom_provider_config(
    definition: CustomProviderDefinition, settings: Settings, *, api_key_number: int = 1
) -> ProviderConfig:
    credential = selected_api_key(definition.api_keys, api_key_number)
    if api_key_number != 1 and credential is None:
        raise ApplicationUnavailableError(
            f"{definition.display_name} Key {api_key_number} is not configured. "
            "Update the fallback key selection in the Admin UI."
        )
    return ProviderConfig(
        api_key=credential,
        base_url=definition.base_url,
        proxy=None,
        http_read_timeout=settings.http_read_timeout,
        http_write_timeout=settings.http_write_timeout,
        http_connect_timeout=settings.http_connect_timeout,
        log_raw_sse_events=settings.log_raw_sse_events,
        log_api_error_tracebacks=settings.log_api_error_tracebacks,
    )
