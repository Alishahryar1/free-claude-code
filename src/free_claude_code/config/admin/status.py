"""Provider configuration status for the Admin UI."""

from collections.abc import Mapping

from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderAuthKind,
)
from free_claude_code.core.json_types import JsonObject

from .manifest import FIELDS
from .state import ConfigValueState


def provider_config_status(
    state: Mapping[str, ConfigValueState],
) -> list[JsonObject]:
    """Return provider configuration status without making network calls."""
    statuses: list[JsonObject] = []
    for provider_id, descriptor in PROVIDER_CATALOG.items():
        metadata = {
            "provider_id": provider_id,
            "display_name": descriptor.display_name,
            "website_url": descriptor.website_url,
            "logo_filename": descriptor.logo_filename,
        }
        settings_keys = [
            field.key for field in FIELDS if provider_id in field.provider_ids
        ]
        if descriptor.auth_kind is ProviderAuthKind.CONNECTED_ACCOUNT:
            statuses.append(
                {
                    **metadata,
                    "kind": "connected_account",
                    "status": "disconnected",
                    "label": "Not connected",
                    "settings_keys": settings_keys,
                }
            )
            continue

        configuration_attrs = descriptor.configuration_attrs()
        configuration_keys = [
            _field_key_for_settings_attr(attr) for attr in configuration_attrs
        ]
        missing_attrs = tuple(
            attr
            for attr in configuration_attrs
            if not _value_for_settings_attr(state, attr)
        )
        missing_configuration_keys = [
            _field_key_for_settings_attr(attr) for attr in missing_attrs
        ]

        if descriptor.local:
            base_url: str | None = None
            if descriptor.base_url_attr is not None:
                base_url = _value_for_settings_attr(state, descriptor.base_url_attr)
            statuses.append(
                {
                    **metadata,
                    "kind": "local",
                    "status": "missing_url" if missing_attrs else "configured",
                    "label": "Missing URL" if missing_attrs else "Configured",
                    "base_url": base_url or descriptor.default_base_url or "",
                    "configuration_keys": configuration_keys,
                    "missing_configuration_keys": missing_configuration_keys,
                    "settings_keys": settings_keys,
                }
            )
            continue

        configured = not missing_attrs
        missing_key = descriptor.credential_attr in missing_attrs
        statuses.append(
            {
                **metadata,
                "kind": "remote",
                "status": (
                    "configured"
                    if configured
                    else "missing_key"
                    if missing_key
                    else "missing_config"
                ),
                "label": (
                    "Configured"
                    if configured
                    else "Missing key"
                    if missing_key
                    else "Missing configuration"
                ),
                "configuration_keys": configuration_keys,
                "missing_configuration_keys": missing_configuration_keys,
                "settings_keys": settings_keys,
            }
        )
    return statuses


def _value_for_settings_attr(
    state: Mapping[str, ConfigValueState], settings_attr: str
) -> str | None:
    for field in FIELDS:
        if field.settings_attr == settings_attr:
            entry = state.get(field.key)
            value = entry.value if entry is not None else field.resolved_default()
            return str(value) if value is not None else None
    return None


def _field_key_for_settings_attr(settings_attr: str) -> str:
    for field in FIELDS:
        if field.settings_attr == settings_attr:
            return field.key
    raise AssertionError(f"No admin field owns settings attribute {settings_attr!r}")
