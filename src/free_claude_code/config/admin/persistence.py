"""Sparse managed-config validation, preview, and atomic persistence."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from free_claude_code.config.api_keys import ApiKeyList, api_key_values, first_api_key
from free_claude_code.config.env_files import (
    FCC_CONFIG_SCHEMA_ENV,
)
from free_claude_code.config.env_migrations import (
    CONFIG_SCHEMA_VERSION,
    recognized_env_keys,
    render_managed_config,
)
from free_claude_code.config.errors import ConfigConflictError
from free_claude_code.config.loader import ManagedConfigSnapshot
from free_claude_code.config.model_refs import normalize_retired_model_settings
from free_claude_code.config.provider_proxies import invalid_provider_proxy_keys
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject

from .credentials import (
    CredentialEditError,
    CredentialRowError,
    CredentialSubmission,
    resolve_credential_edit,
)
from .custom_providers import CustomProviderMutation, merge_custom_provider
from .manifest import FIELD_BY_KEY, FIELDS
from .state import ConfigInputValue
from .validation import settings_from_values
from .values import MASKED_SECRET, is_locked_source, load_value_state, normalize_for_env

_PROVIDER_PROXY_ERROR = (
    "must be a proxy URL with a supported scheme and host "
    "(for example http://127.0.0.1:8080 or socks5://127.0.0.1:1080)"
)


@dataclass(frozen=True, slots=True)
class PreparedAdminUpdate:
    """Validated Admin update ready for an atomic managed-file commit."""

    target_values: dict[str, str] = field(repr=False)
    settings: Settings | None
    errors: tuple[str, ...]
    pending_fields: tuple[str, ...]
    path: Path
    expected: ManagedConfigSnapshot = field(repr=False)
    changed_keys: tuple[str, ...] = ()
    custom_provider_id: str | None = None
    credential_submissions: tuple[CredentialSubmission, ...] = ()
    credential_row_errors: tuple[CredentialRowError, ...] = ()

    @property
    def valid(self) -> bool:
        return self.settings is not None and not self.errors

    def validation_response(self) -> JsonObject:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "credential_row_errors": [
                error.public() for error in self.credential_row_errors
            ],
            "env_preview": render_managed_config(
                self.target_values,
                mask_secrets=True,
            ),
        }

    def applied_response(self) -> JsonObject:
        if not self.valid:
            return self.validation_response() | {
                "applied": False,
                "pending_fields": [],
            }
        return {
            "custom_provider_id": self.custom_provider_id,
            "applied": True,
            "valid": True,
            "errors": [],
            "env_preview": render_managed_config(
                self.target_values,
                mask_secrets=True,
            ),
            "path": str(self.path),
            "pending_fields": list(self.pending_fields),
        }


def target_values_with_updates(
    updates: Mapping[str, ConfigInputValue],
    snapshot: ManagedConfigSnapshot,
) -> dict[str, str]:
    """Return sparse managed state after applying valid partial-update semantics."""

    state = load_value_state(snapshot)
    values = dict(snapshot.managed)
    known = recognized_env_keys()
    for key in tuple(values):
        if key in known and key != FCC_CONFIG_SCHEMA_ENV and not values[key].strip():
            values.pop(key)

    for key, submitted in updates.items():
        field = FIELD_BY_KEY.get(key)
        if field is None or is_locked_source(state[key].source):
            continue
        if field.secret and (
            submitted == MASKED_SECRET
            or (isinstance(submitted, str) and not submitted.strip())
        ):
            continue
        value = normalize_for_env(submitted, key=key)
        if value is None or (field.nullable and not value):
            values.pop(key, None)
        else:
            values[key] = value

    values[FCC_CONFIG_SCHEMA_ENV] = CONFIG_SCHEMA_VERSION
    return normalize_retired_model_settings(values, preserve_empty_overrides=False)


def pending_restart_fields(active: Settings, prospective: Settings) -> tuple[str, ...]:
    """Compare saved intent with the settings actually used by running resources."""
    voice_credentials = {
        _active_voice_credential(active),
        _active_voice_credential(prospective),
    }
    pending = []
    for spec in FIELDS:
        if spec.settings_attr is None or not (
            spec.restart_required
            or spec.session_sensitive
            or spec.key in voice_credentials
        ):
            continue
        before, after = (
            getattr(active, spec.settings_attr),
            getattr(prospective, spec.settings_attr),
        )
        if spec.field_type == "api_key_list":
            before, after = first_api_key(before), first_api_key(after)
        if before != after:
            pending.append(spec.key)
    return tuple(pending)


def prepare_admin_update(
    updates: Mapping[str, ConfigInputValue],
    snapshot: ManagedConfigSnapshot,
    active_settings: Settings,
    custom_provider: CustomProviderMutation | None = None,
    *,
    revision_key: bytes,
) -> PreparedAdminUpdate:
    """Validate an update and construct its prospective Settings snapshot."""

    resolved = dict(updates)
    row_errors: list[CredentialRowError] = []
    credential_errors: list[str] = []
    submissions: list[CredentialSubmission] = []
    for key, submitted in updates.items():
        spec = FIELD_BY_KEY.get(key)
        if spec is None or spec.field_type != "api_key_list":
            continue
        assert spec.settings_attr is not None
        try:
            keys = resolve_credential_edit(
                key,
                getattr(snapshot.settings, spec.settings_attr),
                snapshot.sources[spec.settings_attr].value,
                submitted,
                revision_key,
                key=key,
            )
        except ConfigConflictError:
            raise
        except ValueError as exc:
            resolved.pop(key)
            credential_errors.append(str(exc))
            if isinstance(exc, CredentialEditError):
                row_errors.extend(exc.row_errors)
        else:
            resolved[key] = api_key_values(keys)
            submissions.append(CredentialSubmission(key, keys))

    custom_keys: ApiKeyList | None = None
    if (
        custom_provider is not None
        and custom_provider.action != "delete"
        and "api_keys" in custom_provider.values
    ):
        changes = dict(custom_provider.values)
        submitted = changes.pop("api_keys")
        definition = snapshot.settings.custom_provider(
            custom_provider.provider_id or ""
        )
        try:
            custom_keys = resolve_credential_edit(
                "custom:" + (custom_provider.provider_id or "new"),
                definition.api_keys if definition else (),
                snapshot.sources["custom_providers"].value,
                submitted,
                revision_key,
                key="api_keys",
                provider_id=custom_provider.provider_id,
                new_owner=custom_provider.action == "create",
            )
        except ConfigConflictError:
            raise
        except ValueError as exc:
            credential_errors.append(str(exc))
            if isinstance(exc, CredentialEditError):
                row_errors.extend(exc.row_errors)
        else:
            changes["api_keys"] = api_key_values(custom_keys)
        custom_provider = custom_provider.model_copy(update={"values": changes})

    update_errors = (*credential_errors, *_update_protocol_errors(resolved))
    target_values = target_values_with_updates(resolved, snapshot)
    custom_id = None
    if custom_provider is not None:
        try:
            custom_id = merge_custom_provider(target_values, snapshot, custom_provider)
        except ValueError as exc:
            update_errors = (*update_errors, str(exc))
        if custom_keys is not None:
            submissions.append(CredentialSubmission("api_keys", custom_keys, custom_id))
    settings, settings_errors = settings_from_values(target_values, snapshot.process)
    errors = (
        *update_errors,
        *settings_errors,
        *_provider_proxy_errors(target_values),
    )
    pending_fields = (
        pending_restart_fields(active_settings, settings)
        if settings is not None and not errors
        else ()
    )
    state = load_value_state(snapshot)
    changed_keys = tuple(
        key
        for key, submitted in resolved.items()
        if key in FIELD_BY_KEY
        and not is_locked_source(state[key].source)
        and (value := normalize_for_env(submitted, key=key))
        and value != MASKED_SECRET
        and value != state[key].value
        and key in target_values
        and target_values[key] == value
    )
    return PreparedAdminUpdate(
        target_values=target_values,
        settings=settings,
        errors=errors,
        pending_fields=pending_fields,
        path=snapshot.path,
        expected=snapshot,
        changed_keys=changed_keys,
        custom_provider_id=custom_id,
        credential_submissions=tuple(submissions),
        credential_row_errors=tuple(row_errors),
    )


def _update_protocol_errors(
    updates: Mapping[str, ConfigInputValue],
) -> tuple[str, ...]:
    errors: list[str] = []
    for key, submitted in updates.items():
        field = FIELD_BY_KEY.get(key)
        if field is None or (field.secret and submitted == MASKED_SECRET):
            continue
        if submitted is None and not field.nullable:
            errors.append(f"{key}: this setting cannot be removed")
            continue
        if (
            not field.secret
            and isinstance(submitted, str)
            and not submitted.strip()
            and not field.nullable
        ):
            errors.append(f"{key}: this setting cannot be blank")
    return tuple(errors)


def _provider_proxy_errors(values: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        f"{key}: {_PROVIDER_PROXY_ERROR}" for key in invalid_provider_proxy_keys(values)
    )


def _active_voice_credential(settings: Settings) -> str | None:
    if not settings.voice_note_enabled:
        return None
    if settings.whisper_device == "nvidia_nim":
        return "NVIDIA_NIM_API_KEY"
    return "HUGGINGFACE_API_KEY"
