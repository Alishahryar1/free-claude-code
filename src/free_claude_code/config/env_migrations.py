"""One-time consolidation of legacy FCC dotenv state."""

import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .env_files import (
    ANTHROPIC_AUTH_TOKEN_ENV,
    FCC_CONFIG_SCHEMA_ENV,
    PROXY_AUTH_ENABLED_ENV,
    dotenv_values_from_file,
    dotenv_values_from_text,
    legacy_explicit_env_path,
    verified_checkout_env_path,
)
from .model_refs import normalize_retired_model_settings
from .paths import legacy_env_paths, managed_env_path
from .provider_catalog import PROVIDER_CATALOG
from .settings import Settings

CONFIG_SCHEMA_VERSION = "1"
LEGACY_HUGGINGFACE_TOKEN_ENV = "HF_TOKEN"
HUGGINGFACE_API_KEY_ENV = "HUGGINGFACE_API_KEY"
LEGACY_OPENCODE_PROXY_ENV = "OPENCODE_PROXY"
OPENCODE_ZEN_PROXY_ENV = "OPENCODE_ZEN_PROXY"
LEGACY_OPENCODE_SMOKE_MODEL_ENV = "FCC_SMOKE_MODEL_OPENCODE"
OPENCODE_ZEN_SMOKE_MODEL_ENV = "FCC_SMOKE_MODEL_OPENCODE_ZEN"

_SPECIAL_SMOKE_KEYS = frozenset(
    {
        "FCC_SMOKE_MODEL_MISTRAL_REASONING",
        "FCC_SMOKE_NIM_MODELS",
        "FCC_SMOKE_NIM_EXTRA_MODELS",
        "FCC_SMOKE_OPENROUTER_FREE_MODELS",
        "FCC_SMOKE_OPENROUTER_FREE_EXTRA_MODELS",
    }
)
_DOTENV_ASSIGNMENT_RE = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?)(?P<key>[A-Za-z_][A-Za-z0-9_]*)(?P<suffix>\s*(?:=|$))"
)


@dataclass(frozen=True, slots=True)
class EnvMigration:
    """A migration for one dotenv setting."""

    old_key: str
    new_key: str
    value_map: tuple[tuple[str, str], ...] = ()
    value_prefix_map: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ConfigMigrationResult:
    """Outcome of one managed-config consolidation attempt."""

    changed: bool
    imported_from: tuple[Path, ...] = ()


HUGGINGFACE_TOKEN_MIGRATION = EnvMigration(
    old_key=LEGACY_HUGGINGFACE_TOKEN_ENV,
    new_key=HUGGINGFACE_API_KEY_ENV,
)

_LEGACY_TRUE_VALUES = ("1", "true", "t", "on", "yes", "y")
_LEGACY_FALSE_VALUES = ("0", "false", "f", "off", "no", "n")
_LEGACY_REASONING_BOOLEAN_MAP = (
    *((value, "client") for value in _LEGACY_TRUE_VALUES),
    *((value, "off") for value in _LEGACY_FALSE_VALUES),
)

REASONING_MIGRATIONS = (
    EnvMigration(
        "ENABLE_MODEL_THINKING",
        "REASONING_POLICY",
        _LEGACY_REASONING_BOOLEAN_MAP,
    ),
    *(
        EnvMigration(
            f"ENABLE_{route}_THINKING",
            f"REASONING_{route}",
            (("", "inherit"), *_LEGACY_REASONING_BOOLEAN_MAP),
        )
        for route in ("FABLE", "OPUS", "SONNET", "HAIKU")
    ),
)

OPENCODE_ZEN_KEY_MIGRATIONS = (
    EnvMigration(LEGACY_OPENCODE_PROXY_ENV, OPENCODE_ZEN_PROXY_ENV),
    EnvMigration(
        LEGACY_OPENCODE_SMOKE_MODEL_ENV,
        OPENCODE_ZEN_SMOKE_MODEL_ENV,
        value_prefix_map=(("opencode/", "opencode_zen/"),),
    ),
)

OPENCODE_ZEN_MODEL_REF_MIGRATIONS = tuple(
    EnvMigration(
        key,
        key,
        value_prefix_map=(("opencode/", "opencode_zen/"),),
    )
    for key in (
        "MODEL",
        "MODEL_FABLE",
        "MODEL_OPUS",
        "MODEL_SONNET",
        "MODEL_HAIKU",
        OPENCODE_ZEN_SMOKE_MODEL_ENV,
    )
)

ENV_MIGRATIONS = (
    HUGGINGFACE_TOKEN_MIGRATION,
    *REASONING_MIGRATIONS,
    *OPENCODE_ZEN_KEY_MIGRATIONS,
    *OPENCODE_ZEN_MODEL_REF_MIGRATIONS,
)
_RETIRED_ENV_KEYS = frozenset(
    migration.old_key
    for migration in ENV_MIGRATIONS
    if migration.old_key != migration.new_key
)


def settings_env_keys() -> frozenset[str]:
    """Return the explicit environment aliases owned by Settings."""

    keys: set[str] = set()
    for name, field in Settings.model_fields.items():
        if name == "nim":
            continue
        alias = field.validation_alias
        if not isinstance(alias, str):
            raise AssertionError(f"Settings field {name!r} needs one string alias")
        keys.add(alias)
    return frozenset(keys)


def smoke_env_keys() -> frozenset[str]:
    """Return catalog-derived and special live-smoke configuration keys."""

    catalog_keys = {
        f"FCC_SMOKE_MODEL_{provider_id.upper()}" for provider_id in PROVIDER_CATALOG
    }
    return frozenset(catalog_keys) | _SPECIAL_SMOKE_KEYS


def recognized_env_keys() -> frozenset[str]:
    """Return every dotenv key FCC may import into managed configuration."""

    migrated_keys = {
        key
        for migration in ENV_MIGRATIONS
        for key in (migration.old_key, migration.new_key)
    }
    return (
        settings_env_keys() | smoke_env_keys() | migrated_keys | {FCC_CONFIG_SCHEMA_ENV}
    )


def secret_env_keys() -> frozenset[str]:
    """Return settings aliases whose values must be masked in previews."""

    secret_attrs = {
        name
        for name in Settings.model_fields
        if name.endswith(("_api_key", "_proxy", "_token"))
    }
    secret_attrs.update({"telegram_proxy_url", "proxy_auth_token"})
    return frozenset(
        str(Settings.model_fields[name].validation_alias) for name in secret_attrs
    )


def consolidate_managed_config(
    env: Mapping[str, str] | None = None,
) -> ConfigMigrationResult:
    """Consolidate pre-schema configuration into ``~/.fcc/.env`` once."""

    process = env if env is not None else os.environ
    managed = managed_env_path()
    base_path: Path | None = managed if managed.exists() else _legacy_base_path()
    base_is_managed = base_path == managed
    base_text = _read_text(base_path) if base_path is not None else ""
    base_values = dotenv_values_from_text(base_text)

    if base_is_managed:
        schema = base_values.get(FCC_CONFIG_SCHEMA_ENV)
        if schema == CONFIG_SCHEMA_VERSION:
            normalized = normalize_retired_model_settings(
                base_values, preserve_empty_overrides=False
            )
            if normalized == base_values:
                return ConfigMigrationResult(changed=False)
            atomic_write_managed_config(normalized, path=managed)
            _log_model_repairs(base_values, normalized)
            return ConfigMigrationResult(changed=True)
        if schema not in {None, ""}:
            raise ValueError(
                f"Managed config {managed} uses unsupported schema {schema!r}; "
                f"this FCC version supports {CONFIG_SCHEMA_VERSION}."
            )

    known = recognized_env_keys()
    base_values = _migrated_values(base_text)
    values = (
        dict(base_values)
        if base_is_managed
        else {key: value for key, value in base_values.items() if key in known}
    )
    imported: list[Path] = []
    if base_path is not None and not base_is_managed:
        imported.append(base_path.resolve())

    explicit_path = legacy_explicit_env_path(process)
    if explicit_path is not None and not _same_path(explicit_path, managed):
        explicit_values = _migrated_values(_read_required_text(explicit_path))
        values.update(
            {key: value for key, value in explicit_values.items() if key in known}
        )
        imported.append(explicit_path.resolve())

    for key in _RETIRED_ENV_KEYS:
        values.pop(key, None)
    values.pop(FCC_CONFIG_SCHEMA_ENV, None)
    _materialize_auth_state(values, process, had_legacy_state=base_path is not None)
    for key in tuple(values):
        if key in known and key != FCC_CONFIG_SCHEMA_ENV and not values[key].strip():
            values.pop(key)
    values[FCC_CONFIG_SCHEMA_ENV] = CONFIG_SCHEMA_VERSION
    normalized = normalize_retired_model_settings(
        values, preserve_empty_overrides=False
    )
    atomic_write_managed_config(normalized, path=managed)
    _log_model_repairs(values, normalized)
    return ConfigMigrationResult(changed=True, imported_from=tuple(imported))


def _log_model_repairs(before: Mapping[str, str], after: Mapping[str, str]) -> None:
    changed = sorted(key for key in before if before[key] != after.get(key))
    if changed:
        logger.info(
            "Repaired retired provider selections in managed config: {}",
            ", ".join(changed),
        )


def atomic_write_managed_config(
    values: Mapping[str, str],
    *,
    path: Path | None = None,
    mask_secrets: bool = False,
) -> str:
    """Atomically write canonical managed config and return the rendered text."""

    target = path or managed_env_path()
    rendered = render_managed_config(values, mask_secrets=mask_secrets)
    if mask_secrets:
        return rendered

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            temp_path.chmod(0o600)
        os.replace(temp_path, target)
        if os.name != "nt":
            target.chmod(0o600)
    finally:
        temp_path.unlink(missing_ok=True)
    return rendered


def render_managed_config(
    values: Mapping[str, str],
    *,
    mask_secrets: bool = False,
) -> str:
    """Render sparse canonical FCC configuration."""

    known = recognized_env_keys()
    secret_keys = secret_env_keys()
    canonical = dict(values)
    canonical[FCC_CONFIG_SCHEMA_ENV] = CONFIG_SCHEMA_VERSION
    lines = [
        "# Managed by Free Claude Code.",
        "# Edit settings in /admin when possible.",
        f"{FCC_CONFIG_SCHEMA_ENV}={CONFIG_SCHEMA_VERSION}",
    ]
    configured = [
        key
        for key in canonical
        if key != FCC_CONFIG_SCHEMA_ENV and key in known and canonical[key].strip()
    ]
    if configured:
        lines.extend(("", "# Configured settings"))
        for key in sorted(configured):
            value = (
                "********" if mask_secrets and key in secret_keys else canonical[key]
            )
            lines.append(f"{key}={quote_env_value(value)}")

    unknown = [key for key in canonical if key not in known]
    if unknown:
        lines.extend(("", "# Preserved unrecognized settings"))
        lines.extend(
            f"{key}={quote_env_value('********' if mask_secrets else canonical[key])}"
            for key in sorted(unknown)
        )
    return "\n".join(lines) + "\n"


def quote_env_value(value: str) -> str:
    """Quote a value when dotenv syntax requires it."""

    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    if (
        not value
        or any(char.isspace() for char in value)
        or any(char in value for char in ('"', "#", "=", "$"))
    ):
        return f'"{escaped}"'
    return value


def migrate_env_setting_in_text(
    text: str,
    migration: EnvMigration,
) -> tuple[str, bool]:
    """Return text with one setting migration applied when safe."""

    if migration.old_key != migration.new_key and _defines_key(text, migration.new_key):
        return text, False

    lines = text.splitlines(keepends=True)
    changed = False
    for index, line in enumerate(lines):
        match = _DOTENV_ASSIGNMENT_RE.match(line)
        if match is None or match.group("key") != migration.old_key:
            continue
        remainder = line[match.end() :]
        if migration.value_map:
            remainder = _mapped_value(remainder, migration.value_map)
        for old_prefix, new_prefix in migration.value_prefix_map:
            remainder = _mapped_prefix_value(remainder, old_prefix, new_prefix)
        migrated_line = (
            f"{match.group('prefix')}{migration.new_key}{match.group('suffix')}"
            f"{remainder}"
        )
        if migrated_line == line:
            continue
        lines[index] = migrated_line
        changed = True
    if not changed:
        return text, False
    return "".join(lines), True


def env_text_needs_migration(text: str, migration: EnvMigration) -> bool:
    """Return whether one setting migration would change text."""

    return migrate_env_setting_in_text(text, migration)[1]


def _legacy_base_path() -> Path | None:
    for path in legacy_env_paths():
        if path.is_file():
            return path
    return verified_checkout_env_path()


def _read_text(path: Path | None) -> str:
    return "" if path is None else _read_required_text(path)


def _read_required_text(path: Path) -> str:
    dotenv_values_from_file(path)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not read configuration file {path}: {exc}") from exc


def _migrated_values(text: str) -> dict[str, str]:
    migrated = text
    for migration in ENV_MIGRATIONS:
        migrated, _ = migrate_env_setting_in_text(migrated, migration)
    return dotenv_values_from_text(migrated)


def _materialize_auth_state(
    values: dict[str, str],
    process: Mapping[str, str],
    *,
    had_legacy_state: bool,
) -> None:
    if PROXY_AUTH_ENABLED_ENV in values:
        return
    if PROXY_AUTH_ENABLED_ENV in process:
        return
    if ANTHROPIC_AUTH_TOKEN_ENV in values:
        values[PROXY_AUTH_ENABLED_ENV] = (
            "true" if values[ANTHROPIC_AUTH_TOKEN_ENV].strip() else "false"
        )
        return
    if had_legacy_state:
        values[PROXY_AUTH_ENABLED_ENV] = "false"


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _defines_key(text: str, key: str) -> bool:
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = _DOTENV_ASSIGNMENT_RE.match(line)
        if match is not None and match.group("key") == key:
            return True
    return False


def _mapped_value(value: str, mapping: tuple[tuple[str, str], ...]) -> str:
    line = value.rstrip("\r\n")
    newline = value[len(line) :]
    raw_value, separator, comment = line.partition("#")
    normalized = raw_value.strip().strip("'\"").lower()
    replacement = dict(mapping).get(normalized)
    if replacement is None:
        return value
    suffix = f" #{comment}" if separator else ""
    return f"{replacement}{suffix}{newline}"


def _mapped_prefix_value(value: str, old_prefix: str, new_prefix: str) -> str:
    pattern = rf"^(?P<leading>\s*['\"]?){re.escape(old_prefix)}"
    return re.sub(
        pattern,
        lambda match: f"{match.group('leading')}{new_prefix}",
        value,
        count=1,
    )
