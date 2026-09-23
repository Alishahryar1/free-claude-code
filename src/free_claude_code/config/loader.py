"""Canonical managed-config loading, precedence, provenance, and caching."""

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from free_claude_code.core.interprocess_lock import InterprocessFileLock

from .env_files import (
    ANTHROPIC_AUTH_TOKEN_ENV,
    FCC_CONFIG_SCHEMA_ENV,
    dotenv_values_from_file,
)
from .env_migrations import (
    CONFIG_SCHEMA_VERSION,
    atomic_write_managed_config,
    consolidate_managed_config,
    settings_env_keys,
)
from .model_refs import normalize_retired_model_settings
from .paths import config_lock_path, managed_env_path
from .provider_proxies import invalid_provider_proxy_keys
from .settings import Settings


class ConfigSource(StrEnum):
    """Live owner of one effective setting value."""

    DEFAULT = "default"
    MANAGED = "managed_env"
    PROCESS = "process"


@dataclass(frozen=True, slots=True)
class SettingsSnapshot:
    """One validated settings model and its effective field provenance."""

    settings: Settings
    sources: Mapping[str, ConfigSource]


@dataclass(frozen=True, slots=True)
class ManagedConfigSnapshot:
    """One disk/environment capture and its validated effective settings."""

    settings: Settings
    sources: Mapping[str, ConfigSource]
    managed: Mapping[str, str]
    process: Mapping[str, str]
    path: Path


class ManagedConfigStore:
    """Own initialization and coordinate fresh reads with atomic writes."""

    def __init__(self) -> None:
        self.path = managed_env_path()
        self._lock_path = config_lock_path()

    @contextmanager
    def _storage_lock(self) -> Iterator[None]:
        lock = InterprocessFileLock(self._lock_path)
        if not lock.acquire(wait=True, timeout=10.0):
            raise TimeoutError(
                f"Could not acquire managed-config lock: {self._lock_path}"
            )
        try:
            yield
        finally:
            lock.release()

    def initialize(self, env: Mapping[str, str] | None = None) -> None:
        with self._storage_lock():
            consolidate_managed_config(dict(os.environ if env is None else env))

    def _read_managed(self) -> dict[str, str]:
        if not self.path.is_file():
            raise ValueError(
                "Managed configuration is missing; initialize it before reading."
            )
        managed = dotenv_values_from_file(self.path)
        schema = managed.get(FCC_CONFIG_SCHEMA_ENV)
        if schema != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"Managed config {self.path} uses unsupported schema {schema!r}; "
                f"initialize with a compatible FCC version (supports {CONFIG_SCHEMA_VERSION})."
            )
        return managed

    def read(self, env: Mapping[str, str] | None = None) -> ManagedConfigSnapshot:
        process = dict(os.environ if env is None else env)
        # Windows readers deny replacement while their file handles remain open.
        with self._storage_lock():
            managed = self._read_managed()
        snapshot = compose_settings_snapshot(managed, process)
        return ManagedConfigSnapshot(
            snapshot.settings,
            snapshot.sources,
            MappingProxyType(managed),
            MappingProxyType(process),
            self.path,
        )

    def commit(self, values: Mapping[str, str]) -> None:
        with self._storage_lock():
            self._read_managed()  # Never overwrite a newer schema or missing storage.
            atomic_write_managed_config(values, path=self.path)

    def repair_invalid_provider_proxies(
        self,
        env: Mapping[str, str] | None = None,
    ) -> tuple[str, ...]:
        process = dict(os.environ if env is None else env)
        if not self.path.is_file():
            return ()
        with self._storage_lock():
            if not self.path.is_file():
                return ()
            managed = self._read_managed()
            removed = tuple(
                key
                for key in invalid_provider_proxy_keys(managed)
                if key not in process
            )
            if removed:
                for key in removed:
                    managed.pop(key)
                atomic_write_managed_config(managed, path=self.path)
            return removed


def compose_settings_snapshot(
    managed: Mapping[str, str],
    env: Mapping[str, str],
) -> SettingsSnapshot:
    """Validate prospective managed values with live process precedence."""

    process = normalize_retired_model_settings(env, preserve_empty_overrides=True)
    managed = normalize_retired_model_settings(managed, preserve_empty_overrides=False)
    aliases = _settings_aliases()
    recognized = settings_env_keys()
    values: dict[str, str] = {
        key: value for key, value in managed.items() if key in recognized
    }
    sources = dict.fromkeys(Settings.model_fields, ConfigSource.DEFAULT)
    for key in values:
        if name := aliases.get(key):
            sources[name] = ConfigSource.MANAGED

    for key in recognized:
        if key not in process:
            continue
        # Servers and independently opened clients share the managed token/default.
        if key == ANTHROPIC_AUTH_TOKEN_ENV:
            continue
        values[key] = process[key]
        if name := aliases.get(key):
            sources[name] = ConfigSource.PROCESS

    settings = Settings.model_validate(values)
    return SettingsSnapshot(
        settings=settings,
        sources=MappingProxyType(sources),
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings model."""

    store = ManagedConfigStore()
    store.initialize()
    return store.read().settings


def clear_settings_cache() -> None:
    """Discard the runtime Settings cache after a committed configuration change."""

    get_settings.cache_clear()


def _settings_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name, field in Settings.model_fields.items():
        if name == "nim":
            continue
        alias = field.validation_alias
        if not isinstance(alias, str):
            raise AssertionError(f"Settings field {name!r} needs one string alias")
        aliases[alias] = name
    return aliases
