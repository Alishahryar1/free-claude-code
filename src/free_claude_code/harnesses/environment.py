"""Shared child-only environment setup for native clients."""

from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

_LOOPBACK_BYPASS_HOSTS = ("127.0.0.1", "localhost", "::1")
_NO_PROXY_KEYS = ("NO_PROXY", "no_proxy")


def client_environment(
    base_env: Mapping[str, str],
    *,
    proxy_root_url: str,
    remove_keys: Sequence[str] = (),
    remove_prefixes: tuple[str, ...] = (),
    updates: Mapping[str, str] | None = None,
    case_sensitive: bool = True,
) -> dict[str, str]:
    """Replace owned connection settings while preserving native user state."""

    keys = set(remove_keys if case_sensitive else map(str.casefold, remove_keys))
    prefixes = (
        remove_prefixes
        if case_sensitive
        else tuple(prefix.casefold() for prefix in remove_prefixes)
    )
    env = {
        key: value
        for key, value in base_env.items()
        if (name := key if case_sensitive else key.casefold()) not in keys
        and not name.startswith(prefixes)
    }
    if updates:
        env.update(updates)
    return with_local_proxy_bypass(env, proxy_root_url=proxy_root_url)


def require_unset_environment(base_env: Mapping[str, str], keys: Sequence[str]) -> None:
    """Avoid replacing another owner's explicit process configuration."""

    for key in keys:
        if base_env.get(key, "").strip():
            raise ValueError(f"{key} is already set. Unset it before using FCC.")


def with_local_proxy_bypass(
    base_env: Mapping[str, str],
    *,
    proxy_root_url: str,
) -> dict[str, str]:
    """Copy an environment and keep its FCC-local destination off proxies."""

    host = urlsplit(proxy_root_url).hostname
    if host is None:
        raise ValueError("Local proxy root URL must include a host.")

    env = dict(base_env)
    entries: list[str] = []
    seen: set[str] = set()
    for key in _NO_PROXY_KEYS:
        for raw_entry in base_env.get(key, "").split(","):
            _append_unique(entries, seen, raw_entry)
    for local_host in (*_LOOPBACK_BYPASS_HOSTS, host):
        _append_unique(entries, seen, local_host)

    value = ",".join(entries)
    for key in _NO_PROXY_KEYS:
        env[key] = value
    return env


def _append_unique(entries: list[str], seen: set[str], raw_entry: str) -> None:
    entry = raw_entry.strip()
    normalized = entry.casefold()
    if not entry or normalized in seen:
        return
    entries.append(entry)
    seen.add(normalized)
