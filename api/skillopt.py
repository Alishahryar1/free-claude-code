"""SkillOpt — runtime consumer of the EvoForge-produced skill policy.

Reads ``${FCC_CACHE_DIR}/skillopt_policy.json`` (an EvoForge artefact) and
returns the primary ``provider/model`` ref for a given skill. The gateway
consults this from ``ModelRouter.resolve`` **after** direct-slug routing
(``provider/model`` requests still go where the caller asked) and
**before** falling through to ``MODEL_*`` tier overrides.

Kill switch: ``SKILLOPT_ENABLED`` — off by default. When disabled the
lookup returns ``None`` and the router behaves exactly as it did before
this module existed. When enabled but the policy file is missing /
malformed / silent about a skill, the lookup also returns ``None``. Never
raises into the request path.

Cache: the policy file's mtime is checked on every lookup. When it
changes, the JSON is reparsed and cached; when the file disappears, the
cache is invalidated so a subsequent publish is picked up without a
gateway restart.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_CACHE: _CachedPolicy | None = None


@dataclass(frozen=True, slots=True)
class SkillPolicy:
    """One skill's routing decision."""

    primary: str
    fallbacks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _CachedPolicy:
    path: Path
    mtime_ns: int
    version: int
    policies: dict[str, SkillPolicy]


def is_enabled() -> bool:
    """Return whether SkillOpt is currently allowed to override routing."""
    return os.environ.get("SKILLOPT_ENABLED", "").lower() in {"1", "true", "yes"}


def policy_path() -> Path:
    base = os.environ.get("FCC_CACHE_DIR")
    root = Path(base) if base else Path.home() / ".fcc-cache"
    return root / "skillopt_policy.json"


def lookup(skill: str | None) -> SkillPolicy | None:
    """Return the policy for *skill*, or ``None`` when nothing applies."""
    if not skill or not is_enabled():
        return None
    cached = _current_cache()
    if cached is None:
        return None
    return cached.policies.get(skill)


def fallbacks(skill: str | None) -> tuple[str, ...]:
    """Return the ordered fallback refs for *skill*, or an empty tuple.

    Only usable when SkillOpt is enabled AND a policy covers *skill*.
    Returns bare, syntactically-well-formed ``provider/model`` strings —
    callers still verify the provider is registered before use.
    """
    policy = lookup(skill)
    if policy is None:
        return ()
    return policy.fallbacks


def snapshot() -> dict[str, Any]:
    """Non-sensitive view for the admin UI."""
    cached = _current_cache()
    if cached is None:
        return {
            "enabled": is_enabled(),
            "loaded": False,
            "path": str(policy_path()),
        }
    return {
        "enabled": is_enabled(),
        "loaded": True,
        "path": str(cached.path),
        "version": cached.version,
        "policies": {
            skill: {"primary": p.primary, "fallbacks": list(p.fallbacks)}
            for skill, p in cached.policies.items()
        },
    }


def invalidate_cache() -> None:
    """Force reparse on next lookup (for tests / manual reload)."""
    global _CACHE
    with _LOCK:
        _CACHE = None


def _current_cache() -> _CachedPolicy | None:
    global _CACHE
    path = policy_path()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except FileNotFoundError:
        with _LOCK:
            _CACHE = None
        return None
    except OSError:
        return None

    with _LOCK:
        if _CACHE is not None and _CACHE.path == path and _CACHE.mtime_ns == mtime_ns:
            return _CACHE
        parsed = _parse(path, mtime_ns)
        _CACHE = parsed
        return parsed


def _parse(path: Path, mtime_ns: int) -> _CachedPolicy | None:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except OSError:
        return None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    version_raw = data.get("version")
    version = int(version_raw) if isinstance(version_raw, int) else 0

    policies_raw = data.get("policies")
    if not isinstance(policies_raw, dict):
        return None

    policies: dict[str, SkillPolicy] = {}
    for skill, entry in policies_raw.items():
        if not isinstance(skill, str) or not isinstance(entry, dict):
            continue
        primary = entry.get("primary")
        if not isinstance(primary, str) or not primary.strip():
            continue
        fallbacks_raw = entry.get("fallbacks") or []
        if not isinstance(fallbacks_raw, list):
            continue
        fallbacks = tuple(f for f in fallbacks_raw if isinstance(f, str) and f.strip())
        policies[skill] = SkillPolicy(primary=primary, fallbacks=fallbacks)

    return _CachedPolicy(
        path=path,
        mtime_ns=mtime_ns,
        version=version,
        policies=policies,
    )
