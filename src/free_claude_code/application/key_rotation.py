"""Per-credential cooldowns backed by FCC SQLite; stores fingerprints only."""

import hashlib
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Protocol

from pydantic import SecretStr

from free_claude_code.config.settings import Settings
from free_claude_code.core.failure_evidence import (
    error_text,
    reported_status,
    reports_quota_403,
    reports_tpm_limit,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject


class CooldownDatabase(Protocol):
    """Writer-pool surface of FCC's SQLite database that this registry needs."""

    def execute[T](
        self, operation: Callable[[sqlite3.Connection], T], *, write: bool = True
    ) -> T: ...


def _read_cooldown(
    database: CooldownDatabase, identity: str
) -> tuple[float, str | None] | None:
    def read(connection: sqlite3.Connection) -> tuple[float, str | None] | None:
        row: object = connection.execute(
            "SELECT cooldown_until, last_error FROM provider_key_cooldowns "
            "WHERE credential_id = ?",
            (identity,),
        ).fetchone()
        if not isinstance(row, sqlite3.Row):
            return None
        until: object = row[0]
        error: object = row[1]
        if not isinstance(until, int | float):
            return None
        return float(until), error if isinstance(error, str) else None

    return database.execute(read, write=False)


def _response_headers(error: BaseException) -> Mapping[str, str]:
    response: object = getattr(error, "response", None)
    headers: object = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return {}
    return {
        str(key).lower(): str(value)
        for key, value in headers.items()
        if isinstance(key, str | bytes)
    }


@dataclass(slots=True)
class KeyState:
    provider_id: str
    key_number: int
    cooldown_until: float = 0.0
    last_error: str | None = None
    advertised: bool = False  # window came from the provider, not inferred


_RETRY_SUFFIX = re.compile(r"\s*Retry this provider route in \d+ seconds?\.")


class KeyRotationRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._states: dict[str, KeyState] = {}
        self._tpm_limits: dict[str, tuple[int, float]] = {}
        self._details: dict[str, str] = {}
        self._database: CooldownDatabase | None = None

    def bind_database(self, database: CooldownDatabase | None) -> None:
        with self._lock:
            self._database = database
            states = tuple(self._states.items()) if database is not None else ()
        if database is not None and states:
            database.execute(
                lambda connection: connection.executemany(
                    "INSERT INTO provider_key_cooldowns VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(credential_id) DO UPDATE SET "
                    "cooldown_until=MAX(provider_key_cooldowns.cooldown_until, excluded.cooldown_until), "
                    "last_error=excluded.last_error",
                    (
                        (
                            identity,
                            state.provider_id,
                            state.key_number,
                            state.cooldown_until,
                            state.last_error,
                        )
                        for identity, state in states
                    ),
                )
            )

    @staticmethod
    def _id(provider: str, key: str) -> str:
        return f"{provider}:{hashlib.sha256(key.encode()).hexdigest()}"

    def available(self, provider: str, number: int, key: str) -> bool:
        identity = self._id(provider, key)
        now = time.time()
        if self._database is not None:
            persisted = _read_cooldown(self._database, identity)
            if persisted is not None:
                return persisted[0] <= now
        with self._lock:
            state = self._states.get(identity)
            return state is None or state.cooldown_until <= now

    def mark(
        self,
        provider: str,
        number: int,
        key: str,
        error: Exception,
        *,
        fallback_seconds: int = 3600,
    ) -> None:
        if not is_quota_failure(error):
            return
        seconds = cooldown_seconds(error, fallback_seconds=fallback_seconds)
        advertised = cooldown_is_advertised(error)
        until = time.time() + seconds
        identity = self._id(provider, key)
        safe = safe_error(error)
        with self._lock:
            previous = self._states.get(identity)
            if previous is not None and previous.cooldown_until > until:
                until = previous.cooldown_until
            self._states[identity] = KeyState(provider, number, until, safe, advertised)
        if self._database is not None:
            self._database.execute(
                lambda connection: connection.execute(
                    "INSERT INTO provider_key_cooldowns VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(credential_id) DO UPDATE SET "
                    "provider_id=excluded.provider_id, key_number=excluded.key_number, "
                    "cooldown_until=MAX(provider_key_cooldowns.cooldown_until, excluded.cooldown_until), "
                    "last_error=excluded.last_error",
                    (identity, provider, number, until, safe),
                ),
            )

    def mark_provider(
        self,
        provider: str,
        keys: list[tuple[int, str]],
        error: Exception,
        *,
        fallback_seconds: int = 3600,
    ) -> None:
        """Cool every key of a provider after an organization-wide cap."""
        for number, key in keys:
            self.mark(provider, number, key, error, fallback_seconds=fallback_seconds)

    def remaining(self, provider: str, keys: list[tuple[int, str]]) -> float:
        """Smallest remaining cooldown across keys (0 when any key is usable)."""
        now = time.time()
        waits = []
        for number, key in keys:
            if self.available(provider, number, key):
                return 0.0
            identity = self._id(provider, key)
            with self._lock:
                state = self._states.get(identity)
            until = state.cooldown_until if state else now
            if self._database is not None:
                stored = _read_cooldown(self._database, identity)
                if stored is not None:
                    until = max(until, stored[0])
            waits.append(max(0.0, until - now))
        return min(waits) if waits else 0.0

    def provider_advertised(self, provider: str, keys: list[tuple[int, str]]) -> bool:
        """True when every key is cooling on a provider-advertised window."""
        now = time.time()
        with self._lock:
            states = [self._states.get(self._id(provider, key)) for _, key in keys]
        return bool(keys) and all(
            state is not None and state.advertised and state.cooldown_until > now
            for state in states
        )

    def remember_detail(self, provider: str, message: str) -> None:
        """Keep FCC's already-redacted failure text in memory only (never persisted)."""
        with self._lock:
            self._details[provider] = message[:4000]

    def cooldown_message(self, provider: str, seconds: int) -> str:
        """Last redacted failure text for a route, restated with the wait left."""
        unit = "second" if seconds == 1 else "seconds"
        suffix = f"Retry this provider route in {seconds} {unit}."
        with self._lock:
            text = self._details.get(provider)
        base = "All configured API keys are cooling down."
        if not text:
            return f"{base} {suffix}"
        text = _RETRY_SUFFIX.sub("", text)
        head, marker, tail = text.partition("\n\nUpstream error:")
        return f"{head.rstrip()} {suffix}{marker}{tail}"

    def learn_tpm_limit(self, provider: str, limit: int) -> None:
        with self._lock:
            self._tpm_limits[provider] = (limit, time.time() + 3600.0)

    def tpm_limit(self, provider: str) -> int | None:
        with self._lock:
            entry = self._tpm_limits.get(provider)
            if entry is None or entry[1] <= time.time():
                self._tpm_limits.pop(provider, None)
                return None
            return entry[0]

    def clear(self) -> None:
        with self._lock:
            self._states.clear()
            self._tpm_limits.clear()
            self._details.clear()

    def status(self, settings: Settings) -> list[JsonObject]:
        rows: list[JsonObject] = []
        with self._lock:
            for provider, number, key in configured_keys(settings):
                identity = self._id(provider, key)
                state = self._states.get(identity)
                persisted = (
                    _read_cooldown(self._database, identity)
                    if self._database is not None
                    else None
                )
                until = (
                    persisted[0]
                    if persisted
                    else (state.cooldown_until if state else 0.0)
                )
                last_error = (
                    persisted[1] if persisted else (state.last_error if state else None)
                )
                rows.append(
                    {
                        "provider_id": provider,
                        "key_number": number,
                        "status": "cooldown" if until > time.time() else "available",
                        "cooldown_until": until or None,
                        "cooldown_seconds": max(0, int(until - time.time())),
                        "last_error": last_error if until > time.time() else None,
                    }
                )
        return rows


def configured_keys(settings: Settings) -> Iterator[tuple[str, int, str]]:
    from free_claude_code.config.api_keys import api_key_values
    from free_claude_code.config.provider_catalog import PROVIDER_CATALOG

    for provider, descriptor in PROVIDER_CATALOG.items():
        attr = descriptor.credential_attr
        if attr:
            keys: object = getattr(settings, attr)
            if isinstance(keys, tuple) and all(
                isinstance(item, SecretStr) for item in keys
            ):
                for number, key in enumerate(api_key_values(tuple(keys)), 1):
                    yield provider, number, key
    for definition in settings.custom_providers:
        for number, key in enumerate(api_key_values(definition.api_keys), 1):
            yield definition.provider_id, number, key


def error_chain(error: BaseException) -> Iterator[BaseException]:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


class RotationScope(StrEnum):
    """Which credential set a quota-style failure exhausts."""

    KEY = "key"  # one credential's quota; another key may work
    PROVIDER = "provider"  # organization-wide cap; sibling keys will fail too


def rotation_scope(error: Exception) -> RotationScope | None:
    """Decide from status plus structured evidence whether rotation applies.

    Free text is never sufficient: 400/401 never rotate, 413 rotates only with a
    structured rate-limit code (and then only to another provider), 403 needs
    structured quota evidence, and 429 is a rate-limit/quota response.
    """
    chain = [item for item in error_chain(error) if isinstance(item, Exception)]
    # Structured TPM evidence on a raw upstream cause wins over a wrapper that a
    # provider remapped to 400 (e.g. Groq's "reduce the token budget" failure).
    if any(
        reports_tpm_limit(item) and not isinstance(item, ExecutionFailure)
        for item in chain
    ):
        return RotationScope.PROVIDER
    if any(reported_status(item) in {400, 401} for item in chain):
        return None
    scope: RotationScope | None = None
    for current in chain:
        status = reported_status(current)
        if (
            status == 429
            or reports_quota_403(current)
            or (
                isinstance(current, ExecutionFailure)
                and current.kind is FailureKind.RATE_LIMIT
                and current.status_code == 429
            )
        ):
            scope = RotationScope.KEY
    return scope


def is_quota_failure(error: Exception) -> bool:
    return rotation_scope(error) is not None


def failure_category(error: Exception) -> str:
    """Stable, secret-free label stored and shown in status."""
    chain = [item for item in error_chain(error) if isinstance(item, Exception)]
    if any(reports_tpm_limit(item) for item in chain):
        return "tpm_limit"
    text = " ".join(error_text(item) for item in chain)
    if any(m in text for m in _DAILY_MARKERS):
        return "daily_quota"
    if "insufficient_quota" in text:
        return "provider_quota"
    return "rate_limit"


_DAILY_MARKERS = (
    "daily quota",
    "daily limit",
    "quota exceeded",
    "free usage is used up",
)


def cooldown_is_advertised(error: Exception) -> bool:
    """Whether the wait comes from the provider (Retry-After/reset), not a default."""
    for current in error_chain(error):
        headers = _response_headers(current)
        if any(
            headers.get(name)
            for name in (
                "retry-after",
                "x-ratelimit-reset",
                "x-ratelimit-reset-requests",
                "ratelimit-reset",
            )
        ):
            return True
        if _REPORTED_ROUTE_WAIT.search(str(current).lower()):
            return True
    return False


_REPORTED_ROUTE_WAIT = re.compile(r"retry this provider route in (\d+) seconds?")


def cooldown_seconds(error: Exception, *, fallback_seconds: int = 3600) -> float:
    text = " ".join(
        f"{str(item).lower()} {error_text(item)}" for item in error_chain(error)
    )
    for current in error_chain(error):
        headers = _response_headers(current)
        retry = headers.get("retry-after")
        if retry:
            try:
                return max(1.0, float(retry))
            except TypeError, ValueError:
                try:
                    parsed = parsedate_to_datetime(retry)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=UTC)
                    return max(1.0, parsed.timestamp() - time.time())
                except TypeError, ValueError, OverflowError:
                    pass
        for name in (
            "x-ratelimit-reset",
            "x-ratelimit-reset-requests",
            "ratelimit-reset",
        ):
            value = headers.get(name)
            if value:
                try:
                    numeric = float(value)
                    if numeric > 1_000_000_000:
                        numeric -= time.time()
                    return max(1.0, numeric)
                except TypeError, ValueError:
                    duration = re.fullmatch(
                        r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?",
                        value,
                    )
                    if duration and any(duration.groups()):
                        return max(
                            1.0,
                            sum(
                                float(part or 0) * scale
                                for part, scale in zip(
                                    duration.groups(), (3600, 60, 1), strict=True
                                )
                            ),
                        )
    reported = re.search(r"retry this provider route in (\d+) seconds?", text)
    if reported:
        return max(1.0, float(reported.group(1)))
    if any(marker in text for marker in _DAILY_MARKERS):
        return 86400.0
    if any(reports_tpm_limit(item) for item in error_chain(error)):
        return 60.0
    return float(max(60, fallback_seconds))


_SAFE_LABELS = {
    "tpm_limit": "Token-per-minute limit reached",
    "daily_quota": "Daily quota exhausted",
    "provider_quota": "Provider quota exhausted",
    "rate_limit": "Provider rate limit reached",
}


def safe_error(error: Exception) -> str:
    """Store only a short category label, never provider-supplied error text."""
    return _SAFE_LABELS[failure_category(error)]


key_rotation_registry = KeyRotationRegistry()
