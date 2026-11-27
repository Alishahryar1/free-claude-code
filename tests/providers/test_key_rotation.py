from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path

import pytest

from free_claude_code.application.key_rotation import (
    KeyRotationRegistry,
    cooldown_seconds,
    is_quota_failure,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from tests.conftest import DatabaseFactory
from tests.providers.upstream_errors import UpstreamError, http_error


def test_rotation_marks_rate_limited_credential_without_retaining_plaintext(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = KeyRotationRegistry()
    error = ExecutionFailure(FailureKind.RATE_LIMIT, 429, "rate limited", True)
    monkeypatch.setattr(
        "free_claude_code.application.key_rotation.time.time", lambda: 100
    )
    registry.mark("groq", 1, "secret-value", error)
    assert not registry.available("groq", 1, "secret-value")
    assert registry.available("groq", 2, "other-secret")
    rows = registry.status(Settings(GROQ_API_KEY=["secret-value"]))
    assert all("secret-value" not in str(row) for row in rows)
    assert (
        next(row for row in rows if row["provider_id"] == "groq")["status"]
        == "cooldown"
    )


def test_cooldown_expiry_restores_key(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = KeyRotationRegistry()
    now = [100.0]
    monkeypatch.setattr(
        "free_claude_code.application.key_rotation.time.time", lambda: now[0]
    )
    registry.mark(
        "p",
        1,
        "secret",
        ExecutionFailure(FailureKind.RATE_LIMIT, 429, "rate limit", True),
        fallback_seconds=60,
    )
    assert not registry.available("p", 1, "secret")
    now[0] += 61
    assert registry.available("p", 1, "secret")


def test_retry_after_and_reset_headers_set_cooldown():
    retry = UpstreamError("quota", headers={"retry-after": "123"})
    assert cooldown_seconds(retry) == 123
    reset = UpstreamError("quota", headers={"x-ratelimit-reset-requests": "90"})
    assert cooldown_seconds(reset) == 90


def test_retry_after_http_date_uses_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "free_claude_code.application.key_rotation.time.time", lambda: 1_700_000_000
    )
    retry = UpstreamError(
        "quota",
        headers={
            "retry-after": format_datetime(
                datetime.fromtimestamp(1_700_000_120, UTC), usegmt=True
            )
        },
    )
    assert cooldown_seconds(retry) == 120


@pytest.mark.parametrize(
    "error",
    [
        http_error(429, message="Today's free usage is used up."),
        http_error(429, code="insufficient_quota", message="quota"),
        http_error(403, code="insufficient_quota", message="quota"),
    ],
)
def test_quota_exhaustion_is_detected(error: UpstreamError) -> None:
    assert is_quota_failure(error)


@pytest.mark.parametrize(
    "error",
    [
        UpstreamError("insufficient_quota"),  # prose alone is never evidence
        UpstreamError("rate_limit_error daily quota exhausted"),
        http_error(500, message="rate limit"),
        http_error(403, message="access denied"),
        http_error(400, code="rate_limit_exceeded"),
        http_error(401, code="rate_limit_exceeded"),
    ],
)
def test_text_without_status_or_structure_is_not_quota_failure(
    error: UpstreamError,
) -> None:
    assert not is_quota_failure(error)


def test_daily_quota_uses_day_long_cooldown():
    assert cooldown_seconds(UpstreamError("daily quota exhausted")) == 86400

    assert cooldown_seconds(UpstreamError("rate limit"), fallback_seconds=7200) == 7200


def test_http_403_without_explicit_quota_marker_is_not_quota_failure():
    assert not is_quota_failure(
        ExecutionFailure(FailureKind.PERMISSION, 403, "access denied", False)
    )


def test_http_403_with_structured_quota_evidence_is_quota_failure():
    assert is_quota_failure(
        http_error(403, code="insufficient_quota", message="daily quota exceeded")
    )


@pytest.mark.asyncio
async def test_persisted_cooldown_survives_database_reopen(
    database_factory: DatabaseFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, lock = tmp_path / "fcc.db", tmp_path / "fcc.lock"
    now = [100.0]
    monkeypatch.setattr(
        "free_claude_code.application.key_rotation.time.time", lambda: now[0]
    )
    first = database_factory(path, lock)
    await first.start()
    registry = KeyRotationRegistry()
    registry.bind_database(first)
    registry.mark(
        "p",
        1,
        "secret",
        ExecutionFailure(FailureKind.RATE_LIMIT, 429, "rate limited", True),
        fallback_seconds=60,
    )
    assert not registry.available("p", 1, "secret")
    registry.bind_database(None)
    registry.clear()
    await first.close()

    now[0] += 30
    reopened = database_factory(path, lock)
    await reopened.start()
    registry.bind_database(reopened)
    assert not registry.available("p", 1, "secret")
    assert registry.remaining("p", [(1, "secret")]) > 0
    stored = reopened.execute(
        lambda connection: tuple(
            connection.execute("SELECT * FROM provider_key_cooldowns").fetchall()
        ),
        write=False,
    )
    assert "secret" not in str(stored)
    now[0] += 31
    assert registry.available("p", 1, "secret")
    registry.bind_database(None)
    await reopened.close()


def test_wrapped_retry_after_overrides_daily_default() -> None:
    original = UpstreamError("daily quota exhausted", headers={"retry-after": "123"})
    wrapped = ExecutionFailure(FailureKind.RATE_LIMIT, 429, "rate limited", True)
    wrapped.__cause__ = original
    assert cooldown_seconds(wrapped) == 123


def test_duration_reset_header():
    error = UpstreamError("rate limit", headers={"x-ratelimit-reset-requests": "1m30s"})
    assert cooldown_seconds(error) == 90


def test_wrapped_quota_403_is_detected():
    original = http_error(403, code="insufficient_quota", message="quota")
    wrapped = ExecutionFailure(FailureKind.PERMISSION, 403, "access denied", False)
    wrapped.__cause__ = original
    assert is_quota_failure(wrapped)


@pytest.mark.parametrize("status", [400, 401])
def test_raw_auth_or_request_status_prevents_rotation(status: int) -> None:
    original = ExecutionFailure(
        FailureKind.AUTHENTICATION, status, "quota exceeded", False
    )
    wrapped = ExecutionFailure(FailureKind.RATE_LIMIT, 429, "rate limit", True)
    wrapped.__cause__ = original
    assert not is_quota_failure(wrapped)


@pytest.mark.asyncio
async def test_all_keys_cooling_down_returns_rate_limit_without_opening_provider():
    from unittest.mock import AsyncMock

    from free_claude_code.application.model_fallback.runner import (
        stream_model_candidates,
    )
    from free_claude_code.core.stream_delivery import StreamDeliveryState

    opener = AsyncMock()
    with pytest.raises(ExecutionFailure) as captured:
        async for _ in stream_model_candidates(
            (),
            opener,
            get_delivery=StreamDeliveryState,
            request_id="cooldown-test",
            wire_api="messages",
        ):
            pytest.fail("A request without available keys must not emit output")
    assert captured.value.status_code == 429
    opener.assert_not_called()


def test_configured_fallback_remains_explicit_when_its_first_key_is_cooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from free_claude_code.application.execution import ProviderExecutor
    from free_claude_code.application.ports import ProviderPort
    from free_claude_code.application.routing import (
        ProviderModelCandidate,
        ProviderModelTarget,
        ResolvedModelRoute,
    )
    from free_claude_code.config.reasoning import ReasoningPreference

    registry = KeyRotationRegistry()
    monkeypatch.setattr(
        "free_claude_code.application.execution.key_rotation_registry", registry
    )
    registry.mark("groq", 1, "fallback-first", http_error(429))
    settings = Settings(
        ANTHROPIC_API_KEY=["primary"],
        GROQ_API_KEY=["fallback-first", "fallback-second"],
    )
    primary = ProviderModelTarget(
        "anthropic", "primary-model", "anthropic/primary-model"
    )
    fallback = ProviderModelCandidate(
        ProviderModelTarget("groq", "fallback-model", "groq/fallback-model")
    )

    async def resolver(provider_id: str, key_number: int) -> ProviderPort:
        raise AssertionError("candidate expansion must not open providers")

    executor = ProviderExecutor(resolver, settings=settings)
    candidates = executor._rotation_candidates(
        ResolvedModelRoute("test", primary, (fallback,), ReasoningPreference.INHERIT)
    )
    assert [
        (item.target.provider_id, item.target.api_key_number) for item in candidates
    ] == [("anthropic", 1), ("groq", 2)]
    assert not candidates[1].is_key_rotation


def test_non_quota_failure_does_not_mark_key():
    registry = KeyRotationRegistry()
    registry.mark(
        "p", 1, "secret", ExecutionFailure(FailureKind.PERMISSION, 403, "denied", False)
    )
    assert registry.available("p", 1, "secret")


def test_groq_413_tpm_rate_limit_is_quota_failure() -> None:
    upstream = UpstreamError(
        "Request too large for model in tokens per minute (TPM): "
        "Limit 8000, Requested 52172",
        status=413,
        body=(
            '{"message":"tokens per minute limit exceeded",'
            '"type":"tokens","code":"rate_limit_exceeded"}'
        ),
    )
    failure = ExecutionFailure(
        FailureKind.INVALID_REQUEST,
        413,
        "Provider rejected the request as too large.",
        False,
    )
    failure.__cause__ = upstream

    assert is_quota_failure(failure)


def test_plain_413_payload_too_large_is_not_quota_failure() -> None:
    upstream = UpstreamError(
        "Payload exceeds the maximum request body size",
        status=413,
        body='{"message":"payload too large","code":"request_too_large"}',
    )
    failure = ExecutionFailure(
        FailureKind.INVALID_REQUEST,
        413,
        "Provider rejected the request as too large.",
        False,
    )
    failure.__cause__ = upstream

    assert not is_quota_failure(failure)


@pytest.mark.asyncio
async def test_status_reads_persisted_cooldown_with_bound_database(
    database_factory: DatabaseFactory, tmp_path: Path
) -> None:
    database = database_factory(tmp_path / "fcc.db", tmp_path / "fcc.lock")
    await database.start()
    registry = KeyRotationRegistry()
    registry.bind_database(database)
    registry.mark("groq", 1, "secret", http_error(429), fallback_seconds=600)
    registry.clear()  # force the read through SQLite rather than memory
    rows = registry.status(Settings(GROQ_API_KEY=["secret", "other"]))
    assert [row["status"] for row in rows] == ["cooldown", "available"]
    cooldown = rows[0]["cooldown_seconds"]
    assert isinstance(cooldown, int)
    assert 0 < cooldown <= 600
    assert all("secret" not in str(row) for row in rows)
    registry.bind_database(None)
