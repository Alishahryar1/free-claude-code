"""Usage application service coordinating recording, adapters, and querying."""

import time
from collections.abc import Mapping

from loguru import logger

from free_claude_code.config.paths import usage_database_path
from free_claude_code.runtime.usage_sqlite import UsageSqliteStore

from .adapter import get_usage_adapter
from .models import (
    FallbackAnalytics,
    ModelUsageStats,
    ProviderUsageStats,
    RateLimitSnapshot,
    TimeSeriesPoint,
    UsageRecord,
    UsageSummary,
)


def _range_to_seconds(range_name: str) -> float:
    match range_name.lower().strip():
        case "1h":
            return 3600.0
        case "24h":
            return 86400.0
        case "7d":
            return 7 * 86400.0
        case "30d":
            return 30 * 86400.0
        case "all":
            return 3650 * 86400.0  # 10 years
        case _:
            return 86400.0


def _range_to_bucket_interval(range_name: str) -> float:
    match range_name.lower().strip():
        case "1h":
            return 60.0  # 1 minute buckets
        case "24h":
            return 1800.0  # 30 minute buckets
        case "7d":
            return 14400.0  # 4 hour buckets
        case "30d" | "all":
            return 86400.0  # 1 day buckets
        case _:
            return 1800.0


class UsageService:
    """Core domain service for usage recording, rate limit tracking, and analytics."""

    def __init__(self, store: UsageSqliteStore | None = None) -> None:
        self._store = store or UsageSqliteStore(usage_database_path())
        self._last_retention_purge: float = 0.0

    @property
    def store(self) -> UsageSqliteStore:
        return self._store

    async def _maybe_purge_retention(self) -> None:
        now = time.time()
        if now - self._last_retention_purge > 3600.0:
            self._last_retention_purge = now
            try:
                await self._store.purge_retention(30)
            except Exception as exc:
                logger.debug("USAGE_SERVICE: retention purge failed: {}", exc)

    async def record_usage(self, record: UsageRecord) -> None:
        """Asynchronously persist one completed or failed request attempt."""
        try:
            await self._store.insert_record(record)
            await self._maybe_purge_retention()
        except Exception as exc:
            logger.warning(
                "USAGE_SERVICE: could not record usage for request_id={} exc={}",
                record.request_id,
                type(exc).__name__,
            )

    async def update_provider_headers(
        self,
        provider_id: str,
        headers: Mapping[str, str],
        *,
        timestamp: float | None = None,
    ) -> RateLimitSnapshot | None:
        """Extract rate limits from upstream headers and persist snapshot."""
        try:
            adapter = get_usage_adapter(provider_id)
            snapshot = adapter.extract_rate_limits(headers, updated_at=timestamp)
            await self._store.update_quota(snapshot)
            return snapshot
        except Exception as exc:
            logger.debug(
                "USAGE_SERVICE: header extraction failed for provider={} exc={}",
                provider_id,
                type(exc).__name__,
            )
            return None

    async def get_summary(
        self,
        time_range: str = "24h",
        *,
        provider_id: str | None = None,
        model: str | None = None,
        status: str | None = None,
    ) -> UsageSummary:
        await self._maybe_purge_retention()
        now = time.time()
        delta = _range_to_seconds(time_range)
        since = 0.0 if time_range == "all" else max(0.0, now - delta)
        return await self._store.query_summary(
            since,
            now,
            time_range_label=time_range,
            provider_id=provider_id,
            model=model,
            status=status,
        )

    async def get_providers(self, time_range: str = "24h") -> list[ProviderUsageStats]:
        now = time.time()
        delta = _range_to_seconds(time_range)
        since = 0.0 if time_range == "all" else max(0.0, now - delta)
        return await self._store.query_providers(since, now)

    async def get_models(
        self,
        time_range: str = "24h",
        *,
        provider_id: str | None = None,
    ) -> list[ModelUsageStats]:
        now = time.time()
        delta = _range_to_seconds(time_range)
        since = 0.0 if time_range == "all" else max(0.0, now - delta)
        return await self._store.query_models(since, now, provider_id=provider_id)

    async def get_fallbacks(self, time_range: str = "24h") -> FallbackAnalytics:
        now = time.time()
        delta = _range_to_seconds(time_range)
        since = 0.0 if time_range == "all" else max(0.0, now - delta)
        return await self._store.query_fallbacks(since, now)

    async def get_timeseries(
        self,
        time_range: str = "24h",
        *,
        provider_id: str | None = None,
        model: str | None = None,
    ) -> list[TimeSeriesPoint]:
        now = time.time()
        delta = _range_to_seconds(time_range)
        interval = _range_to_bucket_interval(time_range)
        since = (
            max(0.0, now - delta)
            if time_range != "all"
            else max(0.0, now - (30 * 86400.0))
        )
        return await self._store.query_timeseries(
            since,
            now,
            interval_seconds=interval,
            provider_id=provider_id,
            model=model,
        )

    async def clear_history(self) -> None:
        """Purge all telemetry records."""
        await self._store.clear_all()

    async def get_rate_limit(self, provider_id: str) -> RateLimitSnapshot | None:
        quotas = await self._store.get_all_quotas()
        return quotas.get(provider_id)

    get_provider_breakdown = get_providers
    get_model_breakdown = get_models
    get_fallback_analytics = get_fallbacks
    clear_all = clear_history


_GLOBAL_USAGE_SERVICE: UsageService | None = None


def get_usage_service() -> UsageService:
    """Return the active global UsageService instance."""
    global _GLOBAL_USAGE_SERVICE
    if _GLOBAL_USAGE_SERVICE is None:
        _GLOBAL_USAGE_SERVICE = UsageService()
    return _GLOBAL_USAGE_SERVICE


def set_usage_service(service: UsageService | None) -> None:
    """Explicitly override the global UsageService instance (e.g. for testing)."""
    global _GLOBAL_USAGE_SERVICE
    _GLOBAL_USAGE_SERVICE = service
