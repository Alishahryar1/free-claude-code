"""SQLite persistence for provider usage telemetry, rate limits, and analytics."""

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from free_claude_code.application.usage.models import (
    FallbackAnalytics,
    FallbackDestinationStats,
    ModelUsageStats,
    ProviderUsageStats,
    RateLimitSnapshot,
    TimeSeriesPoint,
    UsageRecord,
    UsageSummary,
)
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG


class UsageSqliteStore:
    """Thread-safe SQLite store for telemetry records and provider quota tracking."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _execute[T](self, operation: Callable[[sqlite3.Connection], T]) -> T:
        with closing(sqlite3.connect(self._path, timeout=10.0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode = WAL;")
            connection.execute("PRAGMA synchronous = NORMAL;")
            connection.execute("PRAGMA busy_timeout = 5000;")
            result = operation(connection)
            connection.commit()
            return result

    async def _run[T](self, operation: Callable[[sqlite3.Connection], T]) -> T:
        from typing import cast

        result = await asyncio.to_thread(self._execute, operation)
        return cast(T, result)

    def _initialize_schema(self) -> None:
        def op(conn: sqlite3.Connection) -> None:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_records (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    wire_api TEXT NOT NULL,
                    gateway_model TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    provider_model TEXT NOT NULL,
                    provider_model_ref TEXT NOT NULL,
                    is_primary INTEGER NOT NULL,
                    fallback_index INTEGER NOT NULL DEFAULT 0,
                    fallback_from_ref TEXT,
                    fallback_reason TEXT,
                    status TEXT NOT NULL,
                    http_status INTEGER,
                    error_category TEXT,
                    duration_ms REAL NOT NULL,
                    ttft_ms REAL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    is_estimated INTEGER NOT NULL DEFAULT 0,
                    tokens_per_second REAL
                );
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_records(timestamp);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_prov ON usage_records(provider_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_mod ON usage_records(gateway_model);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_status ON usage_records(status);"
            )

            conn.execute("""
                CREATE TABLE IF NOT EXISTS provider_quotas (
                    provider_id TEXT PRIMARY KEY,
                    updated_at REAL NOT NULL,
                    quota_available INTEGER NOT NULL,
                    status_label TEXT NOT NULL,
                    requests_limit INTEGER,
                    requests_remaining INTEGER,
                    requests_reset TEXT,
                    tokens_limit INTEGER,
                    tokens_remaining INTEGER,
                    tokens_reset TEXT,
                    retry_after INTEGER,
                    raw_headers_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_pricing (
                    model_ref TEXT PRIMARY KEY,
                    input_cost_per_m REAL,
                    output_cost_per_m REAL,
                    cached_cost_per_m REAL,
                    is_free INTEGER DEFAULT 0
                );
            """)

        self._execute(op)

    # -------------------------------------------------------------------------
    # Ingestion Methods
    # -------------------------------------------------------------------------

    async def insert_record(self, record: UsageRecord) -> None:
        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT OR REPLACE INTO usage_records (
                    id, request_id, timestamp, wire_api, gateway_model,
                    provider_id, provider_model, provider_model_ref,
                    is_primary, fallback_index, fallback_from_ref, fallback_reason,
                    status, http_status, error_category, duration_ms, ttft_ms,
                    input_tokens, output_tokens, total_tokens, cached_tokens,
                    reasoning_tokens, is_estimated, tokens_per_second
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.request_id,
                    record.timestamp,
                    record.wire_api,
                    record.gateway_model,
                    record.provider_id,
                    record.provider_model,
                    record.provider_model_ref,
                    1 if record.is_primary else 0,
                    record.fallback_index,
                    record.fallback_from_ref,
                    record.fallback_reason,
                    record.status,
                    record.http_status,
                    record.error_category,
                    record.duration_ms,
                    record.ttft_ms,
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    record.cached_tokens,
                    record.reasoning_tokens,
                    1 if record.is_estimated else 0,
                    record.tokens_per_second,
                ),
            )

        await self._run(op)

    async def update_quota(self, snapshot: RateLimitSnapshot) -> None:
        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT OR REPLACE INTO provider_quotas (
                    provider_id, updated_at, quota_available, status_label,
                    requests_limit, requests_remaining, requests_reset,
                    tokens_limit, tokens_remaining, tokens_reset,
                    retry_after, raw_headers_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.provider_id,
                    snapshot.updated_at,
                    1 if snapshot.quota_available else 0,
                    snapshot.status_label,
                    snapshot.requests_limit,
                    snapshot.requests_remaining,
                    snapshot.requests_reset,
                    snapshot.tokens_limit,
                    snapshot.tokens_remaining,
                    snapshot.tokens_reset,
                    snapshot.retry_after,
                    json.dumps(dict(snapshot.raw_headers)),
                ),
            )

        await self._run(op)

    async def get_all_quotas(self) -> dict[str, RateLimitSnapshot]:
        def op(conn: sqlite3.Connection) -> dict[str, RateLimitSnapshot]:
            cursor = conn.execute("SELECT * FROM provider_quotas")
            results: dict[str, RateLimitSnapshot] = {}
            for row in cursor.fetchall():
                try:
                    headers = json.loads(row["raw_headers_json"] or "{}")
                except Exception:
                    headers = {}
                results[row["provider_id"]] = RateLimitSnapshot(
                    provider_id=row["provider_id"],
                    updated_at=row["updated_at"],
                    quota_available=bool(row["quota_available"]),
                    status_label=row["status_label"],
                    requests_limit=row["requests_limit"],
                    requests_remaining=row["requests_remaining"],
                    requests_reset=row["requests_reset"],
                    tokens_limit=row["tokens_limit"],
                    tokens_remaining=row["tokens_remaining"],
                    tokens_reset=row["tokens_reset"],
                    retry_after=row["retry_after"],
                    raw_headers=headers,
                )
            return results

        return await self._run(op)

    # -------------------------------------------------------------------------
    # Aggregation & Query Methods
    # -------------------------------------------------------------------------

    async def query_summary(
        self,
        since: float,
        until: float | None = None,
        *,
        time_range_label: str = "24h",
        provider_id: str | None = None,
        model: str | None = None,
        status: str | None = None,
    ) -> UsageSummary:
        def op(conn: sqlite3.Connection) -> UsageSummary:
            where_clauses = ["timestamp >= ?"]
            params: list[Any] = [since]
            if until is not None:
                where_clauses.append("timestamp <= ?")
                params.append(until)
            if provider_id:
                where_clauses.append("provider_id = ?")
                params.append(provider_id)
            if model:
                where_clauses.append("(gateway_model = ? OR provider_model_ref = ?)")
                params.extend([model, model])
            if status:
                where_clauses.append("status = ?")
                params.append(status)

            where_sql = " AND ".join(where_clauses)
            cursor = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures,
                    COALESCE(SUM(input_tokens), 0) as in_tok,
                    COALESCE(SUM(output_tokens), 0) as out_tok,
                    COALESCE(SUM(total_tokens), 0) as tot_tok,
                    COALESCE(SUM(cached_tokens), 0) as cached_tok,
                    COALESCE(SUM(reasoning_tokens), 0) as reasoning_tok,
                    COALESCE(SUM(CASE WHEN is_estimated = 1 THEN total_tokens ELSE 0 END), 0) as est_tok,
                    COALESCE(SUM(CASE WHEN is_primary = 0 THEN 1 ELSE 0 END), 0) as fallbacks,
                    AVG(duration_ms) as avg_dur,
                    AVG(CASE WHEN ttft_ms IS NOT NULL THEN ttft_ms ELSE NULL END) as avg_ttft
                FROM usage_records
                WHERE {where_sql}
                """,
                params,
            )
            row = cursor.fetchone()
            if not row or not row["total"]:
                return UsageSummary(time_range=time_range_label)

            # Check pricing table for cost calculation
            cost_cursor = conn.execute("SELECT * FROM model_pricing")
            pricings = {r["model_ref"]: r for r in cost_cursor.fetchall()}

            # Calculate verified cost if models have prices
            total_cost: float | None = None
            cost_has_verified = False

            if pricings:
                detail_cursor = conn.execute(
                    f"""
                    SELECT provider_model_ref, SUM(input_tokens) as inp, SUM(output_tokens) as out, SUM(cached_tokens) as cch
                    FROM usage_records
                    WHERE {where_sql}
                    GROUP BY provider_model_ref
                    """,
                    params,
                )
                accum = 0.0
                for d_row in detail_cursor.fetchall():
                    m_ref = d_row["provider_model_ref"]
                    pricing = pricings.get(m_ref)
                    if pricing:
                        if pricing["is_free"]:
                            cost_has_verified = True
                        elif (
                            pricing["input_cost_per_m"] is not None
                            and pricing["output_cost_per_m"] is not None
                        ):
                            cost_has_verified = True
                            accum += (d_row["inp"] / 1_000_000.0) * pricing[
                                "input_cost_per_m"
                            ]
                            accum += (d_row["out"] / 1_000_000.0) * pricing[
                                "output_cost_per_m"
                            ]
                            if pricing["cached_cost_per_m"] is not None:
                                accum += (d_row["cch"] / 1_000_000.0) * pricing[
                                    "cached_cost_per_m"
                                ]
                if cost_has_verified:
                    total_cost = round(accum, 4)

            cost_label = "Unknown"
            if total_cost is not None:
                cost_label = f"${total_cost:,.4f}" if total_cost > 0 else "$0 (Free)"

            return UsageSummary(
                time_range=time_range_label,
                total_requests=row["total"] or 0,
                successful_requests=row["successes"] or 0,
                failed_requests=row["failures"] or 0,
                input_tokens=row["in_tok"] or 0,
                output_tokens=row["out_tok"] or 0,
                total_tokens=row["tot_tok"] or 0,
                cached_tokens=row["cached_tok"] or 0,
                reasoning_tokens=row["reasoning_tok"] or 0,
                estimated_tokens=row["est_tok"] or 0,
                fallback_count=row["fallbacks"] or 0,
                avg_duration_ms=round(row["avg_dur"] or 0.0, 1),
                avg_ttft_ms=round(row["avg_ttft"], 1)
                if row["avg_ttft"] is not None
                else None,
                estimated_cost=total_cost,
                cost_label=cost_label,
            )

        return await self._run(op)

    async def query_models(
        self,
        since: float,
        until: float | None = None,
        *,
        provider_id: str | None = None,
    ) -> list[ModelUsageStats]:
        def op(conn: sqlite3.Connection) -> list[ModelUsageStats]:
            where_clauses = ["timestamp >= ?"]
            params: list[Any] = [since]
            if until is not None:
                where_clauses.append("timestamp <= ?")
                params.append(until)
            if provider_id:
                where_clauses.append("provider_id = ?")
                params.append(provider_id)

            where_sql = " AND ".join(where_clauses)
            cursor = conn.execute(
                f"""
                SELECT
                    provider_model_ref,
                    provider_id,
                    provider_model,
                    COUNT(*) as requests,
                    COALESCE(SUM(input_tokens), 0) as input_tokens,
                    COALESCE(SUM(output_tokens), 0) as output_tokens,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(cached_tokens), 0) as cached_tokens,
                    COALESCE(SUM(reasoning_tokens), 0) as reasoning_tokens,
                    AVG(duration_ms) as avg_duration_ms,
                    AVG(CASE WHEN tokens_per_second IS NOT NULL THEN tokens_per_second ELSE NULL END) as avg_speed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as error_count
                FROM usage_records
                WHERE {where_sql}
                GROUP BY provider_model_ref
                ORDER BY requests DESC
                """,
                params,
            )
            rows = cursor.fetchall()
            results: list[ModelUsageStats] = []
            for r in rows:
                reqs = r["requests"] or 0
                errs = r["error_count"] or 0
                results.append(
                    ModelUsageStats(
                        model_ref=r["provider_model_ref"],
                        provider_id=r["provider_id"],
                        provider_model=r["provider_model"],
                        requests=reqs,
                        input_tokens=r["input_tokens"],
                        output_tokens=r["output_tokens"],
                        total_tokens=r["total_tokens"],
                        cached_tokens=r["cached_tokens"],
                        reasoning_tokens=r["reasoning_tokens"],
                        avg_duration_ms=round(r["avg_duration_ms"] or 0.0, 1),
                        tokens_per_second=round(r["avg_speed"], 1)
                        if r["avg_speed"] is not None
                        else None,
                        error_count=errs,
                        error_rate=round(errs / reqs, 3) if reqs else 0.0,
                    )
                )
            return results

        return await self._run(op)

    async def query_providers(
        self,
        since: float,
        until: float | None = None,
    ) -> list[ProviderUsageStats]:
        quotas = await self.get_all_quotas()

        def op(conn: sqlite3.Connection) -> list[ProviderUsageStats]:
            where_clauses = ["timestamp >= ?"]
            params: list[Any] = [since]
            if until is not None:
                where_clauses.append("timestamp <= ?")
                params.append(until)

            where_sql = " AND ".join(where_clauses)
            cursor = conn.execute(
                f"""
                SELECT
                    provider_id,
                    COUNT(*) as requests,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(input_tokens), 0) as input_tokens,
                    COALESCE(SUM(output_tokens), 0) as output_tokens,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failure_count,
                    AVG(CASE WHEN tokens_per_second IS NOT NULL THEN tokens_per_second ELSE NULL END) as avg_speed
                FROM usage_records
                WHERE {where_sql}
                GROUP BY provider_id
                ORDER BY requests DESC
                """,
                params,
            )
            rows = cursor.fetchall()
            observed_providers = {r["provider_id"] for r in rows}
            results: list[ProviderUsageStats] = []

            for r in rows:
                p_id = r["provider_id"]
                reqs = r["requests"] or 0
                fails = r["failure_count"] or 0
                desc = PROVIDER_CATALOG.get(p_id)
                display_name = desc.display_name if desc else p_id
                is_local = desc.local if desc else False

                results.append(
                    ProviderUsageStats(
                        provider_id=p_id,
                        display_name=display_name,
                        is_local=is_local,
                        requests=reqs,
                        total_tokens=r["total_tokens"],
                        input_tokens=r["input_tokens"],
                        output_tokens=r["output_tokens"],
                        success_count=r["success_count"],
                        failure_count=fails,
                        error_rate=round(fails / reqs, 3) if reqs else 0.0,
                        quota=quotas.get(p_id),
                        avg_speed_tok_s=round(r["avg_speed"], 1)
                        if r["avg_speed"] is not None
                        else None,
                    )
                )

            # Include any configured or quota-registered providers not yet seen in traffic
            for p_id, q in quotas.items():
                if p_id not in observed_providers:
                    desc = PROVIDER_CATALOG.get(p_id)
                    display_name = desc.display_name if desc else p_id
                    is_local = desc.local if desc else False
                    results.append(
                        ProviderUsageStats(
                            provider_id=p_id,
                            display_name=display_name,
                            is_local=is_local,
                            requests=0,
                            total_tokens=0,
                            input_tokens=0,
                            output_tokens=0,
                            success_count=0,
                            failure_count=0,
                            error_rate=0.0,
                            quota=q,
                            avg_speed_tok_s=None,
                        )
                    )

            return results

        return await self._run(op)

    async def query_fallbacks(
        self,
        since: float,
        until: float | None = None,
    ) -> FallbackAnalytics:
        def op(conn: sqlite3.Connection) -> FallbackAnalytics:
            where_clauses = ["timestamp >= ?"]
            params: list[Any] = [since]
            if until is not None:
                where_clauses.append("timestamp <= ?")
                params.append(until)

            where_sql = " AND ".join(where_clauses)
            cursor = conn.execute(
                f"""
                SELECT
                    COUNT(DISTINCT request_id) as total_reqs,
                    SUM(CASE WHEN is_primary = 1 AND status = 'success' THEN 1 ELSE 0 END) as prim_succ,
                    SUM(CASE WHEN is_primary = 1 AND status = 'failed' THEN 1 ELSE 0 END) as prim_fail,
                    SUM(CASE WHEN is_primary = 0 THEN 1 ELSE 0 END) as fallback_events
                FROM usage_records
                WHERE {where_sql}
                """,
                params,
            )
            row = cursor.fetchone()
            prim_succ = (row["prim_succ"] or 0) if row else 0
            prim_fail = (row["prim_fail"] or 0) if row else 0
            primary_reqs = prim_succ + prim_fail

            # Fallback destinations
            dest_cursor = conn.execute(
                f"""
                SELECT
                    provider_model_ref,
                    COUNT(*) as triggered,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as succeeded,
                    AVG(duration_ms) as avg_dur
                FROM usage_records
                WHERE {where_sql} AND is_primary = 0
                GROUP BY provider_model_ref
                ORDER BY triggered DESC
                """,
                params,
            )
            destinations: list[FallbackDestinationStats] = []
            total_fallback_attempts = 0
            total_fallback_successes = 0
            for d in dest_cursor.fetchall():
                trig = d["triggered"] or 0
                succ = d["succeeded"] or 0
                total_fallback_attempts += trig
                total_fallback_successes += succ
                destinations.append(
                    FallbackDestinationStats(
                        target_ref=d["provider_model_ref"],
                        triggered_count=trig,
                        succeeded_count=succ,
                        avg_duration_ms=round(d["avg_dur"] or 0.0, 1),
                    )
                )

            # Trigger reasons
            reasons_cursor = conn.execute(
                f"""
                SELECT fallback_reason, COUNT(*) as cnt
                FROM usage_records
                WHERE {where_sql} AND is_primary = 0 AND fallback_reason IS NOT NULL
                GROUP BY fallback_reason
                ORDER BY cnt DESC
                LIMIT 10
                """,
                params,
            )
            reasons = tuple(
                (r["fallback_reason"], r["cnt"]) for r in reasons_cursor.fetchall()
            )

            success_rate = (
                round(total_fallback_successes / total_fallback_attempts, 3)
                if total_fallback_attempts
                else 1.0
            )

            return FallbackAnalytics(
                primary_requests=primary_reqs,
                primary_successes=prim_succ,
                primary_fallbacks=prim_fail,
                fallback_success_rate=success_rate,
                destinations=tuple(destinations),
                reasons=reasons,
            )

        return await self._run(op)

    async def query_timeseries(
        self,
        since: float,
        until: float,
        interval_seconds: float,
        *,
        provider_id: str | None = None,
        model: str | None = None,
    ) -> list[TimeSeriesPoint]:
        def op(conn: sqlite3.Connection) -> list[TimeSeriesPoint]:
            where_clauses = ["timestamp >= ?", "timestamp <= ?"]
            params: list[Any] = [since, until]
            if provider_id:
                where_clauses.append("provider_id = ?")
                params.append(provider_id)
            if model:
                where_clauses.append("(gateway_model = ? OR provider_model_ref = ?)")
                params.extend([model, model])

            where_sql = " AND ".join(where_clauses)
            cursor = conn.execute(
                f"""
                SELECT
                    CAST((timestamp - {since}) / {interval_seconds} AS INTEGER) as bucket_idx,
                    COUNT(*) as requests,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(input_tokens), 0) as input_tokens,
                    COALESCE(SUM(output_tokens), 0) as output_tokens,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as errors,
                    SUM(CASE WHEN is_primary = 0 THEN 1 ELSE 0 END) as fallbacks,
                    AVG(duration_ms) as avg_duration_ms
                FROM usage_records
                WHERE {where_sql}
                GROUP BY bucket_idx
                ORDER BY bucket_idx ASC
                """,
                params,
            )
            rows = cursor.fetchall()
            buckets: dict[int, sqlite3.Row] = {r["bucket_idx"]: r for r in rows}

            total_buckets = int((until - since) / interval_seconds) + 1
            points: list[TimeSeriesPoint] = []

            for idx in range(min(total_buckets, 100)):  # capped at 100 sample points
                b_start = since + (idx * interval_seconds)
                b_row = buckets.get(idx)

                # Format label
                time_struct = time.localtime(b_start)
                if interval_seconds < 86400:
                    label = time.strftime("%H:%M", time_struct)
                else:
                    label = time.strftime("%b %d", time_struct)

                if b_row:
                    points.append(
                        TimeSeriesPoint(
                            bucket_start=b_start,
                            label=label,
                            requests=b_row["requests"] or 0,
                            total_tokens=b_row["total_tokens"] or 0,
                            input_tokens=b_row["input_tokens"] or 0,
                            output_tokens=b_row["output_tokens"] or 0,
                            errors=b_row["errors"] or 0,
                            fallbacks=b_row["fallbacks"] or 0,
                            avg_duration_ms=round(b_row["avg_duration_ms"] or 0.0, 1),
                        )
                    )
                else:
                    points.append(
                        TimeSeriesPoint(
                            bucket_start=b_start,
                            label=label,
                            requests=0,
                            total_tokens=0,
                            input_tokens=0,
                            output_tokens=0,
                            errors=0,
                            fallbacks=0,
                            avg_duration_ms=0.0,
                        )
                    )

            return points

        return await self._run(op)

    async def purge_retention(self, retention_days: int = 30) -> int:
        cutoff = time.time() - (retention_days * 86400.0)

        def op(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "DELETE FROM usage_records WHERE timestamp < ?", (cutoff,)
            )
            return cursor.rowcount

        return await self._run(op)

    async def clear_all(self) -> None:
        def op(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM usage_records;")
            conn.execute("DELETE FROM provider_quotas;")

        await self._run(op)


SQLiteUsageStore = UsageSqliteStore
