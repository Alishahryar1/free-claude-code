"""Unit tests for UsageSqliteStore persistence and analytics."""

import tempfile
import time
from pathlib import Path

import pytest

from free_claude_code.application.usage.models import UsageRecord
from free_claude_code.runtime.usage_sqlite import UsageSqliteStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_usage.db"
        usage_store = UsageSqliteStore(db_path)
        yield usage_store


@pytest.mark.asyncio
async def test_sqlite_usage_store_record_and_summary(store):
    now = time.time()
    rec1 = UsageRecord(
        request_id="req-1",
        timestamp=now - 10,
        wire_api="messages",
        gateway_model="claude-3-5-sonnet",
        provider_id="groq",
        provider_model="llama-3.3-70b-versatile",
        provider_model_ref="groq:llama-3.3-70b-versatile",
        is_primary=True,
        status="success",
        http_status=200,
        duration_ms=450.0,
        ttft_ms=120.0,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cached_tokens=20,
        reasoning_tokens=10,
        tokens_per_second=111.1,
    )
    rec2 = UsageRecord(
        request_id="req-2",
        timestamp=now - 5,
        wire_api="messages",
        gateway_model="claude-3-5-sonnet",
        provider_id="groq",
        provider_model="llama-3.3-70b-versatile",
        provider_model_ref="groq:llama-3.3-70b-versatile",
        is_primary=True,
        status="failed",
        http_status=429,
        duration_ms=200.0,
        input_tokens=80,
        output_tokens=0,
        total_tokens=80,
    )
    await store.insert_record(rec1)
    await store.insert_record(rec2)

    summary = await store.query_summary(since=now - 86400, time_range_label="24h")
    assert summary.total_requests == 2
    assert summary.successful_requests == 1
    assert summary.failed_requests == 1
    assert summary.input_tokens == 180
    assert summary.output_tokens == 50
    assert summary.total_tokens == 230
    assert summary.cached_tokens == 20
    assert summary.reasoning_tokens == 10
    assert summary.avg_duration_ms > 0
    assert summary.avg_ttft_ms == 120.0


@pytest.mark.asyncio
async def test_sqlite_usage_store_model_and_provider_breakdown(store):
    now = time.time()
    await store.insert_record(
        UsageRecord(
            request_id="r1",
            timestamp=now - 20,
            wire_api="messages",
            gateway_model="claude-3-5-sonnet",
            provider_id="groq",
            provider_model="llama-3.3-70b-versatile",
            provider_model_ref="groq:llama-3.3-70b-versatile",
            is_primary=True,
            status="success",
            duration_ms=300.0,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            tokens_per_second=666.0,
        )
    )
    await store.insert_record(
        UsageRecord(
            request_id="r2",
            timestamp=now - 10,
            wire_api="messages",
            gateway_model="claude-3-5-sonnet",
            provider_id="ollama",
            provider_model="qwen2.5-coder:7b",
            provider_model_ref="ollama:qwen2.5-coder:7b",
            is_primary=False,
            status="success",
            duration_ms=600.0,
            input_tokens=100,
            output_tokens=100,
            total_tokens=200,
            tokens_per_second=166.0,
        )
    )

    models = await store.query_models(since=now - 86400)
    assert len(models) == 2
    model_refs = {m.model_ref for m in models}
    assert "groq:llama-3.3-70b-versatile" in model_refs
    assert "ollama:qwen2.5-coder:7b" in model_refs

    providers = await store.query_providers(since=now - 86400)
    assert len(providers) == 2
    prov_ids = {p.provider_id for p in providers}
    assert "groq" in prov_ids
    assert "ollama" in prov_ids
    ollama_stat = next(p for p in providers if p.provider_id == "ollama")
    assert ollama_stat.is_local is True


@pytest.mark.asyncio
async def test_sqlite_usage_store_fallback_analytics(store):
    now = time.time()
    # Primary failed
    await store.insert_record(
        UsageRecord(
            request_id="req-fb-1",
            timestamp=now - 10,
            wire_api="messages",
            gateway_model="claude-3-5-sonnet",
            provider_id="groq",
            provider_model="llama-3.3-70b-versatile",
            provider_model_ref="groq:llama-3.3-70b-versatile",
            is_primary=True,
            status="failed",
            http_status=429,
            duration_ms=150.0,
        )
    )
    # Fallback candidate succeeded
    await store.insert_record(
        UsageRecord(
            request_id="req-fb-1",
            timestamp=now - 8,
            wire_api="messages",
            gateway_model="claude-3-5-sonnet",
            provider_id="ollama",
            provider_model="qwen2.5-coder:7b",
            provider_model_ref="ollama:qwen2.5-coder:7b",
            is_primary=False,
            fallback_index=1,
            fallback_from_ref="groq:llama-3.3-70b-versatile",
            fallback_reason="rate_limit",
            status="success",
            duration_ms=500.0,
        )
    )

    analytics = await store.query_fallbacks(since=now - 86400)
    assert analytics.primary_requests == 1
    assert analytics.primary_fallbacks == 1
    assert analytics.primary_successes == 0
    assert analytics.fallback_success_rate == 1.0
    assert len(analytics.destinations) == 1
    assert analytics.destinations[0].target_ref == "ollama:qwen2.5-coder:7b"
    assert analytics.destinations[0].triggered_count == 1
    assert analytics.destinations[0].succeeded_count == 1
    assert len(analytics.reasons) == 1
    assert analytics.reasons[0][0] == "rate_limit"


@pytest.mark.asyncio
async def test_sqlite_usage_store_prune_and_clear(store):
    now = time.time()
    old_time = now - (35 * 86400)  # 35 days ago
    await store.insert_record(
        UsageRecord(
            request_id="old-req",
            timestamp=old_time,
            wire_api="messages",
            gateway_model="old-model",
            provider_id="groq",
            provider_model="old-model",
            provider_model_ref="groq:old-model",
            is_primary=True,
            status="success",
        )
    )
    await store.insert_record(
        UsageRecord(
            request_id="new-req",
            timestamp=now - 100,
            wire_api="messages",
            gateway_model="new-model",
            provider_id="groq",
            provider_model="new-model",
            provider_model_ref="groq:new-model",
            is_primary=True,
            status="success",
        )
    )

    # Before prune
    summary = await store.query_summary(since=0, time_range_label="all")
    assert summary.total_requests == 2

    # Prune 30 days
    await store.purge_retention(retention_days=30)
    summary_after_prune = await store.query_summary(since=0, time_range_label="all")
    assert summary_after_prune.total_requests == 1

    # Clear all
    await store.clear_all()
    summary_after_clear = await store.query_summary(since=0, time_range_label="all")
    assert summary_after_clear.total_requests == 0
