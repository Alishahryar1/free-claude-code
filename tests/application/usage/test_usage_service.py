"""Unit tests for UsageService orchestration and rate limit caching."""

import tempfile
import time
from pathlib import Path

import pytest

from free_claude_code.application.usage.models import UsageRecord
from free_claude_code.application.usage.service import UsageService
from free_claude_code.runtime.usage_sqlite import SQLiteUsageStore


@pytest.fixture
def service():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_svc_usage.db"
        store = SQLiteUsageStore(db_path)
        svc = UsageService(store=store)
        yield svc


@pytest.mark.asyncio
async def test_usage_service_header_updates_and_caching(service):
    headers = {
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "990",
        "x-ratelimit-limit-tokens": "50000",
        "x-ratelimit-remaining-tokens": "49500",
    }
    await service.update_provider_headers("groq", headers)
    snapshot = await service.get_rate_limit("groq")
    assert snapshot is not None
    assert snapshot.quota_available is True
    assert snapshot.requests_limit == 1000
    assert snapshot.requests_remaining == 990


@pytest.mark.asyncio
async def test_usage_service_record_and_breakdowns(service):
    now = time.time()
    await service.record_usage(
        UsageRecord(
            request_id="svc-req-1",
            timestamp=now - 5,
            wire_api="messages",
            gateway_model="claude-3-5-sonnet",
            provider_id="groq",
            provider_model="llama-3.3-70b-versatile",
            provider_model_ref="groq:llama-3.3-70b-versatile",
            is_primary=True,
            status="success",
            duration_ms=400.0,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
    )

    summary = await service.get_summary("24h")
    assert summary.total_requests == 1
    assert summary.total_tokens == 150

    providers = await service.get_provider_breakdown("24h")
    assert len(providers) == 1
    assert providers[0].provider_id == "groq"

    models = await service.get_model_breakdown("24h")
    assert len(models) == 1
    assert models[0].model_ref == "groq:llama-3.3-70b-versatile"
