"""Integration and security tests for Provider Usage & Observability API routes."""

import tempfile
import time
from pathlib import Path

import httpx
import pytest

from free_claude_code.application.usage.models import UsageRecord
from free_claude_code.application.usage.service import UsageService
from free_claude_code.core.version import package_version
from free_claude_code.runtime.usage_sqlite import SQLiteUsageStore
from tests.api.support import create_test_app


@pytest.fixture
def test_usage_env():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_api_usage.db"
        store = SQLiteUsageStore(db_path)
        service = UsageService(store=store)
        app = create_test_app(usage=service)
        yield app, service


@pytest.mark.asyncio
async def test_usage_routes_loopback_security(test_usage_env):
    app, _ = test_usage_env

    # Remote non-loopback client should be blocked with 403
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("203.0.113.1", 54321)),
        base_url="http://203.0.113.1",
    ) as remote_client:
        response = await remote_client.get("/admin/api/usage/summary")
        assert response.status_code == 403

    # Local loopback client should succeed
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 54321)),
        base_url="http://127.0.0.1",
    ) as local_client:
        response = await local_client.get("/admin/api/usage/summary")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_usage_api_endpoints(test_usage_env):
    app, service = test_usage_env
    now = time.time()

    # Pre-populate some records
    await service.record_usage(
        UsageRecord(
            request_id="req-test-1",
            timestamp=now - 20,
            wire_api="messages",
            gateway_model="claude-3-5-sonnet",
            provider_id="groq",
            provider_model="llama-3.3-70b-versatile",
            provider_model_ref="groq:llama-3.3-70b-versatile",
            is_primary=True,
            status="success",
            duration_ms=250.0,
            ttft_ms=90.0,
            input_tokens=150,
            output_tokens=75,
            total_tokens=225,
            cached_tokens=25,
            reasoning_tokens=10,
        )
    )

    await service.update_provider_headers(
        "groq",
        {
            "x-ratelimit-limit-requests": "14400",
            "x-ratelimit-remaining-requests": "14399",
            "x-ratelimit-limit-tokens": "18000",
            "x-ratelimit-remaining-tokens": "17900",
        },
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 54321)),
        base_url="http://127.0.0.1",
    ) as client:
        # 1. Summary
        summary_res = await client.get("/admin/api/usage/summary?time_range=24h")
        assert summary_res.status_code == 200
        summary_data = summary_res.json()
        assert summary_data["total_requests"] == 1
        assert summary_data["successful_requests"] == 1
        assert summary_data["total_tokens"] == 225
        assert summary_data["input_tokens"] == 150
        assert summary_data["output_tokens"] == 75
        assert summary_data["cached_tokens"] == 25
        assert summary_data["reasoning_tokens"] == 10
        assert summary_data["avg_ttft_ms"] == 90.0

        # 2. Providers
        providers_res = await client.get("/admin/api/usage/providers?time_range=24h")
        assert providers_res.status_code == 200
        providers_data = providers_res.json()
        assert len(providers_data["providers"]) == 1
        groq_provider = providers_data["providers"][0]
        assert groq_provider["provider_id"] == "groq"
        assert groq_provider["quota"] is not None
        assert groq_provider["quota"]["requests_remaining"] == 14399

        # 3. Models
        models_res = await client.get("/admin/api/usage/models?time_range=24h")
        assert models_res.status_code == 200
        models_data = models_res.json()
        assert len(models_data["models"]) == 1
        assert models_data["models"][0]["model_ref"] == "groq:llama-3.3-70b-versatile"

        # 4. Fallbacks
        fb_res = await client.get("/admin/api/usage/fallbacks?time_range=24h")
        assert fb_res.status_code == 200
        fb_data = fb_res.json()
        assert fb_data["primary_requests"] == 1
        assert fb_data["primary_successes"] == 1

        # 5. Timeseries
        ts_res = await client.get("/admin/api/usage/timeseries?time_range=24h")
        assert ts_res.status_code == 200
        ts_data = ts_res.json()
        assert "points" in ts_data
        assert len(ts_data["points"]) > 0

        # 6. Clear Data
        clear_res = await client.post("/admin/api/usage/clear")
        assert clear_res.status_code == 200
        assert clear_res.json() == {"status": "cleared"}

        # Verify summary is now 0
        cleared_summary = (
            await client.get("/admin/api/usage/summary?time_range=24h")
        ).json()
        assert cleared_summary["total_requests"] == 0


@pytest.mark.asyncio
async def test_admin_usage_page_and_assets(test_usage_env):
    app, _ = test_usage_env
    version = package_version()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 54321)),
        base_url="http://127.0.0.1",
    ) as client:
        # Page serving
        page_res = await client.get("/admin/usage")
        assert page_res.status_code == 200
        assert 'id="view-usage"' in page_res.text
        assert "usage.js" in page_res.text
        assert "usage.css" in page_res.text

        # Static assets
        js_res = await client.get(f"/admin/assets/{version}/usage.js")
        assert js_res.status_code == 200
        assert "UsageDashboard" in js_res.text

        css_res = await client.get(f"/admin/assets/{version}/usage.css")
        assert css_res.status_code == 200
        assert ".usage-root" in css_res.text


@pytest.mark.asyncio
async def test_security_privacy_audit(test_usage_env):
    app, service = test_usage_env
    now = time.time()

    await service.record_usage(
        UsageRecord(
            request_id="privacy-req",
            timestamp=now,
            wire_api="messages",
            gateway_model="claude-3-5-sonnet",
            provider_id="groq",
            provider_model="llama-3.3-70b-versatile",
            provider_model_ref="groq:llama-3.3-70b-versatile",
            is_primary=True,
            status="success",
            duration_ms=100.0,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 54321)),
        base_url="http://127.0.0.1",
    ) as client:
        for endpoint in [
            "/admin/api/usage/summary",
            "/admin/api/usage/providers",
            "/admin/api/usage/models",
            "/admin/api/usage/fallbacks",
            "/admin/api/usage/timeseries",
        ]:
            res = await client.get(endpoint)
            assert res.status_code == 200
            text = res.text.lower()
            # Verify no sensitive payload leakages
            assert "prompt" not in text
            assert "completion" not in text
            assert "api_key" not in text
            assert "authorization" not in text
            assert "bearer" not in text
