import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tests.api.support import create_test_app


@pytest.mark.asyncio
async def test_local_provider_results_complete_independently():
    app = create_test_app()
    entered, release = asyncio.Event(), asyncio.Event()

    async def check(provider_id, url, path):
        if provider_id == "ollama":
            entered.set()
            await release.wait()
        return {"provider_id": provider_id, "status": "reachable"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 50000)),
        base_url="http://127.0.0.1",
    ) as client:
        with patch(
            "free_claude_code.api.admin_routes._check_local_provider", side_effect=check
        ):
            slow = asyncio.create_task(
                client.get("/admin/api/providers/ollama/local-status")
            )
            try:
                await asyncio.wait_for(entered.wait(), 1)
                fast = await asyncio.wait_for(
                    client.get("/admin/api/providers/lmstudio/local-status"), 1
                )
                assert fast.status_code == 200
                assert fast.json()["provider_id"] == "lmstudio"
                assert not slow.done()
            finally:
                release.set()
                await asyncio.gather(slow, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["openai", "unknown"])
async def test_nonlocal_provider_status_is_rejected(provider_id):
    app = create_test_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 50000)),
        base_url="http://127.0.0.1",
    ) as client:
        with patch(
            "free_claude_code.api.admin_routes._check_local_provider",
            new_callable=AsyncMock,
        ) as check:
            response = await client.get(
                f"/admin/api/providers/{provider_id}/local-status"
            )
            assert response.status_code == 404
            check.assert_not_awaited()
