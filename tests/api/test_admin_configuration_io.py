import asyncio
import threading
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from anyio import to_thread

from free_claude_code.config.loader import ManagedConfigStore
from tests.api.support import create_test_app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["config", "status", "providers/local-status", "config/apply"]
)
async def test_admin_storage_wait_allows_health_and_authenticated_requests(path):
    app = create_test_app()
    entered, release = threading.Event(), threading.Event()
    read = ManagedConfigStore.read
    worker_limiter = to_thread.current_default_thread_limiter()
    original_tokens = worker_limiter.total_tokens

    def blocked_read(store, *args):
        entered.set()
        assert release.wait(5), "Admin blocked the event loop"
        return read(store, *args)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 50000)),
        base_url="http://127.0.0.1",
    ) as client:

        async def health_during_read():
            while not entered.is_set():
                await asyncio.sleep(0)
            try:
                assert (await client.get("/health")).status_code == 200
                response = await asyncio.wait_for(client.get("/"), timeout=1)
                assert response.status_code == 200
            finally:
                release.set()

        observer = asyncio.create_task(health_during_read())
        try:
            # Model the last shared worker being occupied by configuration I/O.
            worker_limiter.total_tokens = 1
            with (
                patch.object(ManagedConfigStore, "read", blocked_read),
                patch(
                    "free_claude_code.api.admin_routes._check_local_provider",
                    AsyncMock(return_value={}),
                ),
            ):
                if path == "config/apply":
                    response = await client.post(
                        f"/admin/api/{path}", json={"values": {"PORT": None}}
                    )
                else:
                    response = await client.get(f"/admin/api/{path}")
                assert response.status_code == 200
            await asyncio.wait_for(observer, 5)
        finally:
            release.set()
            observer.cancel()
            worker_limiter.total_tokens = original_tokens
