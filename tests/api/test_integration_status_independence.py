import asyncio
import threading

import httpx
import pytest

from free_claude_code.harnesses import vscode_chat_integration as vscode
from tests.api.support import create_test_app, runtime_for_app


@pytest.mark.asyncio
async def test_integration_endpoints_finish_while_one_worker_is_held(monkeypatch):
    app = create_test_app()
    runtime = runtime_for_app(app)
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def held(path):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return {"connected": False, "paths": {}}

    monkeypatch.setattr(vscode, "status", held)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 50000)),
        base_url="http://127.0.0.1",
    ) as client:
        task = asyncio.create_task(client.get("/admin/api/integrations/vscode-chat"))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            responses = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        client.get(f"/admin/api/integrations/{integration}")
                        for integration in [
                            "claude-vscode",
                            "claude-desktop",
                            "codex",
                            "jetbrains-acp",
                        ]
                    )
                ),
                1,
            )
            assert all(
                r.status_code == 200 and r.json()["connected"] is False
                for r in responses
            )
            assert not task.done()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await runtime.close()
