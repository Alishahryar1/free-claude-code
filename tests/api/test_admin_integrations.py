import asyncio
import json
import threading

import httpx
import pytest

from free_claude_code.application.integrations import IntegrationError
from free_claude_code.application.integrations import IntegrationId as Item
from free_claude_code.runtime.integrations.discovery import LocalInstallations
from free_claude_code.runtime.integrations.service import IntegrationService
from tests.api.support import create_test_app, runtime_for_app
from tests.integration_support import connection, installed_clients


@pytest.fixture
def integration_app(tmp_path, monkeypatch):
    locator = installed_clients(tmp_path)
    monkeypatch.setattr(LocalInstallations, "current", classmethod(lambda cls: locator))
    app = create_test_app(connection(tmp_path).settings)
    return app, locator


@pytest.mark.asyncio
async def test_inspect_preview_apply_and_conflict_are_local_no_store(integration_app):
    app, locator = integration_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        response = await client.get("/admin/api/integrations")
        assert response.status_code == 200
        assert "no-store" in response.headers["cache-control"]
        assert len(response.json()["items"]) == 4
        preview = await client.post(
            "/admin/api/integrations/claude-vscode/preview", json={}
        )
        assert preview.status_code == 200
        assert "fixture-proxy-token" not in preview.text
        assert not locator.vscode_settings.exists()
        payload = {"revision": preview.json()["revision"]}
        applied = await client.post(
            "/admin/api/integrations/claude-vscode/apply", json=payload
        )
        assert applied.status_code == 200
        assert applied.json()["applied"] is True
        assert (
            json.loads(locator.vscode_settings.read_bytes())[
                "claudeCode.disableLoginPrompt"
            ]
            is True
        )
        repeated = await client.post(
            "/admin/api/integrations/claude-vscode/apply", json=payload
        )
        assert repeated.status_code == 409
        assert "no-store" in repeated.headers["cache-control"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_host,headers",
    [
        ("192.0.2.10", {}),
        ("127.0.0.1", {"host": "evil.example"}),
        ("127.0.0.1", {"origin": "https://evil.example"}),
    ],
)
async def test_integration_endpoints_enforce_loopback(
    integration_app, client_host, headers
):
    app, locator = integration_app
    transport = httpx.ASGITransport(app=app, client=(client_host, 1234))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://localhost", headers=headers
    ) as client:
        for method, path, payload in [
            ("GET", "", None),
            ("POST", "/claude-login/preview", {}),
            ("POST", "/claude-login/apply", {"revision": "a" * 64}),
        ]:
            response = await client.request(
                method, "/admin/api/integrations" + path, json=payload
            )
            assert response.status_code == 403
            assert "no-store" in response.headers["cache-control"]
    assert not (locator.home / ".claude.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item,payload,status",
    [
        ("anything", {}, 422),
        ("codex", {"action": "disconnect"}, 422),
        ("codex", {"action": "recover"}, 422),
        ("codex", {"action": "execute"}, 422),
        ("claude-login", {"action": "setup"}, 422),
        ("codex", {"action": "repair"}, 422),
        ("claude-login", {"action": "repair", "path": "arbitrary.json"}, 422),
    ],
)
async def test_fixed_items_actions_and_payloads(integration_app, item, payload, status):
    app, locator = integration_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        response = await client.post(
            f"/admin/api/integrations/{item}/preview", json=payload
        )
        assert response.status_code == status
        assert "no-store" in response.headers["cache-control"]
    assert not (locator.home / ".claude.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [{"action": "disconnect"}, {"action": "recover"}, {"path": "arbitrary.json"}],
)
async def test_apply_rejects_retired_actions_even_with_a_valid_revision(
    integration_app, extra
):
    app, locator = integration_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        preview = await client.post(
            "/admin/api/integrations/claude-login/preview", json={}
        )
        assert preview.status_code == 200
        response = await client.post(
            "/admin/api/integrations/claude-login/apply",
            json={"revision": preview.json()["revision"], **extra},
        )
        assert response.status_code == 422
        assert "no-store" in response.headers["cache-control"]
    assert not (locator.home / ".claude.json").exists()


@pytest.mark.asyncio
async def test_pending_settings_block_connection_setup_but_allow_onboarding_repair(
    integration_app,
):
    app, _locator = integration_app
    runtime = runtime_for_app(app)
    runtime._pending_fields = ["PORT"]
    with pytest.raises(IntegrationError, match="Restart"):
        await runtime.preview_integration(Item.CLAUDE_VSCODE)
    preview = await runtime.preview_integration(Item.CLAUDE_LOGIN)
    assert preview["writes_file"]
    runtime.begin_shutdown()
    with pytest.raises(IntegrationError, match="shutting down"):
        await runtime.preview_integration(Item.CLAUDE_LOGIN)


@pytest.mark.asyncio
async def test_cancelled_integration_write_settles_under_configuration_lock(
    integration_app, monkeypatch
):
    app, locator = integration_app
    runtime = runtime_for_app(app)
    preview = await runtime.preview_integration(Item.CLAUDE_LOGIN)
    entered, release = threading.Event(), threading.Event()
    actual = IntegrationService.apply

    def wait_then_apply(self, *args):
        entered.set()
        assert release.wait(timeout=5)
        return actual(self, *args)

    monkeypatch.setattr(IntegrationService, "apply", wait_then_apply)
    revision = preview["revision"]
    assert isinstance(revision, str)
    task = asyncio.create_task(runtime.apply_integration(Item.CLAUDE_LOGIN, revision))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert runtime._config_lock.locked()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (
        json.loads((locator.home / ".claude.json").read_bytes())[
            "hasCompletedOnboarding"
        ]
        is True
    )
    assert not runtime._config_lock.locked()
