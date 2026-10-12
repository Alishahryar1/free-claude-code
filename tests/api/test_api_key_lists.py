"""Apply checks explicit rows and commits only a fresh, entirely valid draft."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.providers.credential_validation import (
    CredentialCheck,
    CredentialStatus,
)
from tests.api.support import create_test_app, provider_manager_for_app, runtime_for_app


def client_for(app):
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))


def edit(client, key, entries):
    field = next(
        field
        for field in client.get("/admin/api/config").json()["fields"]
        if field["key"] == key
    )
    return {"revision": field["value"]["revision"], "entries": entries}


def test_rejected_inactive_key_prevents_all_writes_and_checks_keep_row_identity(
    monkeypatch,
):
    app = create_test_app()
    client = client_for(app)
    store = ManagedConfigStore()
    before = store.path.read_bytes()
    checks = AsyncMock(
        return_value=(
            CredentialCheck("GROQ_API_KEY", CredentialStatus.VERIFIED, "Accepted", 0),
            CredentialCheck("GROQ_API_KEY", CredentialStatus.REJECTED, "Rejected", 1),
        )
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.application.check_credentials", checks
    )
    payload = {
        "values": {
            "GROQ_API_KEY": edit(
                client,
                "GROQ_API_KEY",
                [{"value": "active-private"}, {"value": "inactive-private"}],
            ),
            "HTTP_READ_TIMEOUT": "181",
        }
    }
    response = client.post("/admin/api/config/apply", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert not body["applied"]
    assert [check["index"] for check in body["credential_checks"]] == [0, 1]
    assert store.path.read_bytes() == before
    assert provider_manager_for_app(app).current_generation_id == 1
    assert checks.await_args is not None
    assert (
        checks.await_args.args[1][0].credentials[1].get_secret_value()
        == "inactive-private"
    )
    assert "active-private" not in response.text
    assert "inactive-private" not in response.text


def test_stale_revision_returns_conflict_without_checking_or_writing(monkeypatch):
    app = create_test_app()
    client = client_for(app)
    checks = AsyncMock(return_value=())
    monkeypatch.setattr(
        "free_claude_code.runtime.application.check_credentials", checks
    )
    stale = edit(client, "OPENAI_API_KEY", [{"value": "old-draft-private"}])
    current = edit(client, "OPENAI_API_KEY", [{"value": "saved-private"}])
    assert client.post(
        "/admin/api/config/apply", json={"values": {"OPENAI_API_KEY": current}}
    ).json()["applied"]
    before = ManagedConfigStore().path.read_bytes()
    checks.reset_mock()
    response = client.post(
        "/admin/api/config/apply", json={"values": {"OPENAI_API_KEY": stale}}
    )
    assert response.status_code == 409
    checks.assert_not_awaited()
    assert ManagedConfigStore().path.read_bytes() == before
    assert "old-draft-private" not in response.text


@pytest.mark.asyncio
async def test_intervening_managed_write_during_probe_cannot_publish_or_overwrite(
    monkeypatch,
):
    app = create_test_app()
    store = ManagedConfigStore()
    started, release = asyncio.Event(), asyncio.Event()

    async def held_checks(settings, submissions):
        started.set()
        await release.wait()
        return ()

    monkeypatch.setattr(
        "free_claude_code.runtime.application.check_credentials", held_checks
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
    ) as client:
        field = next(
            field
            for field in (await client.get("/admin/api/config")).json()["fields"]
            if field["key"] == "OPENAI_API_KEY"
        )
        payload = {
            "values": {
                "OPENAI_API_KEY": {
                    "revision": field["value"]["revision"],
                    "entries": [{"value": "probe-private"}],
                }
            }
        }
        request = asyncio.create_task(
            client.post("/admin/api/config/apply", json=payload)
        )
        await asyncio.wait_for(started.wait(), 5)
        store.commit(dict(store.read().managed) | {"HTTP_READ_TIMEOUT": "199"})
        before = store.path.read_bytes()
        release.set()
        response = await request
        assert response.status_code == 409
        assert store.path.read_bytes() == before
        assert provider_manager_for_app(app).current_generation_id == 1
        assert "probe-private" not in response.text
    await runtime_for_app(app).close()


def test_malformed_custom_edit_does_not_echo_nested_credential_inputs():
    app = create_test_app()
    response = client_for(app).post(
        "/admin/api/config/apply",
        json={
            "custom_provider": {
                "values": {
                    "api_keys": {
                        "revision": None,
                        "entries": [{"value": "malformed-private-sentinel"}],
                    }
                }
            }
        },
    )
    assert response.status_code == 422
    assert "malformed-private-sentinel" not in response.text
    assert "input" not in json.dumps(response.json())
