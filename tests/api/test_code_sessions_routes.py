import asyncio
import uuid

import httpx
import pytest
import pytest_asyncio

from free_claude_code.application.code_sessions import CodeService
from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from tests.api.support import create_test_app
from tests.code_sessions_support import CodexPackets, FakeHarness


@pytest_asyncio.fixture
async def code_api(tmp_path):
    harness = FakeHarness()
    code = CodeService(
        SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock"), harness
    )
    await code.start()
    app = create_test_app(code=code)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            yield client, code, harness, tmp_path, app
    finally:
        await code.close()
        await app.state.services.admin.close()


async def create_session(code_api):
    client, _, _, folder, _ = code_api
    response = await client.post(
        "/admin/api/code/sessions",
        json={"session_id": str(uuid.uuid4()), "cwd": str(folder), "harness": "codex"},
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.asyncio
async def test_detail_exposes_child_review_liveness_without_native_source(code_api):
    client, code, harness, _, _ = code_api
    session = await create_session(code_api)
    path = f"/admin/api/code/sessions/{session['id']}"
    await code.send(
        session["id"],
        str(uuid.uuid4()),
        session["revision"],
        "Delegate",
        expected_epoch=code.epoch,
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    connection = harness.connections[0]
    packets = CodexPackets(connection)
    await packets.spawn()
    await packets.review()
    await connection.finish("turn-1")
    detail = (await client.get(path)).json()
    review = detail["items"][-1]
    assert detail.get("active_review_ids") == [review["id"]]
    assert review["kind"] == "subagent_auto_review"
    assert not {"raw", "native_turn_id", "native_item_id", "generation"} & review.keys()
    await connection.close()
    closed = (await client.get(path)).json()
    assert closed["active_review_ids"] == []
    assert closed["items"] == detail["items"]


@pytest.mark.asyncio
async def test_mode_is_saved_but_native_defaults_stay_private(code_api):
    client, code, harness, _, _ = code_api
    harness.configurations["other/vendor/model"] = "other"
    session = await create_session(code_api)
    path = f"/admin/api/code/sessions/{session['id']}"
    changed = await client.patch(
        path,
        json={
            "expected_revision": session["revision"],
            "mode": "auto_review",
            "model": "other/vendor/model",
        },
    )
    assert changed.status_code == 200
    assert changed.json()["mode"] == "auto_review"
    assert changed.json()["provider_id"] == "other"
    assert changed.json()["model_name"] == "vendor/model"
    for mode in (None, "plan", "never"):
        rejected = await client.patch(
            path, json={"expected_revision": changed.json()["revision"], "mode": mode}
        )
        assert rejected.status_code == 422
    await code.send(
        session["id"],
        str(uuid.uuid4()),
        changed.json()["revision"],
        "hello",
        expected_epoch=code.epoch,
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    detail = (await client.get(path)).json()
    assert detail["run"]["mode"] == "auto_review"
    assert "native_permission_defaults" not in detail["session"]
    assert (await code.get_detail(session["id"])).session.native_permission_defaults
    rejected = await client.patch(
        path,
        json={
            "expected_revision": detail["session"]["revision"],
            "native_permission_defaults": {},
        },
    )
    assert rejected.status_code == 422


@pytest.mark.asyncio
async def test_code_library_has_one_harness_and_no_native_work_on_open(code_api):
    client, _, harness, _, _ = code_api
    bootstrap = await client.get("/admin/api/code/bootstrap")
    assert bootstrap.json()["harnesses"] == [{"id": "codex", "name": "Codex"}]
    session = await create_session(code_api)
    detail = await client.get(f"/admin/api/code/sessions/{session['id']}")
    assert detail.status_code == 200
    assert detail.headers["cache-control"] == "no-store"
    assert detail.json()["items"] == []
    assert harness.connections == []
    page = await client.get(f"/admin/code/{session['id']}")
    assert page.status_code == 200


@pytest.mark.asyncio
async def test_context_usage_and_raw_provider_capacity_are_public_session_state(
    code_api,
):
    client, code, harness, _, _ = code_api
    harness.context_windows[harness.model] = 100_000
    harness.configurations["unknown/model"] = "unknown"
    bootstrap = (await client.get("/admin/api/code/bootstrap")).json()
    models = {model["id"]: model for model in bootstrap["models"]}
    assert models[harness.model]["context_window_tokens"] == 100_000
    assert models["unknown/model"]["context_window_tokens"] is None

    session = await create_session(code_api)
    assert session["context_used_tokens"] is None
    await code.send(
        session["id"],
        str(uuid.uuid4()),
        session["revision"],
        "hello",
        expected_epoch=code.epoch,
    )
    await harness.started.wait()
    await harness.connections[0].context_usage("turn-1", 12_438)

    detail = (await client.get(f"/admin/api/code/sessions/{session['id']}")).json()
    assert detail["session"]["context_used_tokens"] == 12_438
    library = (await client.get("/admin/api/code/sessions")).json()
    assert library["sessions"][0]["context_used_tokens"] == 12_438


@pytest.mark.asyncio
async def test_http_send_competition_and_safe_item_projection(code_api):
    client, code, harness, _, _ = code_api
    session = await create_session(code_api)
    path = f"/admin/api/code/sessions/{session['id']}"
    responses = await asyncio.gather(
        *(
            client.post(
                path + "/turns",
                json={
                    "operation_id": str(uuid.uuid4()),
                    "expected_revision": session["revision"],
                    "expected_epoch": code.epoch,
                    "text": text,
                },
            )
            for text in ("one", "two")
        )
    )
    assert sorted(response.status_code for response in responses) == [202, 409]
    await harness.started.wait()
    await harness.connections[0].text(
        "turn-1", "answer", "**safe** <script>bad()</script>", complete=True
    )
    await harness.connections[0].finish("turn-1")
    detail = (await client.get(path)).json()
    item = detail["items"][-1]
    assert "<strong>safe</strong>" in item["html"]
    assert "<script>" not in item["html"]
    assert "raw" not in item
    assert detail["run"]["status"] == "completed"


@pytest.mark.asyncio
async def test_code_commands_reject_unplanned_fields_and_other_harnesses(code_api):
    client, code, _, folder, _ = code_api
    response = await client.post(
        "/admin/api/code/sessions",
        json={"session_id": str(uuid.uuid4()), "cwd": str(folder), "harness": "cline"},
    )
    assert response.status_code == 422
    session = await create_session(code_api)
    response = await client.post(
        f"/admin/api/code/sessions/{session['id']}/turns",
        json={
            "operation_id": str(uuid.uuid4()),
            "expected_revision": session["revision"],
            "expected_epoch": code.epoch,
            "text": "hi",
            "model": "override",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers", [{"host": "remote.example"}, {"origin": "https://remote.example"}]
)
async def test_code_routes_share_admin_access_boundary(code_api, headers):
    client, _, harness, _, _ = code_api
    for path in (
        "/admin/code",
        "/admin/api/code/bootstrap",
        "/admin/api/code/sessions",
        "/admin/api/code/events",
    ):
        response = await client.get(path, headers=headers)
        assert response.status_code == 403
    assert harness.connections == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_folder_selection_only_returns_to_requesting_form(
    code_api, monkeypatch, cancelled
):
    client, _, harness, folder, app = code_api
    selected = None if cancelled else str(folder)
    hints = []

    async def pick(initial_path):
        hints.append(initial_path)
        return selected

    monkeypatch.setattr(app.state.services.admin, "pick_folder", pick)
    response = await client.post(
        "/admin/api/code/folder-picker", json={"initial_path": str(folder)}
    )
    assert response.status_code == 200
    assert response.json() == {"path": selected}
    assert response.headers["cache-control"] == "no-store"
    assert hints == [str(folder)]
    assert (await client.get("/admin/api/code/sessions")).json()["sessions"] == []
    assert harness.connections == []


@pytest.mark.asyncio
async def test_folder_picker_errors_use_admin_detail(code_api, monkeypatch):
    client, _, _, _, app = code_api

    async def pick(_initial_path):
        raise ApplicationUnavailableError("A folder picker is already open")

    monkeypatch.setattr(app.state.services.admin, "pick_folder", pick)
    response = await client.post("/admin/api/code/folder-picker", json={})
    assert response.status_code == 503
    assert response.json() == {"detail": "A folder picker is already open"}


@pytest.mark.asyncio
async def test_folder_picker_only_opens_on_an_authorized_explicit_post(
    code_api, monkeypatch
):
    client, _, _, _, app = code_api

    async def unexpected(_initial_path):
        pytest.fail("Unauthorized or passive request opened a dialog")

    monkeypatch.setattr(app.state.services.admin, "pick_folder", unexpected)
    for headers in (
        {"host": "remote.example"},
        {"origin": "https://remote.example"},
    ):
        response = await client.post(
            "/admin/api/code/folder-picker", json={}, headers=headers
        )
        assert response.status_code == 403
    for payload in ({"initial_path": "x" * 4097}, {"other": True}):
        response = await client.post("/admin/api/code/folder-picker", json=payload)
        assert response.status_code == 422
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("192.0.2.1", 1234)),
        base_url="http://127.0.0.1",
    ) as remote:
        assert (
            await remote.post("/admin/api/code/folder-picker", json={})
        ).status_code == 403
    assert (await client.get("/admin/api/code/folder-picker")).status_code == 405
    assert (await client.get("/admin/code")).status_code == 200
    assert (await client.get("/admin/api/code/bootstrap")).status_code == 200
