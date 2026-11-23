import asyncio
import uuid
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from free_claude_code.api.dependencies import app_services
from free_claude_code.application.code_sessions import CodeService
from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from tests.api.support import create_test_app, runtime_for_app
from tests.code_sessions_support import CodexPackets, FakeHarness, close_code_database
from tests.fixture_types import DatabaseFactory
from tests.wire_payload import (
    array_payload,
    int_payload,
    object_array,
    object_payload,
    response_object,
    response_payload,
    string_payload,
)


@pytest_asyncio.fixture
async def code_api(database_factory: DatabaseFactory, tmp_path: Path):
    harness = FakeHarness()
    code = CodeService(
        SQLiteCodeStore(database_factory(tmp_path / "code.db", tmp_path / "code.lock")),
        harness,
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
        await close_code_database(code)
        await runtime_for_app(app).close()


async def create_session(
    code_api: tuple[httpx.AsyncClient, CodeService, FakeHarness, Path, FastAPI],
):
    client, _, _, folder, _ = code_api
    response = await client.post(
        "/admin/api/code/sessions",
        json={"session_id": str(uuid.uuid4()), "cwd": str(folder), "harness": "codex"},
    )
    assert response.status_code == 201
    return response_payload(response)


@pytest.mark.asyncio
async def test_detail_exposes_child_review_liveness_without_native_source(
    code_api: tuple[httpx.AsyncClient, CodeService, FakeHarness, Path, FastAPI],
) -> None:
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
    detail = response_object(await client.get(path))
    review = array_payload(detail["items"])[-1]
    assert detail.get("active_review_ids") == [object_payload(review)["id"]]
    assert object_payload(review)["kind"] == "subagent_auto_review"
    assert (
        not {"raw", "native_turn_id", "native_item_id", "generation"}
        & object_payload(review).keys()
    )
    await connection.close()
    closed = response_object(await client.get(path))
    assert closed["active_review_ids"] == []
    assert closed["items"] == detail["items"]


@pytest.mark.asyncio
async def test_mode_is_saved_but_native_defaults_stay_private(
    code_api: tuple[httpx.AsyncClient, CodeService, FakeHarness, Path, FastAPI],
) -> None:
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
    assert response_object(changed)["mode"] == "auto_review"
    assert response_object(changed)["provider_id"] == "other"
    assert response_object(changed)["model_name"] == "vendor/model"
    for mode in (None, "plan", "never"):
        rejected = await client.patch(
            path,
            json={
                "expected_revision": response_object(changed)["revision"],
                "mode": mode,
            },
        )
        assert rejected.status_code == 422
    await code.send(
        session["id"],
        str(uuid.uuid4()),
        int_payload(response_object(changed)["revision"]),
        "hello",
        expected_epoch=code.epoch,
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    detail = response_object(await client.get(path))
    assert object_payload(detail["run"])["mode"] == "auto_review"
    assert "native_permission_defaults" not in object_payload(detail["session"])
    assert (await code.get_detail(session["id"])).session.native_permission_defaults
    rejected = await client.patch(
        path,
        json={
            "expected_revision": object_payload(detail["session"])["revision"],
            "native_permission_defaults": {},
        },
    )
    assert rejected.status_code == 422


@pytest.mark.asyncio
async def test_code_library_has_one_harness_and_no_native_work_on_open(
    code_api: tuple[httpx.AsyncClient, CodeService, FakeHarness, Path, FastAPI],
) -> None:
    client, _, harness, _, _ = code_api
    bootstrap = await client.get("/admin/api/code/bootstrap")
    assert response_object(bootstrap)["harnesses"] == [{"id": "codex", "name": "Codex"}]
    session = await create_session(code_api)
    detail = await client.get(f"/admin/api/code/sessions/{session['id']}")
    assert detail.status_code == 200
    assert detail.headers["cache-control"] == "no-store"
    assert response_object(detail)["items"] == []
    assert harness.connections == []
    page = await client.get(f"/admin/code/{session['id']}")
    assert page.status_code == 200


@pytest.mark.asyncio
async def test_context_usage_and_raw_provider_capacity_are_public_session_state(
    code_api: tuple[httpx.AsyncClient, CodeService, FakeHarness, Path, FastAPI],
) -> None:
    client, code, harness, _, _ = code_api
    harness.context_windows[harness.model] = 100_000
    harness.configurations["unknown/model"] = "unknown"
    bootstrap = response_object(await client.get("/admin/api/code/bootstrap"))
    models = {
        object_payload(model)["id"]: model
        for model in array_payload(bootstrap["models"])
    }
    assert object_payload(models[harness.model])["context_window_tokens"] == 100_000
    assert object_payload(models["unknown/model"])["context_window_tokens"] is None

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

    detail = response_object(
        await client.get(f"/admin/api/code/sessions/{session['id']}")
    )
    assert object_payload(detail["session"])["context_used_tokens"] == 12_438
    library = response_object(await client.get("/admin/api/code/sessions"))
    assert object_array(library["sessions"])[0]["context_used_tokens"] == 12_438


@pytest.mark.asyncio
async def test_http_send_competition_and_safe_item_projection(
    code_api: tuple[httpx.AsyncClient, CodeService, FakeHarness, Path, FastAPI],
) -> None:
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
    detail = response_object(await client.get(path))
    item = array_payload(detail["items"])[-1]
    assert "<strong>safe</strong>" in string_payload(object_payload(item)["html"])
    assert "<script>" not in string_payload(object_payload(item)["html"])
    assert "raw" not in object_payload(item)
    assert object_payload(detail["run"])["status"] == "completed"


@pytest.mark.asyncio
async def test_code_commands_reject_unplanned_fields_and_other_harnesses(
    code_api: tuple[httpx.AsyncClient, CodeService, FakeHarness, Path, FastAPI],
) -> None:
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
async def test_code_routes_share_admin_access_boundary(
    code_api: tuple[httpx.AsyncClient, CodeService, FakeHarness, Path, FastAPI],
    headers: dict[str, str],
) -> None:
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
    code_api: tuple[httpx.AsyncClient, CodeService, FakeHarness, Path, FastAPI],
    monkeypatch: pytest.MonkeyPatch,
    cancelled: bool,
) -> None:
    client, _, harness, folder, app = code_api
    selected = None if cancelled else str(folder)
    hints = []

    async def pick(initial_path: str):
        hints.append(initial_path)
        return selected

    monkeypatch.setattr(app_services(app).admin, "pick_folder", pick)
    response = await client.post(
        "/admin/api/code/folder-picker", json={"initial_path": str(folder)}
    )
    assert response.status_code == 200
    assert response_payload(response) == {"path": selected}
    assert response.headers["cache-control"] == "no-store"
    assert hints == [str(folder)]
    assert (
        response_object(await client.get("/admin/api/code/sessions"))["sessions"] == []
    )
    assert harness.connections == []


@pytest.mark.asyncio
async def test_folder_picker_errors_use_admin_detail(
    code_api: tuple[httpx.AsyncClient, CodeService, FakeHarness, Path, FastAPI],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, _, app = code_api

    async def pick(_initial_path):
        raise ApplicationUnavailableError("A folder picker is already open")

    monkeypatch.setattr(app_services(app).admin, "pick_folder", pick)
    response = await client.post("/admin/api/code/folder-picker", json={})
    assert response.status_code == 503
    assert response_payload(response) == {"detail": "A folder picker is already open"}


@pytest.mark.asyncio
async def test_folder_picker_only_opens_on_an_authorized_explicit_post(
    code_api: tuple[httpx.AsyncClient, CodeService, FakeHarness, Path, FastAPI],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, _, app = code_api

    async def unexpected(_initial_path):
        pytest.fail("Unauthorized or passive request opened a dialog")

    monkeypatch.setattr(app_services(app).admin, "pick_folder", unexpected)
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
