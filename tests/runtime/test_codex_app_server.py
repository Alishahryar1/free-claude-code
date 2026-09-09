import asyncio
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.code_sessions import CodeService
from free_claude_code.application.code_sessions.models import CodeUnavailableError
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from free_claude_code.runtime.codex_app_server import CodexAppServer
from tests.code_sessions_support import FakeHarness


@pytest.mark.asyncio
async def test_child_warning_during_shutdown_allows_connection_replacement(
    tmp_path, monkeypatch
):
    harness = FakeHarness()
    prepare = harness.prepare
    connections = []
    releases = []
    warning = asyncio.Event()

    def selection_for(model, effort, mode):
        selection = prepare(model, effort, mode)

        async def open_native(cwd, sink):
            release = tmp_path / f"release-{len(connections)}"
            releases.append(release)

            async def receive(event):
                await sink(event)
                if event.kind == "notice" and event.thread_id == "child":
                    warning.set()

            native = CodexAppServer(
                [
                    sys.executable,
                    str(Path(__file__).with_name("codex_fake_process.py")),
                    "child-warning-on-close",
                    str(release),
                ],
                dict(os.environ),
                cwd,
                receive,
                model_slugs={model: model},
                fingerprints=selection.catalog,
            )
            await native.start()
            connections.append(native)
            return native

        monkeypatch.setattr(selection, "open", open_native)
        return selection

    monkeypatch.setattr(harness, "prepare", selection_for)
    service = CodeService(
        SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock"), harness
    )
    await service.start()
    observing = asyncio.create_task(warning.wait())
    try:
        session = await service.create_session(str(uuid.uuid4()), str(tmp_path))
        await service.send(
            session.id,
            str(uuid.uuid4()),
            session.revision,
            "First",
            expected_epoch=service.epoch,
        )
        await asyncio.wait_for(service.wait_idle(session.id), 3)
        first = await service.get_detail(session.id)
        assert first.run is not None
        assert first.run.status == "completed"
        harness.configurations[harness.model] = "replacement"
        await service.send(
            session.id,
            str(uuid.uuid4()),
            first.session.revision,
            "Next",
            expected_epoch=service.epoch,
        )
        dispatcher = connections[0]._dispatcher
        assert dispatcher is not None
        processed, _ = await asyncio.wait(
            (observing, dispatcher), timeout=3, return_when=asyncio.FIRST_COMPLETED
        )
        assert processed, "The close-time warning was never processed"
        releases[0].touch()
        await asyncio.wait_for(service.wait_idle(session.id), 3)
        detail = await service.get_detail(session.id)
        assert detail.run is not None
        assert detail.run.status == "completed", detail.run.error
        assert len(connections) == 2 and connections[0].process.returncode is not None
        assert all(item.kind != "notice" for item in detail.items)
    finally:
        for release in releases:
            release.touch()
        observing.cancel()
        await asyncio.gather(observing, return_exceptions=True)
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, {"id": "custom"}])
async def test_complete_mode_overrides_restore_native_defaults(
    tmp_path, monkeypatch, profile
):
    defaults = {
        "approvalPolicy": {"granular": {"sandbox_approval": True, "rules": False}},
        "approvalsReviewer": "user",
        "activePermissionProfile": profile,
        "sandbox": {
            "type": "workspaceWrite",
            "writableRoots": [str(tmp_path / "extra")],
            "networkAccess": True,
            "excludeTmpdirEnvVar": True,
            "excludeSlashTmp": True,
        },
    }
    native = CodexAppServer(
        [],
        {},
        str(tmp_path),
        AsyncMock(),
        model_slugs={"provider/model": "native-slug"},
    )
    rpc = AsyncMock(return_value={"thread": {"id": "native", "turns": []}, **defaults})
    monkeypatch.setattr(native, "rpc", rpc)
    thread = await native.create_thread()
    assert thread.permission_defaults == defaults
    assert rpc.call_args.args[1] == {"cwd": str(tmp_path), "modelProvider": "fcc"}
    rpc.return_value = {"turn": {"id": "turn"}}
    harness = FakeHarness()
    expected = {
        "ask": {
            "approvalPolicy": "on-request",
            "approvalsReviewer": "user",
            "permissions": ":workspace",
        },
        "auto_review": {
            "approvalPolicy": "on-request",
            "approvalsReviewer": "auto_review",
            "permissions": ":workspace",
        },
        "full_access": {
            "approvalPolicy": "never",
            "approvalsReviewer": "user",
            "permissions": ":danger-full-access",
        },
        "config": {
            "approvalPolicy": defaults["approvalPolicy"],
            "approvalsReviewer": "user",
            **(
                {"permissions": "custom"}
                if profile
                else {"sandboxPolicy": defaults["sandbox"]}
            ),
        },
    }
    for mode in ("ask", "auto_review", "full_access", "config"):
        selection = harness.prepare(harness.model, "high", mode)
        await native.start_turn("hello", selection, "input", thread.permission_defaults)
        params = rpc.call_args.args[1]
        assert {
            key: value
            for key, value in params.items()
            if key
            in {"approvalPolicy", "approvalsReviewer", "permissions", "sandboxPolicy"}
        } == expected[mode]
        assert params["model"] == "native-slug" and params["effort"] == "high"


@pytest.mark.asyncio
async def test_missing_native_permission_settings_rejects_thread_preparation(
    tmp_path, monkeypatch
):
    native = CodexAppServer([], {}, str(tmp_path), AsyncMock())
    monkeypatch.setattr(
        native, "rpc", AsyncMock(return_value={"thread": {"id": "native", "turns": []}})
    )
    with pytest.raises(CodeUnavailableError, match="permission"):
        await native.create_thread()


async def connect(tmp_path, mode):
    events = []
    completed = asyncio.Event()
    prompted = asyncio.Event()

    async def receive(event):
        events.append(event)
        if event.kind == "turn_completed":
            completed.set()
        if event.kind == "prompt":
            prompted.set()

    native = CodexAppServer(
        [sys.executable, str(Path(__file__).with_name("codex_fake_process.py")), mode],
        dict(os.environ),
        str(tmp_path),
        receive,
        model_slugs={"provider/model": "provider/model"},
        fingerprints={"provider/model": "capabilities-1"},
    )
    await native.start()
    return native, events, completed, prompted


@pytest.mark.asyncio
async def test_jsonl_large_unicode_events_can_precede_rpc_ack(tmp_path):
    native, events, completed, _ = await connect(tmp_path, "large")
    try:
        assert (await native.create_thread()).id == "native-1"
        assert (
            await native.start_turn(
                "hello",
                FakeHarness().prepare("provider/model", None, "config"),
                "input-1",
                FakeHarness().permission_defaults,
            )
            == "turn-1"
        )
        await asyncio.wait_for(completed.wait(), 3)
        texts = [
            event.item.text for event in events if event.item and event.item.complete
        ]
        assert texts == ["snow ☃ " * 16000]
    finally:
        await native.close()


@pytest.mark.asyncio
async def test_server_rpc_during_start_preserves_numeric_zero_id(tmp_path):
    native, events, completed, prompted = await connect(tmp_path, "prompt")
    try:
        await native.create_thread()
        await native.start_turn(
            "hello",
            FakeHarness().prepare("provider/model", None, "config"),
            "input-1",
            FakeHarness().permission_defaults,
        )
        await asyncio.wait_for(prompted.wait(), 3)
        response = native.prepare_answer(0, {"choice": "0"})
        await native.respond(0, response)
        await asyncio.wait_for(completed.wait(), 3)
        assert any(event.item and event.item.text == "accept" for event in events)
        assert any(
            event.kind == "resolved" and event.request_id == 0 for event in events
        )
    finally:
        await native.close()


@pytest.mark.asyncio
async def test_cancelled_creation_waiter_retains_late_native_identity(tmp_path):
    native, _, _, _ = await connect(tmp_path, "delayed-create")
    try:
        creation = asyncio.create_task(native.create_thread())
        await native.rpc("test/barrier", {})
        creation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await creation
        await native.rpc("test/release-create", {})
        assert native.thread_id == "native-1"
    finally:
        await native.close()


@pytest.mark.asyncio
async def test_eof_completes_pending_calls_and_cleanup(tmp_path):
    native, _, _, _ = await connect(tmp_path, "large")
    try:
        with pytest.raises(CodeUnavailableError):
            await native.rpc("test/eof", {})
    finally:
        await native.close()
    assert native.process.returncode is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["test/malformed", "test/malformed-flood"])
async def test_malformed_frame_reports_closed_only_after_process_termination(
    tmp_path, method
):
    closed = asyncio.Event()
    returncodes = []

    async def receive(event):
        if event.kind == "closed":
            returncodes.append(native.process.returncode)
            closed.set()

    native = CodexAppServer(
        [
            sys.executable,
            str(Path(__file__).with_name("codex_fake_process.py")),
            "large",
        ],
        dict(os.environ),
        str(tmp_path),
        receive,
    )
    try:
        await native.start()
        with pytest.raises(CodeUnavailableError):
            await native.rpc(method, {})
        await asyncio.wait_for(closed.wait(), 8)
        assert returncodes and returncodes[0] is not None
        assert native.process.stdout is not None and native.process.stdout.at_eof()
    finally:
        if native.process.stdout is not None and not native.process.stdout.at_eof():
            await native.process.stdout.read()
        await native.close()


@pytest.mark.asyncio
async def test_spawned_agent_prompt_is_visible_in_its_registered_root_session(tmp_path):
    native, events, completed, prompted = await connect(tmp_path, "child-prompt")
    try:
        await native.create_thread()
        await native.start_turn(
            "delegate",
            FakeHarness().prepare("provider/model", None, "config"),
            "input-1",
            FakeHarness().permission_defaults,
        )
        await asyncio.wait_for(prompted.wait(), 3)
        event = next(event for event in events if event.kind == "prompt")
        assert event.thread_id == "native-1"
        assert event.prompt.turn_id is None
        assert event.prompt.raw["threadId"] == "child-1"
        await native.respond(0, native.prepare_answer(0, {"choice": "0"}))
        await asyncio.wait_for(completed.wait(), 3)
    finally:
        await native.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", [True, False])
async def test_turn_start_resets_sticky_effort_and_preserves_client_identity(
    tmp_path, monkeypatch, reasoning
):
    async def receive(event):
        pass

    native = CodexAppServer(
        [],
        {},
        str(tmp_path),
        receive,
        model_slugs={"provider/model": "native-slug"},
        reasoning={"provider/model": reasoning},
    )
    native.thread_id = "thread"
    requests = []

    async def rpc(method, params):
        assert method == "turn/start"
        requests.append(params)
        return {"turn": {"id": "turn"}}

    monkeypatch.setattr(native, "rpc", rpc)
    harness = FakeHarness()
    for effort in (None, "high", "off", "max", None):
        await native.start_turn(
            "hello",
            harness.prepare("provider/model", effort, "config"),
            "operation",
            harness.permission_defaults,
        )
    assert [request.get("effort") for request in requests] == (
        ["medium", "high", "none", "max", "medium"] if reasoning else [None] * 5
    )
    assert [request.get("summary") for request in requests] == (
        ["auto", "auto", "none", "auto", "auto"] if reasoning else [None] * 5
    )
    assert all(
        request["clientUserMessageId"] == "operation"
        and request["model"] == "native-slug"
        for request in requests
    )
