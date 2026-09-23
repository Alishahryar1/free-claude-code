import asyncio
import json
import os
import sys
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.application.code_sessions import CodeService
from free_claude_code.application.code_sessions.models import (
    CodeConflictError,
    CodeUnavailableError,
)
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import ModelCatalogSnapshot, RequestRuntimePort
from free_claude_code.config.paths import launcher_temp_dir_path
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import no_thinking_gateway_model_id
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from free_claude_code.runtime.codex_app_server import (
    CodexAppServer,
    CodexHarnessFactory,
)
from tests.code_sessions_support import FakeHarness


@dataclass
class FactoryPeer:
    mode: str = "factory-ready"
    command: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict, repr=False)
    catalog: Path | None = None
    process: asyncio.subprocess.Process | None = None


@pytest.fixture
def factory_peer(tmp_path, monkeypatch):
    peer = FactoryPeer()
    spawn = asyncio.create_subprocess_exec

    async def start(*command, **kwargs):
        if command[0] != "test-codex":
            return await spawn(*command, **kwargs)
        peer.command, peer.env = command, dict(kwargs["env"])
        config = tomllib.loads(
            "\n".join(command[i + 1] for i, arg in enumerate(command) if arg == "-c")
        )
        path = config.get("model_catalog_json")
        peer.catalog = Path(path) if path else None
        peer.process = await spawn(
            sys.executable,
            str(Path(__file__).with_name("codex_fake_process.py")),
            peer.mode,
            path or "",
            str(tmp_path / "catalog-at-exit.json"),
            **kwargs,
        )
        return peer.process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    return peer


@pytest.fixture
def factory_runtime():
    runtime = MagicMock(spec=RequestRuntimePort)
    runtime.current_settings.return_value = Settings(
        model="nvidia_nim/default", port=8182, proxy_auth_token="factory-token"
    )
    runtime.cached_model_info.return_value = None
    runtime.cached_prefixed_model_infos.return_value = (
        ProviderModelInfo(
            "open_router/selected", supports_thinking=False, context_window_tokens=32000
        ),
    )
    runtime.wait_for_catalog.return_value = ModelCatalogSnapshot(
        runtime.current_settings(), runtime.cached_prefixed_model_infos()
    )
    runtime.acquire.side_effect = AssertionError("Setup must not acquire a provider")
    return runtime


@pytest.mark.asyncio
async def test_factory_open_preserves_setup_and_catalog_until_process_exit(
    tmp_path, factory_peer, factory_runtime
):
    env = dict(os.environ) | {
        "CODEX_HOME": str(tmp_path / "codex-home"),
        "CODEX_THREAD_ID": "parent-thread",
        "OPENAI_API_KEY": "parent-token",
        "OPENAI_CUSTOM": "parent-route",
        "KEEP_ME": "yes",
    }
    before = dict(env)
    factory = CodexHarnessFactory(factory_runtime, binary="test-codex", env=env)
    selection = await factory.prepare("open_router/selected", None, "config")
    native = await selection.open(str(tmp_path), AsyncMock())
    try:
        config = tomllib.loads(
            "\n".join(
                factory_peer.command[i + 1]
                for i, arg in enumerate(factory_peer.command)
                if arg == "-c"
            )
        )
        assert config["model"] == no_thinking_gateway_model_id(selection.model)
        assert config["model_providers"]["fcc"] == {
            "name": "Free Claude Code",
            "base_url": "http://127.0.0.1:8182/v1",
            "auth": {"command": "fcc-codex", "args": ["--print-proxy-auth-token"]},
            "wire_api": "responses",
        }
        assert factory_peer.command[-5:] == (
            "-c",
            "features.default_mode_request_user_input=true",
            "-c",
            "tools.experimental_request_user_input.enabled=true",
            "app-server",
        )
        assert factory_peer.env["CODEX_HOME"] == env["CODEX_HOME"]
        assert factory_peer.env["KEEP_ME"] == "yes"
        assert "CODEX_THREAD_ID" not in factory_peer.env
        assert not any(key.startswith("OPENAI_") for key in factory_peer.env)
        assert "127.0.0.1" in factory_peer.env["NO_PROXY"].split(",")
        assert env == before
        assert factory_peer.catalog is not None
        catalog = json.loads(factory_peer.catalog.read_text())
        assert any(model["slug"] == config["model"] for model in catalog["models"])
        assert "factory-token" not in factory_peer.catalog.read_text()
        assert native.process.returncode is None
    finally:
        await native.close()
    assert native.process.returncode == 0
    assert json.loads((tmp_path / "catalog-at-exit.json").read_text()) == catalog
    assert not factory_peer.catalog.parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["reject", "cancel"])
async def test_factory_initialization_failure_reaps_before_catalog_cleanup(
    tmp_path, monkeypatch, factory_peer, factory_runtime, failure
):
    factory_peer.mode = "factory-fail" if failure == "reject" else "factory-wait"
    initialized = asyncio.Event()
    write = CodexAppServer._write

    async def observe_write(native, message):
        await write(native, message)
        if message.get("method") == "initialize":
            initialized.set()

    monkeypatch.setattr(CodexAppServer, "_write", observe_write)
    factory = CodexHarnessFactory(factory_runtime, binary="test-codex")
    selection = await factory.prepare("open_router/selected", None, "config")
    opening = asyncio.create_task(selection.open(str(tmp_path), AsyncMock()))
    try:
        await asyncio.wait_for(initialized.wait(), 3)
        if failure == "cancel":
            opening.cancel()
        error = asyncio.CancelledError if failure == "cancel" else CodeConflictError
        with pytest.raises(error):
            await asyncio.wait_for(opening, 8)
    finally:
        if not opening.done():
            opening.cancel()
        await asyncio.gather(opening, return_exceptions=True)
    assert factory_peer.process.returncode == 0
    assert json.loads((tmp_path / "catalog-at-exit.json").read_text())["models"]
    assert not factory_peer.catalog.parent.exists()


@pytest.mark.asyncio
async def test_history_open_needs_no_fcc_setup_or_inventory(tmp_path, factory_peer):
    runtime = MagicMock(spec=RequestRuntimePort)
    for name in (
        "current_settings",
        "cached_model_info",
        "cached_prefixed_model_infos",
    ):
        getattr(runtime, name).side_effect = AssertionError(
            "History needs no inventory"
        )
    env = dict(os.environ) | {"CODEX_THREAD_ID": "parent", "OPENAI_API_KEY": "keep"}
    factory = CodexHarnessFactory(runtime, binary="test-codex", env=env)
    native = await factory.open_history(str(tmp_path), AsyncMock())
    try:
        assert factory_peer.command == ("test-codex", "app-server")
        assert factory_peer.env == env
        assert factory_peer.catalog is None
        assert not launcher_temp_dir_path().exists()
    finally:
        await native.close()
    assert native.process.returncode == 0
    assert json.loads((tmp_path / "catalog-at-exit.json").read_text()) is None


@pytest.mark.asyncio
async def test_child_warning_during_shutdown_allows_connection_replacement(
    tmp_path, monkeypatch
):
    harness = FakeHarness()
    prepare = harness.prepare
    connections = []
    releases = []
    warning = asyncio.Event()

    async def selection_for(model, effort, mode):
        selection = await prepare(model, effort, mode)

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
        selection = await harness.prepare(harness.model, "high", mode)
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
async def test_jsonl_sink_preserves_usage_before_turn_completion(tmp_path):
    events = []
    usage_received = asyncio.Event()
    release_usage = asyncio.Event()
    completed = asyncio.Event()

    async def receive(event):
        events.append(event.kind)
        if event.kind == "context_usage":
            usage_received.set()
            await release_usage.wait()
        elif event.kind == "turn_completed":
            completed.set()

    native = CodexAppServer(
        [
            sys.executable,
            str(Path(__file__).with_name("codex_fake_process.py")),
            "ordered-usage",
        ],
        dict(os.environ),
        str(tmp_path),
        receive,
        model_slugs={"provider/model": "provider/model"},
        fingerprints={"provider/model": "capabilities-1"},
    )
    try:
        await native.start()
        await native.create_thread()
        turn_id = await native.start_turn(
            "hello",
            (await FakeHarness().prepare("provider/model", None, "config")),
            "input-1",
            FakeHarness().permission_defaults,
        )
        assert turn_id == "turn-1"
        await asyncio.wait_for(usage_received.wait(), 3)
        assert not completed.is_set()
        release_usage.set()
        await asyncio.wait_for(completed.wait(), 3)
        assert [
            kind for kind in events if kind in {"context_usage", "turn_completed"}
        ] == ["context_usage", "turn_completed"]
    finally:
        release_usage.set()
        await native.close()


@pytest.mark.asyncio
async def test_jsonl_large_unicode_events_can_precede_rpc_ack(tmp_path):
    native, events, completed, _ = await connect(tmp_path, "large")
    try:
        assert (await native.create_thread()).id == "native-1"
        assert (
            await native.start_turn(
                "hello",
                (await FakeHarness().prepare("provider/model", None, "config")),
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
async def test_terminal_storage_failure_drains_the_native_dispatcher(
    tmp_path, monkeypatch
):
    harness = FakeHarness()
    prepare = harness.prepare
    connections = []

    async def select(model, effort, mode):
        selection = await prepare(model, effort, mode)

        async def open_native(cwd, sink):
            native = CodexAppServer(
                [
                    sys.executable,
                    str(Path(__file__).with_name("codex_fake_process.py")),
                    "large",
                ],
                dict(os.environ),
                cwd,
                sink,
                model_slugs={model: model},
                fingerprints=selection.catalog,
            )
            await native.start()
            connections.append(native)
            return native

        monkeypatch.setattr(selection, "open", open_native)
        return selection

    monkeypatch.setattr(harness, "prepare", select)
    store = SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock")
    save = store.save_progress
    rejected = []

    async def reject_terminal(session, revision, **values):
        run = values.get("run")
        if run is not None and run.status == "completed":
            rejected.append(run)
            raise CodeUnavailableError("Terminal commit failed")
        await save(session, revision, **values)

    monkeypatch.setattr(store, "save_progress", reject_terminal)
    service = CodeService(store, harness)
    await service.start()
    try:
        session = await service.create_session(str(uuid.uuid4()), str(tmp_path))
        await service.send(
            session.id,
            str(uuid.uuid4()),
            session.revision,
            "hello",
            expected_epoch=service.epoch,
        )
        await asyncio.wait_for(service.wait_idle(session.id), 5)
        assert len(rejected) == 1 and len(connections) == 1
        native = connections[0]
        assert native.process.returncode is not None
        assert native._reader.done() and native._dispatcher.done()
        saved = await store.latest_run(session.id)
        assert saved is not None and saved.status == "running"
        with pytest.raises(CodeUnavailableError):
            await service.get_detail(session.id)
    finally:
        await asyncio.wait_for(service.close(), 5)


@pytest.mark.asyncio
async def test_server_rpc_during_start_preserves_numeric_zero_id(tmp_path):
    native, events, completed, prompted = await connect(tmp_path, "prompt")
    try:
        await native.create_thread()
        await native.start_turn(
            "hello",
            (await FakeHarness().prepare("provider/model", None, "config")),
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
            (await FakeHarness().prepare("provider/model", None, "config")),
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
            (await harness.prepare("provider/model", effort, "config")),
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
