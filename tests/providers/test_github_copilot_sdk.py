"""Pinned SDK account, endpoint metadata, inertness and lifecycle contracts."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from copilot import CopilotClient, CopilotSession
from copilot.rpc import Model, PermissionDecisionReject, ProviderEndpoint

from free_claude_code.providers.github_copilot import native_cli, sdk
from free_claude_code.providers.github_copilot.types import (
    CopilotEgress,
    CopilotUnavailable,
)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock(spec=CopilotClient)
    for name in ("start", "stop", "force_stop", "delete_session"):
        setattr(client, name, AsyncMock())
    client.get_status = AsyncMock(
        return_value=SimpleNamespace(version=sdk.COPILOT_CLI_VERSION)
    )
    client.get_auth_status = AsyncMock(
        return_value=SimpleNamespace(
            isAuthenticated=True,
            authType="user",
            host="https://github.com",
            login="octocat",
        )
    )
    session = MagicMock(spec=CopilotSession)
    session.disconnect = AsyncMock()
    session.rpc = MagicMock()
    client.create_session = AsyncMock(return_value=session)
    client.rpc = MagicMock()
    monkeypatch.setattr(sdk, "verified_cli_path", AsyncMock(return_value="copilot"))
    monkeypatch.setattr(sdk, "CopilotClient", MagicMock(return_value=client))
    return client


def test_profile_environment_uses_native_account_without_token_or_byok_overrides() -> (
    None
):
    assert native_cli.profile_environment(
        {
            "Path": "bin",
            "HOME": "home",
            "COPILOT_HOME": "native",
            "GH_HOST": "enterprise",
            "gh_token": "secret",
            "GITHUB_TOKEN": "secret",
            "COPILOT_PROVIDER_FOO": "override",
            "COPILOT_DISABLE_KEYTAR": "true",
        }
    ) == {
        "Path": "bin",
        "HOME": "home",
        "COPILOT_HOME": "native",
        "GH_HOST": "enterprise",
    }


@pytest.mark.asyncio
async def test_native_profile_and_inert_session_contract(client: MagicMock) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    identity = await runtime.identity()
    assert identity is not None and identity.display == "@octocat"
    await runtime.session("claude-model")
    args = client.create_session.call_args.kwargs
    assert args["model"] == "claude-model"
    assert args["available_tools"] == []
    for flag in (
        "enable_config_discovery",
        "enable_file_hooks",
        "enable_host_git_operations",
        "enable_skills",
        "enable_session_store",
        "enable_session_telemetry",
    ):
        assert args[flag] is False
    assert args["infinite_sessions"] == {"enabled": False}
    assert args["memory"] == {"enabled": False}
    directory = Path(args["working_directory"])
    assert directory.is_dir()
    await runtime.close()
    client.delete_session.assert_awaited_once_with(args["session_id"])
    assert not directory.exists()
    client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnect_failure_still_deletes_owned_session(
    client: MagicMock,
) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    await runtime.session("model")
    client.create_session.return_value.disconnect.side_effect = RuntimeError(
        "detach failed"
    )
    await runtime.close()
    client.delete_session.assert_awaited_once()
    client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_deletion_retains_record_for_retry_after_stopping(
    client: MagicMock,
) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    await runtime.session("model")
    client.delete_session.side_effect = [RuntimeError("delete failed"), None]
    with pytest.raises(CopilotUnavailable, match="session record"):
        await runtime.close()
    client.stop.assert_awaited_once()
    await runtime.close()
    assert client.delete_session.await_count == 2
    assert client.start.await_count == 2
    assert client.stop.await_count == 2
    await runtime.close()
    assert client.delete_session.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("creation_failed", [False, True])
@pytest.mark.parametrize(
    "delete_error", [RuntimeError("no saved record"), TimeoutError()]
)
async def test_cleanup_finishes_when_owned_session_record_is_confirmed_absent(
    client: MagicMock, creation_failed: bool, delete_error: Exception
) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    if creation_failed:
        client.create_session.side_effect = TimeoutError()
        with pytest.raises(CopilotUnavailable):
            await runtime.session("model")
    else:
        await runtime.session("model")
    args = client.create_session.call_args.kwargs
    directory = Path(args["working_directory"])
    client.delete_session.side_effect = delete_error
    client.get_session_metadata = AsyncMock(return_value=None)

    await runtime.close()

    client.get_session_metadata.assert_awaited_once_with(args["session_id"])
    assert not directory.exists()
    stopped = client.stop.await_count
    await runtime.close()
    assert client.stop.await_count == stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup_failed", [False, True])
async def test_cleanup_retains_record_when_absence_cannot_be_verified(
    client: MagicMock, lookup_failed: bool
) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    await runtime.session("model")
    args = client.create_session.call_args.kwargs
    directory = Path(args["working_directory"])
    client.delete_session.side_effect = [RuntimeError("delete failed"), None]
    client.get_session_metadata = AsyncMock(
        return_value=SimpleNamespace(session_id=args["session_id"]),
        side_effect=TimeoutError() if lookup_failed else None,
    )
    with pytest.raises(CopilotUnavailable, match="session record"):
        await runtime.close()
    assert directory.exists()
    await runtime.close()
    assert not directory.exists()
    assert client.delete_session.await_count == 2


@pytest.mark.asyncio
async def test_runtime_stop_allows_record_cleanup_without_retrying_stale_detach(
    client: MagicMock,
) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    session = await runtime.session("model")
    directory = Path(client.create_session.call_args.kwargs["working_directory"])
    client.create_session.return_value.disconnect.side_effect = RuntimeError(
        "detach failed"
    )
    client.delete_session.side_effect = RuntimeError("record absent")
    client.get_session_metadata = AsyncMock(return_value=None)
    with pytest.raises(RuntimeError, match="record absent"):
        await session.close()
    with pytest.raises(CopilotUnavailable, match="session record"):
        await runtime.close()
    assert directory.exists()
    detach_attempts = client.create_session.return_value.disconnect.await_count

    await runtime.close()

    assert not directory.exists()
    assert client.create_session.return_value.disconnect.await_count == detach_attempts
    with pytest.raises(CopilotUnavailable, match="closed"):
        await session.endpoint()


@pytest.mark.asyncio
async def test_unknown_creation_record_is_cleaned_after_its_runtime_stops(
    client: MagicMock,
) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    client.create_session.side_effect = TimeoutError()
    with pytest.raises(CopilotUnavailable):
        await runtime.session("model")
    pending_creation, record_exists = True, False

    async def stop() -> None:
        nonlocal pending_creation, record_exists
        if pending_creation:
            pending_creation = False
            record_exists = True

    async def delete(_session_id: str) -> None:
        nonlocal record_exists
        if not record_exists:
            raise RuntimeError("record absent")
        record_exists = False

    async def metadata(_session_id: str) -> object:
        return object() if record_exists else None

    client.stop.side_effect = stop
    client.delete_session.side_effect = delete
    client.get_session_metadata = AsyncMock(side_effect=metadata)

    await runtime.close()

    assert not pending_creation
    assert not record_exists


@pytest.mark.asyncio
async def test_cancelled_shutdown_drains_stop_before_cancellation(
    client: MagicMock,
) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    stopping, release = asyncio.Event(), asyncio.Event()

    async def stop() -> None:
        stopping.set()
        await release.wait()

    client.stop.side_effect = stop
    task = asyncio.create_task(runtime.close())
    await stopping.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await runtime.close()
    client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_force_stops_when_graceful_stop_fails(client: MagicMock) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    client.stop.side_effect = RuntimeError("stop failed")
    await runtime.close()
    client.force_stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_session_creation_retains_unknown_record_for_deletion(
    client: MagicMock,
) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    client.create_session.side_effect = TimeoutError()
    with pytest.raises(CopilotUnavailable):
        await runtime.session("model")
    session_id = client.create_session.call_args.kwargs["session_id"]
    await runtime.close()
    client.delete_session.assert_awaited_once_with(session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type", ["env", "token", "api-key"])
async def test_token_override_cannot_become_connected_account(
    client: MagicMock, auth_type: str
) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    client.get_auth_status.return_value.authType = auth_type
    with pytest.raises(CopilotUnavailable, match="existing GitHub profile"):
        await runtime.identity()
    await runtime.close()


@pytest.mark.parametrize(
    "family,wire,egress",
    [
        ("anthropic", None, CopilotEgress.MESSAGES),
        ("openai", "responses", CopilotEgress.RESPONSES),
        ("openai", "completions", CopilotEgress.CHAT),
        ("openai", None, CopilotEgress.CHAT),
    ],
)
def test_endpoint_uses_sdk_family_and_wire_metadata(
    family: str, wire: str | None, egress: CopilotEgress
) -> None:
    endpoint = ProviderEndpoint.from_dict(
        {
            "type": family,
            "wireApi": wire,
            "baseUrl": "https://upstream.invalid",
            "headers": {"X-Session": "secret"},
        }
    )
    result = sdk._endpoint(endpoint, "model")
    assert result.egress is egress
    assert "secret" not in repr(result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseUrl", "http://upstream.invalid"),
        ("baseUrl", "https://user:pass@upstream.invalid"),
        ("baseUrl", "https://upstream.invalid?token=x"),
        ("transport", "websockets"),
        ("type", "azure"),
        ("headers", {"Invalid Header": "secret"}),
        ("apiKey", "secret\nheader"),
    ],
)
def test_endpoint_rejects_unsupported_or_unsafe_metadata(
    field: str, value: object
) -> None:
    data = {
        "type": "openai",
        "baseUrl": "https://upstream.invalid",
        "headers": {},
        field: value,
    }
    with pytest.raises(CopilotUnavailable, match=r"unsupported|invalid"):
        sdk._endpoint(ProviderEndpoint.from_dict(data), "model")


@pytest.mark.asyncio
async def test_discovery_is_uncached_and_filters_auto_and_disabled_models(
    client: MagicMock,
) -> None:
    def model(name: str, policy: str) -> Model:
        return Model.from_dict(
            {
                "id": name,
                "name": name,
                "capabilities": {
                    "supports": {
                        "vision": True,
                        "adaptive_thinking": "required",
                        "reasoningEffort": True,
                    },
                    "limits": {
                        "max_output_tokens": 42,
                        "max_context_window_tokens": 100,
                    },
                },
                "policy": {"state": policy},
                "supportedReasoningEfforts": ["low", "high"],
            }
        )

    client.rpc.models.list = AsyncMock(
        return_value=SimpleNamespace(
            models=[
                model("Auto", "enabled"),
                model("claude", "enabled"),
                model("disabled", "disabled"),
            ]
        )
    )
    runtime = sdk.SdkRuntime()
    await runtime.start()
    result = await runtime.models()
    assert [item.info.model_id for item in result] == ["claude"]
    assert result[0].messages.adaptive_thinking == "required"
    assert result[0].info.max_output_tokens == 42
    await runtime.models()
    assert client.rpc.models.list.await_count == 2
    await runtime.close()


@pytest.mark.asyncio
async def test_inert_constructor_preserves_native_profile_and_disables_overrides(
    client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COPILOT_HOME", "native-profile")
    monkeypatch.setenv("GITHUB_TOKEN", "never-send-this")
    monkeypatch.setenv("COPILOT_PROVIDER_OPENAI_API_KEY", "never-send-this")
    runtime = sdk.SdkRuntime()
    await runtime.start()
    constructor = sdk.CopilotClient
    assert isinstance(constructor, MagicMock)
    options = constructor.call_args.kwargs
    assert options["mode"] == "copilot-cli"
    assert options["use_logged_in_user"] is True
    assert options["log_level"] == "error"
    assert options["connection"] == sdk.RuntimeConnection.for_stdio(
        path="copilot", args=("--disable-builtin-mcps",)
    )
    assert options["env"]["COPILOT_HOME"] == "native-profile"
    assert options["env"]["COPILOT_ALLOW_GET_PROVIDER_ENDPOINT"] == "true"
    assert "never-send-this" not in options["env"].values()
    assert Path(options["working_directory"]).is_dir()
    assert "base_directory" not in options
    assert "github_token" not in options
    await runtime.session("model")
    args = client.create_session.call_args.kwargs
    for flag in (
        "enable_on_demand_instruction_discovery",
        "request_extensions",
    ):
        assert args[flag] is False
    for flag in (
        "skip_custom_instructions",
        "custom_agents_local_only",
        "skip_embedding_retrieval",
    ):
        assert args[flag] is True
    assert args["embedding_cache_storage"] == "in-memory"
    assert args["mcp_oauth_token_storage"] == "in-memory"
    assert args["capi"] == {"enable_web_socket_responses": False}
    assert args["included_builtin_skills"] == []
    decision = args["on_permission_request"](MagicMock(), MagicMock())
    assert isinstance(decision, PermissionDecisionReject)
    client.create_session.return_value.send.assert_not_called()
    await runtime.close()


@pytest.mark.asyncio
async def test_cleanup_reconnect_failure_stops_partial_runtime_and_retains_deletion(
    client: MagicMock,
) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    await runtime.session("model")
    client.delete_session.side_effect = [RuntimeError("delete failed"), None]
    with pytest.raises(CopilotUnavailable, match="session record"):
        await runtime.close()
    client.start.side_effect = [RuntimeError("partial restart"), None]
    with pytest.raises(CopilotUnavailable, match="reconnect"):
        await runtime.close()
    assert client.stop.await_count == 2
    assert client.delete_session.await_count == 1
    await runtime.close()
    assert client.stop.await_count == 3
    assert client.delete_session.await_count == 2


@pytest.mark.asyncio
async def test_failed_stop_and_force_stop_preserve_runtime_for_retry(
    client: MagicMock,
) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    client.stop.side_effect = [RuntimeError("stop failed"), None]
    client.force_stop.side_effect = RuntimeError("force stop failed")
    with pytest.raises(CopilotUnavailable, match="Could not stop"):
        await runtime.close()
    await runtime.close()
    assert client.start.await_count == 1
    assert client.stop.await_count == 2
    assert client.force_stop.await_count == 1


@pytest.mark.asyncio
async def test_cancelled_start_drains_partial_runtime_before_returning(
    client: MagicMock,
) -> None:
    starting, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def start() -> None:
        starting.set()
        await asyncio.Event().wait()

    async def stop() -> None:
        stopping.set()
        await release.wait()

    client.start.side_effect = start
    client.stop.side_effect = stop
    runtime = sdk.SdkRuntime()
    task = asyncio.create_task(runtime.start())
    await starting.wait()
    task.cancel()
    await stopping.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await runtime.close()
    client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_creation_retains_pending_record_until_runtime_close(
    client: MagicMock,
) -> None:
    creating = asyncio.Event()

    async def create(**_kwargs: object) -> None:
        creating.set()
        await asyncio.Event().wait()

    client.create_session.side_effect = create
    runtime = sdk.SdkRuntime()
    await runtime.start()
    task = asyncio.create_task(runtime.session("model"))
    await creating.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await runtime.close()
    client.delete_session.assert_awaited_once_with(
        client.create_session.call_args.kwargs["session_id"]
    )


@pytest.mark.asyncio
async def test_cancelled_session_close_waits_for_record_deletion(
    client: MagicMock,
) -> None:
    deleting, release = asyncio.Event(), asyncio.Event()

    async def delete(_session_id: str) -> None:
        deleting.set()
        await release.wait()

    client.delete_session.side_effect = delete
    runtime = sdk.SdkRuntime()
    await runtime.start()
    session = await runtime.session("model")
    task = asyncio.create_task(session.close())
    await deleting.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await runtime.close()
    client.delete_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_endpoint_refresh_returns_fresh_bound_immutable_snapshot(
    client: MagicMock,
) -> None:
    original = ProviderEndpoint.from_dict(
        {
            "type": "anthropic",
            "baseUrl": "https://upstream.invalid/",
            "headers": {"X-Session": "old", "x-other": "original"},
            "sessionToken": {
                "header": "x-session",
                "token": "first-secret",
                "model": "model",
                "expiresAt": "2026-09-04T20:00:00+02:00",
            },
        }
    )
    rotated = ProviderEndpoint.from_dict(
        {
            "type": "anthropic",
            "baseUrl": "https://other-upstream.invalid",
            "headers": {},
            "sessionToken": {"header": "X-Session", "token": "second-secret"},
        }
    )
    resolve = AsyncMock(side_effect=[original, rotated])
    client.create_session.return_value.rpc.provider.get_endpoint = resolve
    runtime = sdk.SdkRuntime()
    await runtime.start()
    session = await runtime.session("model")
    first = await session.endpoint()
    second = await session.endpoint()
    assert first.http.base_url == "https://upstream.invalid"
    assert first.http.headers == {"x-other": "original", "x-session": "first-secret"}
    assert first.expires_at == datetime(2026, 9, 4, 18, tzinfo=UTC).timestamp()
    assert second.http.headers == {"x-session": "second-secret"}
    assert second.expires_at is None
    original.headers["x-other"] = "changed-after-snapshot"
    assert first.http.headers["x-other"] == "original"
    assert "secret" not in repr(first)
    assert "secret" not in repr(first.http)
    assert resolve.await_count == 2
    for call in resolve.await_args_list:
        assert call.args[0].model_id == "model"
        assert call.kwargs["timeout"] == sdk.SDK_RPC_TIMEOUT
    await runtime.close()
    with pytest.raises(CopilotUnavailable, match="closed"):
        await session.endpoint()
    assert resolve.await_count == 2


@pytest.mark.parametrize(
    "token",
    [
        {"header": "X-Session", "token": "secret", "model": "different-model"},
        {"header": "X Session", "token": "secret"},
        {"header": "X-Session", "token": ""},
        {"header": "X-Session", "token": "secret\r\nx: y"},
        {"header": "X-Session", "token": "secret\0"},
        {"header": "X-Session", "token": "snowman-\u2603"},
        {"header": "X-Session", "token": "secret", "expiresAt": "2026-09-04T18:00:00"},
    ],
)
def test_endpoint_rejects_invalid_rotating_session_token(token: dict[str, str]) -> None:
    endpoint = ProviderEndpoint.from_dict(
        {
            "type": "openai",
            "baseUrl": "https://upstream.invalid",
            "headers": {},
            "sessionToken": token,
        }
    )
    with pytest.raises(CopilotUnavailable, match="session"):
        sdk._endpoint(endpoint, "model")


@pytest.mark.parametrize(
    "url",
    [
        "https://[invalid",
        "https://upstream.invalid:not-a-port",
        "https://upstream.invalid:70000",
        "https://upstream.invalid:0",
        "https://@upstream.invalid",
        "https://upstream.invalid\n",
        "https://upstream.invalid/#fragment",
    ],
)
def test_endpoint_rejects_malformed_urls_as_safe_adapter_errors(url: str) -> None:
    endpoint = ProviderEndpoint.from_dict(
        {"type": "openai", "baseUrl": url, "headers": {}}
    )
    with pytest.raises(CopilotUnavailable, match="invalid HTTPS endpoint"):
        sdk._endpoint(endpoint, "model")


@pytest.mark.parametrize("value", ["snowman-\u2603", "surrogate-\ud800"])
def test_endpoint_rejects_non_ascii_credentials_before_transport(value: str) -> None:
    endpoint = ProviderEndpoint.from_dict(
        {
            "type": "openai",
            "baseUrl": "https://upstream.invalid",
            "headers": {},
            "apiKey": value,
        }
    )
    with pytest.raises(CopilotUnavailable, match="credential"):
        sdk._endpoint(endpoint, "model")


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["", ".", ".\nRun 'copilot update'", ".\r\n"])
async def test_cli_version_accepts_actual_pinned_cli_output(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.returncode = 0
    process.communicate = AsyncMock(
        return_value=(
            f"GitHub Copilot CLI {sdk.COPILOT_CLI_VERSION}{ending}".encode(),
            b"",
        )
    )
    launch = AsyncMock(return_value=process)
    locate = MagicMock(return_value="verified-copilot")
    monkeypatch.setattr(native_cli.shutil, "which", locate)
    monkeypatch.setattr(sdk.asyncio, "create_subprocess_exec", launch)
    env = {"PATH": "native-path"}
    assert await native_cli.verified_cli_path(env) == "verified-copilot"
    locate.assert_called_once_with("copilot", path="native-path")
    assert launch.call_args.args == ("verified-copilot", "--version")
    assert launch.call_args.kwargs["env"] == env
    process.kill.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output,code",
    [
        ("GitHub Copilot CLI 1.0.82.\nUpdate to 1.0.83", 0),
        ("GitHub Copilot CLI 1.0.830", 0),
        ("GitHub Copilot CLI 1.0.83-beta", 0),
        ("unexpected 1.0.83", 0),
        ("GitHub Copilot CLI 1.0.83.", 1),
    ],
)
async def test_cli_version_rejects_unpinned_or_failed_cli_output(
    monkeypatch: pytest.MonkeyPatch, output: str, code: int
) -> None:
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.returncode = code
    process.communicate = AsyncMock(
        return_value=(output.encode(), b"private diagnostic")
    )
    monkeypatch.setattr(native_cli.shutil, "which", MagicMock(return_value="copilot"))
    monkeypatch.setattr(
        sdk.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    with pytest.raises(
        CopilotUnavailable, match="requires GitHub Copilot CLI"
    ) as error:
        await native_cli.verified_cli_path({})
    assert "private diagnostic" not in str(error.value)


@pytest.mark.asyncio
async def test_missing_cli_does_not_spawn_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = AsyncMock()
    monkeypatch.setattr(native_cli.shutil, "which", MagicMock(return_value=None))
    monkeypatch.setattr(sdk.asyncio, "create_subprocess_exec", launch)
    with pytest.raises(CopilotUnavailable, match="Install GitHub Copilot CLI"):
        await native_cli.verified_cli_path({})
    launch.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_cli_version_probe_drains_process_under_repeated_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, reaping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.returncode = None

    async def communicate() -> tuple[bytes, bytes]:
        if not started.is_set():
            started.set()
            await asyncio.Event().wait()
        reaping.set()
        await release.wait()
        process.returncode = -1
        return b"", b""

    process.communicate = AsyncMock(side_effect=communicate)
    monkeypatch.setattr(native_cli.shutil, "which", MagicMock(return_value="copilot"))
    monkeypatch.setattr(
        sdk.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    task = asyncio.create_task(native_cli.verified_cli_path({}))
    await started.wait()
    task.cancel()
    await reaping.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    process.kill.assert_called_once()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.returncode == -1


@pytest.mark.asyncio
async def test_runtime_version_mismatch_stops_runtime_and_never_creates_session(
    client: MagicMock,
) -> None:
    client.get_status.return_value.version = "unverified-version"
    runtime = sdk.SdkRuntime()
    with pytest.raises(CopilotUnavailable, match="version changed"):
        await runtime.start()
    client.stop.assert_awaited_once()
    client.create_session.assert_not_awaited()
    await runtime.close()
    client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_signed_out_profile_is_distinct_from_invalid_auth(
    client: MagicMock,
) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    client.get_auth_status.return_value.isAuthenticated = False
    assert await runtime.identity() is None
    await runtime.close()


@pytest.mark.asyncio
async def test_gh_cli_enterprise_profile_is_supported(client: MagicMock) -> None:
    runtime = sdk.SdkRuntime()
    await runtime.start()
    client.get_auth_status.return_value.authType = "gh-cli"
    client.get_auth_status.return_value.host = "HTTPS://GitHub.Enterprise.Example"
    identity = await runtime.identity()
    assert identity is not None
    assert identity.host == "github.enterprise.example"
    assert "octocat" in identity.display
    await runtime.close()
