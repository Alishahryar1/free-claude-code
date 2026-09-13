import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import free_claude_code.messaging.session.persistence as persistence_module
from free_claude_code.application.connected_accounts import (
    ConnectedAccountState,
    ConnectedAccountStatus,
)
from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.admin.persistence import PreparedAdminUpdate
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.messaging.command_context import StopOutcome
from free_claude_code.messaging.platforms.ports import (
    InboundMessageHandler,
    MessagingPlatformComponents,
    MessagingStartupNotice,
)
from free_claude_code.messaging.session import SessionStore
from free_claude_code.messaging.workflow import MessagingWorkflow
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.credential_validation import (
    CredentialCheck,
    CredentialStatus,
)
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.providers.support import make_provider_config


class TrackingRuntime(ProviderRuntime):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.cleanup_calls = 0

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        await super().cleanup()


class AdminModelProvider(BaseProvider):
    def __init__(
        self,
        model_infos: frozenset[ProviderModelInfo] = frozenset(),
        *,
        error: Exception | None = None,
    ) -> None:
        super().__init__(
            make_provider_config(
                api_key="test",
                base_url="https://provider.invalid/v1",
            )
        )
        self._model_infos = model_infos
        self._error = error

    async def cleanup(self) -> None:
        return None

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        if self._error is not None:
            raise self._error
        return self._model_infos

    async def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""

    async def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


class TrackingFactory:
    def __init__(self) -> None:
        self.runtimes: list[TrackingRuntime] = []
        self.fail = False
        self.events: list[str] = []

    def __call__(self, settings: Settings) -> ProviderRuntime:
        self.events.append(f"construct:{settings.model}")
        if self.fail:
            raise RuntimeError("candidate failed")
        runtime = TrackingRuntime(settings)
        self.runtimes.append(runtime)
        return runtime


class TrackingTranscriber:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.close_calls = 0

    async def transcribe(self, file_path: Path) -> str:
        assert isinstance(file_path, Path)
        return "transcribed"

    async def close(self) -> None:
        self.close_calls += 1
        self.events.append("transcriber.close")


class FailingTranscriber(TrackingTranscriber):
    async def close(self) -> None:
        await super().close()
        raise RuntimeError("transcriber close failed")


class CancelledTranscriber(TrackingTranscriber):
    async def close(self) -> None:
        await super().close()
        raise asyncio.CancelledError


class CancellingOnceTranscriber(TrackingTranscriber):
    async def close(self) -> None:
        await super().close()
        if self.close_calls == 1:
            raise asyncio.CancelledError


class TrackingMessagingRuntime:
    name = "tracking"

    def __init__(
        self,
        events: list[str],
        *,
        fail_quiesce_once: bool = False,
        fail_close_once: bool = False,
    ) -> None:
        self.events = events
        self.fail_quiesce_once = fail_quiesce_once
        self.fail_close_once = fail_close_once

    async def start(self) -> None:
        self.events.append("messaging.start")

    async def quiesce(self) -> None:
        self.events.append("messaging.quiesce")
        if self.fail_quiesce_once:
            self.fail_quiesce_once = False
            raise RuntimeError("quiesce failed")

    async def close(self) -> None:
        self.events.append("messaging.close")
        if self.fail_close_once:
            self.fail_close_once = False
            raise RuntimeError("close failed")

    def on_message(self, handler: InboundMessageHandler) -> None:
        assert callable(handler)

    @property
    def is_connected(self) -> bool:
        return True


class PersistentlyFailingMessagingRuntime(TrackingMessagingRuntime):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.fail_quiesce = True

    async def quiesce(self) -> None:
        self.events.append("messaging.quiesce")
        if self.fail_quiesce:
            raise RuntimeError("quiesce failed")


def _settings(model: str, *, port: int = 8082) -> Settings:
    return Settings().model_copy(update={"model": model, "port": port})


def _prepared(
    settings: Settings,
    tmp_path,
    *,
    pending_fields: tuple[str, ...] = (),
) -> PreparedAdminUpdate:
    return PreparedAdminUpdate(
        target_values={"MODEL": settings.model},
        settings=settings,
        errors=(),
        pending_fields=pending_fields,
        path=tmp_path / ".env",
    )


def _applied_response(pending_fields: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "applied": True,
        "valid": True,
        "errors": [],
        "env_preview": "MODEL=updated\n",
        "path": ".env",
        "pending_fields": list(pending_fields),
    }


def _runtime_with_admin_provider(
    provider: BaseProvider,
) -> tuple[ApplicationRuntime, ProviderRuntimeManager]:
    settings = _settings("nvidia_nim/fallback").model_copy(
        update={"nvidia_nim_api_key": "test-key"}
    )
    manager = ProviderRuntimeManager(
        settings,
        runtime_factory=lambda snapshot: ProviderRuntime(
            snapshot,
            {"nvidia_nim": provider},
        ),
    )
    return ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    ), manager


@pytest.mark.asyncio
async def test_provider_check_caches_and_returns_sorted_models() -> None:
    provider = AdminModelProvider(
        frozenset(
            {
                ProviderModelInfo("vendor/model-b"),
                ProviderModelInfo("vendor/model-a"),
            }
        )
    )
    runtime, manager = _runtime_with_admin_provider(provider)

    result = await runtime.test_provider("nvidia_nim")

    assert result == {
        "provider_id": "nvidia_nim",
        "ok": True,
        "models": ["vendor/model-a", "vendor/model-b"],
    }
    assert manager.cached_model_ids()["nvidia_nim"] == frozenset(
        {"vendor/model-a", "vendor/model-b"}
    )
    await manager.close()


@pytest.mark.asyncio
async def test_provider_check_returns_stable_failure_for_application_error() -> None:
    provider = AdminModelProvider(
        error=ApplicationUnavailableError(
            "NVIDIA_NIM_API_KEY is not set. Add it in the Admin UI."
        )
    )
    runtime, manager = _runtime_with_admin_provider(provider)

    result = await runtime.test_provider("nvidia_nim")

    assert result == {
        "provider_id": "nvidia_nim",
        "ok": False,
        "message": (
            "Could not refresh this provider's models. "
            "Verify its configuration and access."
        ),
    }
    assert "error_type" not in result
    await manager.close()


@pytest.mark.asyncio
async def test_provider_check_never_returns_unrecognized_credentials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "CREDENTIAL[unrecognized-format-987654321]"
    provider = AdminModelProvider(
        error=RuntimeError(f"Provider rejected credential {secret}")
    )
    runtime, manager = _runtime_with_admin_provider(provider)

    with caplog.at_level(logging.WARNING):
        result = await runtime.test_provider("nvidia_nim")

    assert result == {
        "provider_id": "nvidia_nim",
        "ok": False,
        "message": (
            "Could not refresh this provider's models. "
            "Verify its configuration and access."
        ),
    }
    log_text = " | ".join(record.getMessage() for record in caplog.records)
    assert "provider=nvidia_nim" in log_text
    assert "exc_type=RuntimeError" in log_text
    assert secret not in log_text
    assert "error_type" not in result
    await manager.close()


@pytest.mark.asyncio
async def test_stop_all_maps_messaging_outcome_to_application_count() -> None:
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    workflow = MagicMock()
    workflow.stop_all_tasks = AsyncMock(
        return_value=StopOutcome(
            cancelled_count=3,
            status_feedback_scopes=frozenset(),
            fallback_required=False,
        )
    )
    runtime._messaging_workflow = workflow

    result = await runtime.stop_all()

    assert result is not None
    assert result.cancelled_count == 3
    workflow.stop_all_tasks.assert_awaited_once()
    await manager.close()


@pytest.mark.asyncio
async def test_provider_apply_constructs_before_commit_then_publishes(tmp_path) -> None:
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/old"),
        runtime_factory=factory,
    )
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    prepared = _prepared(_settings("nvidia_nim/new"), tmp_path)
    factory.events.clear()

    def commit(_prepared_update: PreparedAdminUpdate) -> dict[str, object]:
        factory.events.append("commit")
        assert manager.current_generation_id == 1
        return _applied_response()

    with (
        patch.object(
            runtime._configuration,
            "prepare",
            return_value=prepared,
        ),
        patch.object(
            runtime._configuration,
            "commit",
            side_effect=commit,
        ),
    ):
        result = await runtime.apply_admin_config({"MODEL": "nvidia_nim/new"})

    assert factory.events == ["construct:nvidia_nim/new", "commit"]
    assert manager.current_generation_id == 2
    assert manager.current_settings().model == "nvidia_nim/new"
    assert result["restart"] == {
        "required": False,
        "automatic": False,
        "admin_url": None,
        "fields": [],
    }
    await manager.close()


@pytest.mark.asyncio
async def test_candidate_failure_never_commits_and_preserves_current(tmp_path) -> None:
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/old"),
        runtime_factory=factory,
    )
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    prepared = _prepared(_settings("nvidia_nim/new"), tmp_path)
    factory.fail = True

    with (
        patch.object(
            runtime._configuration,
            "prepare",
            return_value=prepared,
        ),
        patch.object(runtime._configuration, "commit") as commit,
        pytest.raises(RuntimeError, match="candidate failed"),
    ):
        await runtime.apply_admin_config({"MODEL": "nvidia_nim/new"})

    commit.assert_not_called()
    assert manager.current_generation_id == 1
    assert manager.current_settings().model == "nvidia_nim/old"
    await manager.close()


@pytest.mark.asyncio
async def test_persistence_failure_closes_candidate_and_preserves_current(
    tmp_path,
) -> None:
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/old"),
        runtime_factory=factory,
    )
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    prepared = _prepared(_settings("nvidia_nim/new"), tmp_path)

    with (
        patch.object(
            runtime._configuration,
            "prepare",
            return_value=prepared,
        ),
        patch.object(
            runtime._configuration,
            "commit",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(OSError, match="disk full"),
    ):
        await runtime.apply_admin_config({"MODEL": "nvidia_nim/new"})

    assert manager.current_generation_id == 1
    assert factory.runtimes[0].cleanup_calls == 0
    assert factory.runtimes[1].cleanup_calls == 1
    await manager.close()


@pytest.mark.asyncio
async def test_restart_required_apply_commits_without_hot_publication(tmp_path) -> None:
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/old"),
        runtime_factory=factory,
    )
    restart = MagicMock(return_value=None)
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=None,
        restart_callback=restart,
    )
    prepared = _prepared(
        _settings("nvidia_nim/old", port=9090),
        tmp_path,
        pending_fields=("PORT",),
    )
    with patch.object(
        runtime._configuration, "admin_values", AsyncMock(return_value={})
    ):
        instance_id = (await runtime.admin_status())["instance_id"]

    with (
        patch.object(
            runtime._configuration,
            "prepare",
            return_value=prepared,
        ),
        patch.object(
            runtime._configuration,
            "commit",
            return_value=_applied_response(("PORT",)),
        ) as commit,
    ):
        result = await runtime.apply_admin_config({"PORT": "9090"})

    commit.assert_called_once_with(prepared)
    assert manager.current_generation_id == 1
    assert len(factory.runtimes) == 1
    assert result["restart"] == {
        "required": True,
        "automatic": True,
        "admin_url": "http://127.0.0.1:9090/admin",
        "fields": ["PORT"],
        "instance_id": instance_id,
    }
    restart.assert_called_once_with()
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [(), ("PORT",)])
@pytest.mark.parametrize("status", list(CredentialStatus))
async def test_credential_checks_gate_both_apply_paths(tmp_path, pending, status):
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/old"), runtime_factory=factory
    )
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    prepared = replace(
        _prepared(_settings("nvidia_nim/new"), tmp_path, pending_fields=pending),
        changed_keys=("GROQ_API_KEY",),
    )
    checks = (
        CredentialCheck("GROQ_API_KEY", status, "Safe result"),
        CredentialCheck("OPENROUTER_API_KEY", CredentialStatus.VERIFIED, "Accepted"),
    )
    with (
        patch.object(
            runtime._configuration,
            "prepare",
            return_value=prepared,
        ),
        patch(
            "free_claude_code.runtime.application.check_credentials",
            AsyncMock(return_value=checks),
        ) as check,
        patch.object(
            runtime._configuration,
            "commit",
            return_value=_applied_response(pending),
        ) as commit,
    ):
        result = await runtime.apply_admin_config({"GROQ_API_KEY": "new"})
    check.assert_awaited_once_with(prepared.settings, prepared.changed_keys)
    assert result["credential_checks"] == [
        {"key": check.key, "status": check.status.value, "message": check.message}
        for check in checks
    ]
    if status == CredentialStatus.REJECTED:
        commit.assert_not_called()
        assert result["applied"] is False
        assert manager.current_generation_id == 1
    else:
        commit.assert_called_once_with(prepared)
        assert result["applied"] is True
        assert manager.current_generation_id == (1 if pending else 2)
    await manager.close()


@pytest.mark.asyncio
async def test_cancelled_credential_check_never_commits(tmp_path):
    manager = ProviderRuntimeManager(_settings("nvidia_nim/old"))
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    prepared = replace(
        _prepared(_settings("nvidia_nim/new"), tmp_path), changed_keys=("GROQ_API_KEY",)
    )
    with (
        patch.object(
            runtime._configuration,
            "prepare",
            return_value=prepared,
        ),
        patch(
            "free_claude_code.runtime.application.check_credentials",
            AsyncMock(side_effect=asyncio.CancelledError),
        ),
        patch.object(runtime._configuration, "commit") as commit,
        pytest.raises(asyncio.CancelledError),
    ):
        await runtime.apply_admin_config({"GROQ_API_KEY": "new"})
    commit.assert_not_called()
    assert manager.current_generation_id == 1
    await manager.close()


@pytest.mark.asyncio
async def test_close_drains_messaging_before_transcriber_and_is_idempotent() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    transcriber = TrackingTranscriber(events)
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=transcriber,
    )
    runtime._messaging_runtime = TrackingMessagingRuntime(events)
    workflow = MagicMock()
    workflow.close = AsyncMock(side_effect=lambda: events.append("workflow.close"))
    runtime._messaging_workflow = workflow
    runtime._cli_manager = MagicMock()

    assert runtime.is_closed is False
    assert await runtime.close() is True
    assert await runtime.close() is True

    assert events == [
        "messaging.quiesce",
        "workflow.close",
        "messaging.close",
        "transcriber.close",
    ]
    assert transcriber.close_calls == 1
    assert runtime._transcriber is None
    assert runtime._messaging_runtime is None
    assert runtime._messaging_workflow is None
    assert runtime.is_closed is True


@pytest.mark.asyncio
async def test_close_retains_transcriber_ownership_when_close_fails() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    transcriber = FailingTranscriber(events)
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=transcriber,
    )
    runtime._messaging_runtime = TrackingMessagingRuntime(events)

    assert await runtime.close() is False

    assert events == [
        "messaging.quiesce",
        "messaging.close",
        "transcriber.close",
    ]
    assert transcriber.close_calls == 1
    assert runtime._transcriber is transcriber
    assert runtime._closed is False
    await manager.close()


@pytest.mark.asyncio
async def test_close_retries_runtime_before_closing_later_resources() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    transcriber = TrackingTranscriber(events)
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=transcriber,
    )
    messaging = TrackingMessagingRuntime(events, fail_close_once=True)
    runtime._messaging_runtime = messaging

    assert await runtime.close() is False

    assert events == ["messaging.quiesce", "messaging.close"]
    assert runtime._messaging_runtime is messaging
    assert runtime._transcriber is transcriber
    assert runtime._closed is False

    assert await runtime.close() is True

    assert events == [
        "messaging.quiesce",
        "messaging.close",
        "messaging.quiesce",
        "messaging.close",
        "transcriber.close",
    ]
    assert runtime._messaging_runtime is None
    assert runtime._transcriber is None
    assert runtime._closed is True


@pytest.mark.asyncio
async def test_close_retries_connected_account_without_reclosing_providers() -> None:
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/model"),
        runtime_factory=factory,
    )
    account = MagicMock()
    account.status.return_value = MagicMock(revision=0)
    account.close = AsyncMock(side_effect=[RuntimeError("auth cleanup failed"), None])
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=None,
        connected_accounts={"openai": account},
    )

    assert await runtime.close() is False
    assert manager._closed is True
    assert factory.runtimes[0].cleanup_calls == 1

    assert await runtime.close() is True
    assert account.close.await_count == 2
    assert factory.runtimes[0].cleanup_calls == 1
    assert runtime.is_closed is True


@pytest.mark.asyncio
async def test_connected_account_status_reports_cached_model_count() -> None:
    account = MagicMock()
    account.is_connected.return_value = True
    account.status.return_value = ConnectedAccountStatus(
        provider_id="openai",
        state=ConnectedAccountState.CONNECTED,
        connected=True,
        revision=1,
        email="person@example.com",
    )
    account.close = AsyncMock()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/model"),
        connected_provider_ids=lambda: ("openai",),
    )
    manager.cache_model_infos(
        "openai",
        {
            ProviderModelInfo(model_id="gpt-one"),
            ProviderModelInfo(model_id="gpt-two"),
        },
    )
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=None,
        connected_accounts={"openai": account},
    )

    status = await runtime.connected_account_status("openai")

    assert status.email == "person@example.com"
    assert status.model_count == 2
    assert status.as_dict()["model_count"] == 2
    assert await runtime.close() is True


@pytest.mark.asyncio
async def test_close_retries_workflow_close_before_closing_delivery() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    messaging = TrackingMessagingRuntime(events)
    workflow = MagicMock()
    close_calls = 0

    async def close_workflow() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("drain failed")
        events.append("workflow.close")

    workflow.close = AsyncMock(side_effect=close_workflow)
    runtime._messaging_runtime = messaging
    runtime._messaging_workflow = workflow

    assert await runtime.close() is False

    assert events == ["messaging.quiesce"]
    assert runtime._messaging_runtime is messaging
    assert runtime._messaging_workflow is workflow
    assert runtime._closed is False

    assert await runtime.close() is True

    assert events == [
        "messaging.quiesce",
        "messaging.quiesce",
        "workflow.close",
        "messaging.close",
    ]
    assert runtime._closed is True


@pytest.mark.asyncio
async def test_close_does_not_drain_workflow_until_ingress_is_quiescent() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    messaging = TrackingMessagingRuntime(events, fail_quiesce_once=True)
    workflow = MagicMock()
    workflow.close = AsyncMock(side_effect=lambda: events.append("workflow.close"))
    runtime._messaging_runtime = messaging
    runtime._messaging_workflow = workflow

    assert await runtime.close() is False

    workflow.close.assert_not_awaited()
    assert runtime._messaging_runtime is messaging
    assert runtime._messaging_workflow is workflow

    assert await runtime.close() is True

    assert events == [
        "messaging.quiesce",
        "messaging.quiesce",
        "workflow.close",
        "messaging.close",
    ]
    assert runtime._closed is True


@pytest.mark.asyncio
async def test_close_retries_failed_persistence_before_closing_delivery() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    messaging = TrackingMessagingRuntime(events)
    workflow = MagicMock()
    close_calls = 0

    async def close_workflow() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("flush failed")
        events.append("workflow.close")

    workflow.close = AsyncMock(side_effect=close_workflow)
    runtime._messaging_runtime = messaging
    runtime._messaging_workflow = workflow

    await runtime.close()

    assert runtime._messaging_workflow is workflow
    assert runtime._messaging_runtime is messaging
    assert "messaging.close" not in events

    await runtime.close()

    assert events == [
        "messaging.quiesce",
        "messaging.quiesce",
        "workflow.close",
        "messaging.close",
    ]
    assert runtime._closed is True


@pytest.mark.asyncio
async def test_close_retries_real_workflow_persistence_without_losing_latest_state(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    messaging = TrackingMessagingRuntime(events)
    cli_manager = MagicMock()
    cli_manager.stop_all = AsyncMock()
    outbound = MagicMock()
    store_path = tmp_path / "sessions.json"
    real_replace = persistence_module.os.replace
    replace_calls = 0

    def fail_first_replace(source: str, target: str) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            raise OSError("replace failed once")
        real_replace(source, target)

    with patch.object(persistence_module.threading, "Timer"):
        store = SessionStore(storage_path=str(store_path))
        store.record_message_id(
            "telegram",
            "chat_1",
            "old",
            "out",
            "status",
        )
        store.flush_pending_save()
        store.record_message_id(
            "telegram",
            "chat_1",
            "latest",
            "out",
            "status",
        )
        workflow = MessagingWorkflow(outbound, cli_manager, store)
        runtime._messaging_runtime = messaging
        runtime._messaging_workflow = workflow
        runtime._cli_manager = cli_manager

        with patch.object(
            persistence_module.os,
            "replace",
            side_effect=fail_first_replace,
        ):
            assert await runtime.close() is False

            assert runtime._messaging_runtime is messaging
            assert runtime._messaging_workflow is workflow
            assert runtime._cli_manager is cli_manager
            assert runtime.is_closed is False
            assert store.dirty is True
            assert store.get_tracked_message_ids_for_chat("telegram", "chat_1") == [
                "old",
                "latest",
            ]
            assert SessionStore(
                storage_path=str(store_path)
            ).get_tracked_message_ids_for_chat("telegram", "chat_1") == ["old"]

            assert await runtime.close() is True

        assert runtime._messaging_runtime is None
        assert runtime._messaging_workflow is None
        assert runtime._cli_manager is None
        assert runtime.is_closed is True
        assert store.dirty is False
        assert SessionStore(
            storage_path=str(store_path)
        ).get_tracked_message_ids_for_chat("telegram", "chat_1") == [
            "old",
            "latest",
        ]
        assert events == [
            "messaging.quiesce",
            "messaging.quiesce",
            "messaging.close",
        ]


@pytest.mark.asyncio
async def test_cancelled_transcriber_close_retains_ownership() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    transcriber = CancelledTranscriber(events)
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=transcriber,
    )

    with pytest.raises(asyncio.CancelledError):
        await runtime._cleanup_transcriber()

    assert transcriber.close_calls == 1
    assert runtime._transcriber is transcriber
    await manager.close()


@pytest.mark.asyncio
async def test_cancelled_application_close_remains_retryable() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    transcriber = CancellingOnceTranscriber(events)
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=transcriber,
    )

    with pytest.raises(asyncio.CancelledError):
        await runtime.close()

    assert runtime._closed is False
    assert runtime._transcriber is transcriber

    await runtime.close()

    assert transcriber.close_calls == 2
    assert runtime._transcriber is None
    assert runtime._closed is True


@pytest.mark.asyncio
async def test_startup_failure_closes_owned_transcriber() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    transcriber = TrackingTranscriber(events)
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=transcriber,
    )

    with (
        patch.object(
            manager,
            "warm_referenced_model_cache",
            AsyncMock(side_effect=RuntimeError("startup failed")),
        ),
        pytest.raises(RuntimeError, match="startup failed"),
    ):
        await runtime.start()

    assert transcriber.close_calls == 1


@pytest.mark.asyncio
async def test_startup_cancellation_cleans_partial_messaging_and_reraises() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    transcriber = TrackingTranscriber(events)
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=transcriber,
    )
    messaging = TrackingMessagingRuntime(events)
    entered = asyncio.Event()

    async def start_messaging() -> None:
        runtime._messaging_runtime = messaging
        entered.set()
        await asyncio.Event().wait()

    with patch.object(
        runtime,
        "_start_messaging_if_configured",
        side_effect=start_messaging,
    ):
        start_task = asyncio.create_task(runtime.start())
        await entered.wait()
        start_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start_task

    assert events == [
        "messaging.quiesce",
        "messaging.close",
        "transcriber.close",
    ]
    assert runtime._closed is True
    assert runtime._messaging_runtime is None
    assert runtime._transcriber is None


@pytest.mark.asyncio
async def test_public_start_retries_transient_partial_messaging_cleanup() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    messaging = TrackingMessagingRuntime(events, fail_quiesce_once=True)
    workflow = MagicMock()
    workflow.close = AsyncMock(side_effect=lambda: events.append("workflow.close"))
    cli_manager = MagicMock()
    components = MessagingPlatformComponents(
        name="tracking",
        runtime=messaging,
        outbound=MagicMock(),
    )
    startup_failure = RuntimeError("partial messaging startup failed")

    async def fail_after_publication(
        published: MessagingPlatformComponents,
    ) -> None:
        assert published is components
        runtime._messaging_runtime = messaging
        runtime._messaging_workflow = workflow
        runtime._cli_manager = cli_manager
        raise startup_failure

    with (
        patch.object(
            manager,
            "warm_referenced_model_cache",
            AsyncMock(),
        ),
        patch.object(manager, "start_model_list_refresh"),
        patch(
            "free_claude_code.runtime.application.messaging_platform_factory.create_messaging_components",
            return_value=components,
        ),
        patch.object(
            runtime,
            "_start_messaging_workflow",
            side_effect=fail_after_publication,
        ),
        pytest.raises(RuntimeError, match="cleanup incomplete") as raised,
    ):
        await runtime.start()

    assert raised.value.__cause__ is startup_failure
    assert events == [
        "messaging.quiesce",
        "messaging.quiesce",
        "workflow.close",
        "messaging.close",
    ]
    workflow.close.assert_awaited_once()
    assert runtime._messaging_runtime is None
    assert runtime._messaging_workflow is None
    assert runtime._cli_manager is None
    assert runtime.is_closed is True


@pytest.mark.asyncio
async def test_public_start_retains_persistently_unclean_partial_messaging_graph() -> (
    None
):
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    transcriber = TrackingTranscriber(events)
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=transcriber,
    )
    messaging = PersistentlyFailingMessagingRuntime(events)
    workflow = MagicMock()
    workflow.close = AsyncMock(side_effect=lambda: events.append("workflow.close"))
    cli_manager = MagicMock()
    components = MessagingPlatformComponents(
        name="tracking",
        runtime=messaging,
        outbound=MagicMock(),
    )
    startup_failure = RuntimeError("partial messaging startup failed")

    async def fail_after_publication(
        published: MessagingPlatformComponents,
    ) -> None:
        assert published is components
        runtime._messaging_runtime = messaging
        runtime._messaging_workflow = workflow
        runtime._cli_manager = cli_manager
        raise startup_failure

    with (
        patch.object(
            manager,
            "warm_referenced_model_cache",
            AsyncMock(),
        ),
        patch.object(manager, "start_model_list_refresh"),
        patch(
            "free_claude_code.runtime.application.messaging_platform_factory.create_messaging_components",
            return_value=components,
        ),
        patch.object(
            runtime,
            "_start_messaging_workflow",
            side_effect=fail_after_publication,
        ),
        pytest.raises(RuntimeError, match="cleanup incomplete") as raised,
    ):
        await runtime.start()

    assert raised.value.__cause__ is startup_failure
    assert events == ["messaging.quiesce", "messaging.quiesce"]
    workflow.close.assert_not_awaited()
    assert transcriber.close_calls == 0
    assert runtime._messaging_runtime is messaging
    assert runtime._messaging_workflow is workflow
    assert runtime._cli_manager is cli_manager
    assert runtime._transcriber is transcriber
    assert runtime._provider_manager_closed is False
    assert runtime.is_closed is False

    messaging.fail_quiesce = False
    assert await runtime.close() is True

    assert events == [
        "messaging.quiesce",
        "messaging.quiesce",
        "messaging.quiesce",
        "workflow.close",
        "messaging.close",
        "transcriber.close",
    ]
    assert runtime.is_closed is True


@pytest.mark.asyncio
async def test_messaging_start_failure_is_nonfatal_after_complete_cleanup() -> None:
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )

    with (
        patch(
            "free_claude_code.runtime.application.messaging_platform_factory.create_messaging_components",
            side_effect=RuntimeError("messaging unavailable"),
        ),
        patch.object(runtime, "_cleanup_messaging", AsyncMock(return_value=True)),
    ):
        await runtime._start_messaging_if_configured()

    await manager.close()


@pytest.mark.asyncio
async def test_messaging_start_failure_fails_closed_when_cleanup_is_incomplete() -> (
    None
):
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )

    with (
        patch(
            "free_claude_code.runtime.application.messaging_platform_factory.create_messaging_components",
            side_effect=RuntimeError("messaging unavailable"),
        ),
        patch.object(runtime, "_cleanup_messaging", AsyncMock(return_value=False)),
        pytest.raises(RuntimeError, match="cleanup incomplete") as exc_info,
    ):
        await runtime._start_messaging_if_configured()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    await manager.close()


@pytest.mark.asyncio
async def test_composition_records_runtime_before_workspace_setup() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    messaging = TrackingMessagingRuntime(events)
    components = MessagingPlatformComponents(
        name="tracking",
        runtime=messaging,
        outbound=MagicMock(),
    )

    with (
        patch(
            "free_claude_code.runtime.application.os.makedirs",
            side_effect=OSError("workspace failed"),
        ),
        pytest.raises(OSError, match="workspace failed"),
    ):
        await runtime._start_messaging_workflow(components)

    assert runtime._messaging_runtime is messaging

    await runtime.close()

    assert events == ["messaging.quiesce", "messaging.close"]
    assert runtime._closed is True


@pytest.mark.asyncio
async def test_composition_publishes_startup_notice_after_runtime_and_repair() -> None:
    events: list[str] = []
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    runtime = ApplicationRuntime(
        manager, configuration=AsyncMock(spec=ConfigurationService), transcriber=None
    )
    messaging = TrackingMessagingRuntime(events)
    notice = MessagingStartupNotice(
        chat_id="chat",
        transport_label="test transport",
    )
    components = MessagingPlatformComponents(
        name="tracking",
        runtime=messaging,
        outbound=MagicMock(),
        startup_notice=notice,
    )
    workflow = MagicMock()
    workflow.handle_message = AsyncMock()
    workflow.restore.side_effect = lambda: events.append("workflow.restore")
    workflow.repair_restored_statuses = AsyncMock(
        side_effect=lambda: events.append("workflow.repair")
    )
    workflow.publish_startup_notice = AsyncMock(
        side_effect=lambda published: events.append("workflow.notice")
    )
    workflow.close = AsyncMock()
    cli_manager = MagicMock()

    with (
        patch(
            "free_claude_code.runtime.application.cli_managed.ManagedClaudeSessionManager",
            return_value=cli_manager,
        ) as manager_constructor,
        patch("free_claude_code.runtime.application.messaging_session.SessionStore"),
        patch(
            "free_claude_code.runtime.application.messaging_workflow_module.MessagingWorkflow",
            return_value=workflow,
        ),
    ):
        await runtime._start_messaging_workflow(components)

    assert events == [
        "workflow.restore",
        "messaging.start",
        "workflow.repair",
        "workflow.notice",
    ]
    workflow.publish_startup_notice.assert_awaited_once_with(notice)
    assert manager_constructor.call_args.kwargs["proxy_root_url"] == (
        "http://127.0.0.1:8082"
    )
    assert "api_url" not in manager_constructor.call_args.kwargs
    assert "plans_directory" not in manager_constructor.call_args.kwargs

    assert await runtime.close() is True


@pytest.mark.asyncio
async def test_folder_picker_is_stopped_before_http_shutdown_drains(monkeypatch):
    import uvicorn

    from free_claude_code.cli.commands import RuntimeServer

    runtime, _manager = _runtime_with_admin_provider(AdminModelProvider())
    started = asyncio.Event()
    signals = []

    async def select(_initial, stop):
        signals.append(stop)
        started.set()
        await stop.wait()
        return None

    monkeypatch.setattr(runtime._folder_picker, "_select", select)
    request = asyncio.create_task(runtime.pick_folder(None))
    await started.wait()

    async def drain(_server, sockets=None):
        assert signals[0].is_set()
        assert await request is None

    monkeypatch.setattr(uvicorn.Server, "shutdown", drain)
    server = RuntimeServer(
        uvicorn.Config("unused:app"), begin_shutdown=runtime.begin_shutdown
    )
    await server.shutdown()
    assert await runtime.close() is True
