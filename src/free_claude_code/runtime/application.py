"""Single owner for application startup, shutdown, and runtime operations."""

import asyncio
import importlib
import inspect
import logging
import os
import traceback
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Literal

from loguru import logger

from free_claude_code.application.code_sessions import CodeService
from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountPort,
    ConnectedAccountStatus,
)
from free_claude_code.application.errors import (
    ApplicationError,
    ApplicationUnavailableError,
    InvalidRequestError,
)
from free_claude_code.application.model_metadata import ProviderModelRefreshResult
from free_claude_code.application.ports import StopResult
from free_claude_code.config.admin.persistence import (
    PreparedAdminUpdate,
)
from free_claude_code.config.admin.state import ConfigInputValue, ValueState
from free_claude_code.config.admin.status import provider_config_status
from free_claude_code.config.loader import clear_settings_cache
from free_claude_code.config.model_refs import parse_provider_type
from free_claude_code.config.paths import (
    claude_desktop_disconnect_path,
    codex_model_catalog_path,
    messaging_state_dir_path,
)
from free_claude_code.config.server_urls import local_admin_url, local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses import (
    claude_desktop_integration,
    claude_integration,
    codex_integration,
    devin_acp_integration,
    jetbrains_acp_integration,
)
from free_claude_code.messaging.platforms import factory as messaging_platform_factory
from free_claude_code.messaging.platforms.factory import MessagingPlatformOptions
from free_claude_code.messaging.platforms.ports import (
    MessagingPlatformComponents,
    MessagingRuntime,
)
from free_claude_code.messaging.voice import Transcriber
from free_claude_code.providers.credential_validation import (
    CredentialStatus,
    check_credentials,
)

if TYPE_CHECKING:
    import free_claude_code.cli.managed as cli_managed
    import free_claude_code.messaging.workflow as messaging_workflow_module

from free_claude_code.application.readiness import InitializationWait
from free_claude_code.core.async_tasks import run_sync_owned

from .configuration import ConfigurationService
from .folder_picker import NativeFolderPicker
from .provider_manager import ProviderRuntimeManager
from .retired_chat import remove_retired_chat_history

RestartCallback = Callable[[], None]
IntegrationAction = Literal["status", "connect", "disconnect", "refresh"]


@dataclass
class _IntegrationUpdate:
    state: Literal["starting", "ready", "failed"] = "ready"
    changed: bool = False
    message: str | None = None
    task: asyncio.Task[None] | None = None

    def snapshot(self) -> JsonObject:
        return {"state": self.state, "changed": self.changed, "message": self.message}

    def complete(self, changed: bool = False) -> None:
        self.state = "ready"
        self.changed = changed
        self.message = None


_PROVIDER_CHECK_FAILURE_MESSAGE = (
    "Could not refresh this provider's models. Verify its configuration and access."
)


async def best_effort(
    name: str,
    awaitable: Awaitable[object],
    *,
    log_verbose_errors: bool = False,
) -> bool:
    """Run one cleanup step and report whether it completed.

    The lifecycle owner intentionally applies no generic timeout here. Cancelling
    an arbitrary cleanup at a deadline can abandon a half-closed SDK, thread, or
    provider resource; resource-specific cleanup or the process supervisor owns
    any force-termination deadline.
    """
    try:
        await awaitable
    except Exception as exc:
        if log_verbose_errors:
            logger.warning(
                "Shutdown step failed: {}: {}: {}",
                name,
                type(exc).__name__,
                exc,
            )
        else:
            logger.warning(
                "Shutdown step failed: {}: exc_type={}",
                name,
                type(exc).__name__,
            )
        return False
    return True


def startup_failure_message(settings: Settings, exc: Exception) -> str:
    """Return the existing concise ASGI startup failure message."""
    if isinstance(exc, ApplicationUnavailableError):
        return exc.message.strip() or "Server startup failed."
    if settings.log_api_error_tracebacks:
        return f"{type(exc).__name__}: {exc}"
    return f"Server startup failed: exc_type={type(exc).__name__}"


async def _await_owned_task[T](
    task: asyncio.Task[T],
    *,
    cancel_on_interrupt: Callable[[], bool] | None = None,
) -> T:
    """Keep ownership until a task settles, then propagate caller cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            # wait never cancels the owned task or logs its exception on interruption.
            await asyncio.wait({task})
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
                if cancel_on_interrupt is not None and cancel_on_interrupt():
                    task.cancel()
    try:
        result = task.result()
    except BaseException as exc:
        if cancellation is not None:
            if not isinstance(exc, asyncio.CancelledError):
                logger.warning(
                    "Cancelled runtime operation failed: exc_type={}",
                    type(exc).__name__,
                )
            raise cancellation from exc
        raise
    if cancellation is not None:
        raise cancellation
    return result


class ApplicationRuntime:
    """Own every process-lifetime resource used by one server instance."""

    def __init__(
        self,
        provider_manager: ProviderRuntimeManager,
        *,
        configuration: ConfigurationService,
        transcriber: Transcriber | None,
        code_service: CodeService | None = None,
        transcriber_factory: Callable[[Settings], Awaitable[Transcriber | None]]
        | None = None,
        restart_callback: RestartCallback | None = None,
        connected_accounts: Mapping[str, ConnectedAccountPort] | None = None,
    ) -> None:
        self.provider_manager = provider_manager
        self._configuration = configuration
        self._code_service = code_service
        self._folder_picker = NativeFolderPicker()
        self._transcriber = transcriber
        self._transcriber_factory = transcriber_factory
        self._restart_callback = restart_callback
        self._connected_accounts = dict(connected_accounts or {})
        self._connected_account_revisions = {
            provider_id: manager.status().revision
            for provider_id, manager in self._connected_accounts.items()
        }
        self._config_lock = asyncio.Lock()
        self._pending_fields: list[str] = []
        self._messaging_runtime: MessagingRuntime | None = None
        self._messaging_workflow: messaging_workflow_module.MessagingWorkflow | None = (
            None
        )
        self._cli_manager: cli_managed.ManagedClaudeSessionManager | None = None
        self._started = False
        self._instance_id = uuid.uuid4().hex
        self._draining = False
        self._closed = False
        self._provider_manager_closed = False
        self._connected_accounts_closed = False
        self._lifecycle_lock = asyncio.Lock()
        self._startup_tasks: list[asyncio.Task[None]] = []
        self._claude_update = _IntegrationUpdate()
        self._desktop_update = _IntegrationUpdate()
        self._codex_update = _IntegrationUpdate()
        self._jetbrains_update = _IntegrationUpdate()
        self._devin_update = _IntegrationUpdate()
        self._http_ready = asyncio.Event()
        self._messaging_state = (
            "disabled" if self.settings.messaging_platform == "none" else "starting"
        )
        self._messaging_error: str | None = None

    @property
    def settings(self) -> Settings:
        return self.provider_manager.current_settings()

    @property
    def is_closed(self) -> bool:
        """Whether this runtime released its complete ownership graph."""
        return self._closed

    async def start(self) -> None:
        try:
            async with self._lifecycle_lock:
                if self._draining:
                    raise ApplicationUnavailableError(
                        "Application runtime is shutting down."
                    )
                if self._started:
                    return
                logger.info("Starting Claude Code Proxy...")
                await _await_owned_task(
                    asyncio.create_task(self._configuration.initialize())
                )
                if self._draining:
                    raise ApplicationUnavailableError(
                        "Application runtime is shutting down."
                    )
                self.provider_manager.start_model_list_refresh()
                await self.refresh_claude_vscode()
                await self.refresh_codex_integration()
                await self.refresh_claude_desktop()
                await self.refresh_jetbrains_acp()
                await self.refresh_devin_acp()
                self._startup_tasks.append(
                    asyncio.create_task(
                        run_sync_owned(remove_retired_chat_history),
                        name="fcc-retired-chat-cleanup",
                    )
                )
                if self._code_service is not None:
                    self._startup_tasks.append(
                        asyncio.create_task(
                            self._code_service.start(), name="fcc-code-startup"
                        )
                    )
                self._startup_tasks.append(
                    asyncio.create_task(
                        self._start_messaging_if_configured(),
                        name="fcc-messaging-startup",
                    )
                )
                self._started = True
        except asyncio.CancelledError:
            await self.close()
            raise
        except Exception as exc:
            logger.error(
                "Startup failed:\n{}", startup_failure_message(self.settings, exc)
            )
            await self.close()
            raise

    def http_started(self) -> None:
        """Called by the server after lifespan and socket adoption, not at reservation."""
        if self._draining or self._http_ready.is_set():
            return
        self._http_ready.set()
        logging.getLogger("uvicorn.error").info(
            "Admin UI: %s (local-only)", local_admin_url(self.settings)
        )

    def begin_shutdown(self) -> None:
        """Finish indefinite observer responses before the server drains HTTP."""
        self._draining = True
        self.provider_manager.begin_shutdown()
        self._folder_picker.begin_shutdown()
        if self._code_service is not None:
            self._code_service.begin_shutdown()

    async def close(self) -> bool:
        self.begin_shutdown()
        async with self._lifecycle_lock:
            if self._closed:
                return True
            logger.info("Shutdown requested, cleaning up...")
            for task in self._startup_tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(*self._startup_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(
                        "Background initialization ended with exc_type={}",
                        type(result).__name__,
                    )
            self._startup_tasks.clear()
            async with self._config_lock:
                self._closed = await self._close_owned_resources()
            if self._closed:
                self._started = False
                logger.info("Server shut down cleanly")
            else:
                logger.warning(
                    "Server shutdown incomplete; owned resources remain for retry"
                )
            return self._closed

    async def pick_folder(self, initial_path: str | None) -> str | None:
        return await self._folder_picker.pick_folder(initial_path)

    async def apply_admin_config(
        self,
        updates: Mapping[str, ConfigInputValue],
    ) -> JsonObject:
        """Apply one validated config update without splitting runtime ownership."""
        caller = asyncio.current_task()
        assert caller is not None
        initial_cancellations = caller.cancelling()
        async with self._config_lock:
            if self._draining:
                raise ApplicationUnavailableError(
                    "Configuration runtime is shutting down."
                )
            prepared = await self._configuration.prepare(updates, self.settings)
            if not prepared.valid:
                return prepared.applied_response() | {"credential_checks": []}
            assert prepared.settings is not None

            checks = await check_credentials(prepared.settings, prepared.changed_keys)
            check_response: list[JsonObject] = [
                {
                    "key": check.key,
                    "status": check.status.value,
                    "message": check.message,
                }
                for check in checks
            ]
            rejected = [
                check for check in checks if check.status == CredentialStatus.REJECTED
            ]
            if rejected:
                return prepared.validation_response() | {
                    "applied": False,
                    "valid": False,
                    "errors": [f"{check.key}: {check.message}" for check in rejected],
                    "pending_fields": [],
                    "credential_checks": check_response,
                }

            persistence_started = False

            async def commit() -> JsonObject:
                nonlocal persistence_started
                # The caller's cancellation wakeup may run after finalization starts.
                if caller.cancelling() > initial_cancellations:
                    raise asyncio.CancelledError
                persistence_started = True
                return await self._commit_admin_update(prepared)

            finalization = asyncio.create_task(
                self._finalize_admin_update(prepared, check_response, commit)
            )
            return await _await_owned_task(
                finalization,
                cancel_on_interrupt=lambda: not persistence_started,
            )

    async def _finalize_admin_update(
        self,
        prepared: PreparedAdminUpdate,
        check_response: list[JsonObject],
        commit: Callable[[], Awaitable[JsonObject]],
    ) -> JsonObject:
        assert prepared.settings is not None
        if prepared.pending_fields:
            result = await commit()
        else:
            result: JsonObject = {}

            async def publish_commit() -> None:
                result.update(await commit())

            await self.provider_manager.replace(
                prepared.settings,
                commit=publish_commit,
                reason="admin_apply",
            )
        self._pending_fields = list(prepared.pending_fields)
        automatic = bool(prepared.pending_fields and self._signal_restart())
        if automatic:
            self._pending_fields = []
        result["restart"] = self._restart_metadata(
            prepared.pending_fields,
            prepared.settings,
            automatic=automatic,
        )
        result["credential_checks"] = check_response
        return result

    async def admin_config(self) -> JsonObject:
        return await self._configuration.admin_config()

    async def admin_values(self) -> ValueState:
        return await self._configuration.admin_values()

    async def claude_vscode_status(self) -> JsonObject:
        return await self._claude_vscode("status")

    async def connect_claude_vscode(self) -> JsonObject:
        return await self._claude_vscode("connect")

    async def disconnect_claude_vscode(self) -> JsonObject:
        return await self._claude_vscode("disconnect")

    def _check_integration_available(self) -> None:
        if self._draining or self._pending_fields:
            raise ApplicationUnavailableError(
                "Wait for FCC to restart before changing the integration."
            )

    def _start_integration_update(
        self,
        update: _IntegrationUpdate,
        operation: Callable[[], Awaitable[JsonObject]],
    ) -> JsonObject:
        self._check_integration_available()
        if update.task is None or update.task.done():
            update.state, update.changed, update.message = "starting", False, None

            async def run() -> None:
                try:
                    result = await operation()
                    update.complete(result.get("changed") is True)
                except ApplicationError as exc:
                    update.state, update.message = "failed", exc.message
                except Exception as exc:
                    update.state = "failed"
                    update.message = (
                        "Could not update integration settings. Retry shortly."
                    )
                    logger.warning(
                        "Integration update failed: exc_type={}", type(exc).__name__
                    )

            update.task = asyncio.create_task(run(), name="fcc-integration-update")
            self._startup_tasks.append(update.task)
            update.task.add_done_callback(self._startup_tasks.remove)
        return {"update": update.snapshot()}

    async def refresh_claude_vscode(self) -> JsonObject:
        return self._start_integration_update(
            self._claude_update, partial(self._claude_vscode, "refresh")
        )

    async def refresh_codex_integration(self) -> JsonObject:
        return self._start_integration_update(
            self._codex_update, partial(self._codex_integration, "refresh")
        )

    async def jetbrains_acp_status(self) -> JsonObject:
        return await self._jetbrains_acp("status")

    async def connect_jetbrains_acp(self) -> JsonObject:
        return await self._jetbrains_acp("connect")

    async def disconnect_jetbrains_acp(self) -> JsonObject:
        return await self._jetbrains_acp("disconnect")

    async def refresh_jetbrains_acp(self) -> JsonObject:
        return self._start_integration_update(
            self._jetbrains_update, partial(self._jetbrains_acp, "refresh")
        )

    async def _jetbrains_acp(self, action: IntegrationAction) -> JsonObject:
        async with self._config_lock:
            self._check_integration_available()
            settings = self.settings

            def operate() -> JsonObject:
                path = jetbrains_acp_integration.config_path()
                url = local_proxy_root_url(settings)
                if action == "refresh":
                    return {
                        "changed": jetbrains_acp_integration.refresh_connected(
                            path, url, settings.proxy_auth_token
                        )
                    }
                return jetbrains_acp_integration.configure(
                    path,
                    url,
                    settings.proxy_auth_token,
                    None if action == "status" else action == "connect",
                )

            try:
                result = await run_sync_owned(operate)
            except jetbrains_acp_integration.SetupError as exc:
                raise InvalidRequestError(str(exc)) from None
            except ValueError, UnicodeError:
                raise InvalidRequestError(
                    "Could not read JetBrains ACP configuration. Check the JSON in acp.json and the installed Claude Agent metadata."
                ) from None
            except OSError:
                raise ApplicationUnavailableError(
                    "Could not access JetBrains ACP files. Finish any configuration edits, check file permissions, and retry."
                ) from None
            if action in {"connect", "disconnect"}:
                self._jetbrains_update.complete()
            if action == "status":
                result["update"] = self._jetbrains_update.snapshot()
            return result

    async def devin_acp_status(self) -> JsonObject:
        return await self._devin_acp("status")

    async def connect_devin_acp(self) -> JsonObject:
        return await self._devin_acp("connect")

    async def disconnect_devin_acp(self) -> JsonObject:
        return await self._devin_acp("disconnect")

    async def refresh_devin_acp(self) -> JsonObject:
        return self._start_integration_update(
            self._devin_update, partial(self._devin_acp, "refresh")
        )

    async def _devin_acp(self, action: IntegrationAction) -> JsonObject:
        async with self._config_lock:
            self._check_integration_available()

            def operate() -> JsonObject:
                path = devin_acp_integration.config_path()
                if action == "refresh":
                    return {"changed": devin_acp_integration.refresh_connected(path)}
                return devin_acp_integration.configure(
                    path,
                    None if action == "status" else action == "connect",
                )

            try:
                result = await run_sync_owned(operate)
            except devin_acp_integration.SetupError as exc:
                raise InvalidRequestError(str(exc)) from None
            except ValueError, UnicodeError:
                raise InvalidRequestError(
                    "Could not read Devin ACP configuration. Check the JSON in registry.json."
                ) from None
            except OSError:
                raise ApplicationUnavailableError(
                    "Could not access Devin ACP files. Finish any configuration edits, check file permissions, and retry."
                ) from None
            if action in {"connect", "disconnect"}:
                self._devin_update.complete()
            if action == "status":
                result["update"] = self._devin_update.snapshot()
            return result

    async def claude_desktop_status(self) -> JsonObject:
        return await self._claude_desktop("status")

    async def connect_claude_desktop(self) -> JsonObject:
        return await self._claude_desktop("connect")

    async def disconnect_claude_desktop(self) -> JsonObject:
        return await self._claude_desktop("disconnect")

    async def refresh_claude_desktop(self) -> JsonObject:
        return self._start_integration_update(
            self._desktop_update, partial(self._claude_desktop, "refresh")
        )

    async def _claude_desktop(self, action: IntegrationAction) -> JsonObject:
        async with self._config_lock:
            self._check_integration_available()
            settings = self.settings

            def operate() -> JsonObject:
                root = claude_desktop_integration.config_root()
                url = local_proxy_root_url(settings)
                if action == "refresh":
                    return {
                        "changed": claude_desktop_integration.refresh_connected(
                            root,
                            url,
                            settings.proxy_auth_token,
                            disconnect_path=claude_desktop_disconnect_path(),
                        )
                    }
                return claude_desktop_integration.configure(
                    root,
                    url,
                    settings.proxy_auth_token,
                    None if action == "status" else action == "connect",
                    disconnect_path=claude_desktop_disconnect_path(),
                )

            try:
                result = await run_sync_owned(operate)
            except claude_desktop_integration.ManagedDesktopError:
                raise InvalidRequestError(
                    "Claude Desktop is managed by an organization, or its policy could not be read. FCC can configure only unmanaged Desktop installations."
                ) from None
            except claude_desktop_integration.PendingDisconnectError:
                raise InvalidRequestError(
                    "Finish disconnecting Claude Desktop before connecting again."
                ) from None
            except claude_desktop_integration.PendingMigrationError:
                raise InvalidRequestError(
                    "Claude Desktop has data in its previous Windows location. Launch Claude Desktop once so it can migrate that data, fully quit it, then retry Connect."
                ) from None
            except ValueError, UnicodeError:
                raise InvalidRequestError(
                    "Could not configure Claude Desktop. Check its configuration JSON and FCC disconnect record, and ensure FCC uses a localhost address and a nonempty managed token."
                ) from None
            except OSError:
                raise ApplicationUnavailableError(
                    "Could not access Claude Desktop settings or the FCC disconnect record. Fully quit Claude Desktop, check file permissions, and retry."
                ) from None
            if action in {"connect", "disconnect"}:
                self._desktop_update.complete()
            if action == "status":
                if self._desktop_update.state != "ready":
                    result["connected"] = None
                result["update"] = self._desktop_update.snapshot()
            return result

    async def _claude_vscode(self, action: IntegrationAction) -> JsonObject:
        async with self._config_lock:
            self._check_integration_available()
            settings = self.settings

            def operate() -> JsonObject:
                path = claude_integration.settings_path()
                state_path = claude_integration.claude_state_path()
                if action == "status" and self._claude_update.state != "ready":
                    return {
                        "connected": None,
                        "paths": {
                            "vscode_settings": str(path.resolve()),
                            "claude_state": str(state_path.resolve()),
                        },
                    }
                if action == "refresh":
                    return {
                        "changed": claude_integration.refresh_connected(
                            path,
                            state_path,
                            local_proxy_root_url(settings),
                            settings.proxy_auth_token,
                        )
                    }
                return claude_integration.configure(
                    path,
                    state_path,
                    local_proxy_root_url(settings),
                    settings.proxy_auth_token,
                    None if action == "status" else action == "connect",
                )

            try:
                result = await run_sync_owned(operate)
            except ValueError, UnicodeError:
                raise InvalidRequestError(
                    "Could not read Claude integration settings. Check the JSON in VS Code settings.json and .claude.json."
                ) from None
            except OSError:
                raise ApplicationUnavailableError(
                    "Could not access VS Code settings.json or .claude.json. Check file permissions and try again."
                ) from None
            if action in {"connect", "disconnect"}:
                self._claude_update.complete()
            if action == "status":
                result["update"] = self._claude_update.snapshot()
            return result

    async def codex_integration_status(self) -> JsonObject:
        return await self._codex_integration("status")

    async def connect_codex(self) -> JsonObject:
        return await self._codex_integration("connect")

    async def disconnect_codex(self) -> JsonObject:
        return await self._codex_integration("disconnect")

    async def _codex_integration(self, action: IntegrationAction) -> JsonObject:
        wait = InitializationWait(None) if action == "refresh" else InitializationWait()
        needs_catalog = action in {"connect", "refresh"}
        try:
            if action == "refresh":
                async with self._config_lock:
                    self._check_integration_available()
                    url = local_proxy_root_url(self.settings)
                    if not await run_sync_owned(
                        lambda: codex_integration.recognizes_connection(
                            codex_integration.config_path(), url
                        )
                    ):
                        return {"changed": False}
            while True:
                generation_id = (
                    await self.provider_manager.wait_for_catalog_file(wait)
                    if needs_catalog
                    else None
                )
                async with self._config_lock:
                    self._check_integration_available()
                    if needs_catalog and (
                        generation_id != self.provider_manager.current_generation_id
                        or self.provider_manager.catalog_status()["catalog"] != "ready"
                    ):
                        continue
                    url = local_proxy_root_url(self.settings)

                    def operate(url: str = url) -> JsonObject:
                        path = codex_integration.config_path()
                        if action == "status" and self._codex_update.state != "ready":
                            return {
                                "connected": None,
                                "paths": {"codex_config": str(path.resolve())},
                            }
                        if action == "refresh":
                            return {
                                "changed": codex_integration.refresh_connected(
                                    path, codex_model_catalog_path(), url
                                )
                            }
                        return codex_integration.configure(
                            path,
                            codex_model_catalog_path(),
                            url,
                            None if action == "status" else action == "connect",
                        )

                    result = await run_sync_owned(operate)
                    if action in {"connect", "disconnect"}:
                        self._codex_update.complete()
                    if action == "status":
                        result["update"] = self._codex_update.snapshot()
                    return result
        except ValueError, UnicodeError:
            raise InvalidRequestError(
                "Could not read Codex settings. Check the TOML in config.toml."
            ) from None
        except OSError:
            raise ApplicationUnavailableError(
                "Could not access Codex config.toml. Check file permissions and try again."
            ) from None

    async def admin_status(self) -> JsonObject:
        values = await self.admin_values()
        settings = self.settings
        return {
            "status": "stopping" if self._draining else "running",
            "instance_id": self._instance_id,
            "startup": {
                **self.provider_manager.catalog_status(),
                "code": self._code_service.storage_status()
                if self._code_service
                else {"state": "disabled"},
                "messaging": {
                    "state": self._messaging_state,
                    "message": self._messaging_error,
                },
                "integrations": {
                    "claude-vscode": self._claude_update.snapshot(),
                    "claude-desktop": self._desktop_update.snapshot(),
                    "codex": self._codex_update.snapshot(),
                    "jetbrains-acp": self._jetbrains_update.snapshot(),
                    "devin-acp": self._devin_update.snapshot(),
                },
            },
            "host": settings.host,
            "port": settings.port,
            "model": settings.model,
            "provider": parse_provider_type(settings.model),
            "pending_fields": list(self._pending_fields),
            "provider_status": provider_config_status(values),
            "cached_models": {
                provider_id: sorted(model_ids)
                for provider_id, model_ids in self.provider_manager.cached_model_ids().items()
            },
        }

    async def test_provider(self, provider_id: str) -> JsonObject:
        result = await self.provider_manager.refresh_provider(provider_id)
        if result.failed_provider_ids:
            return {
                "provider_id": provider_id,
                "ok": False,
                "message": _PROVIDER_CHECK_FAILURE_MESSAGE,
            }
        return {
            "provider_id": provider_id,
            "ok": True,
            "models": sorted(
                self.provider_manager.cached_model_ids().get(provider_id, ())
            ),
        }

    async def refresh_models(self) -> ProviderModelRefreshResult:
        return await self.provider_manager.refresh_model_list_cache()

    async def connected_account_status(
        self, provider_id: str
    ) -> ConnectedAccountStatus:
        """Return safe account state and synchronize model availability."""

        manager = self._connected_account(provider_id)
        status = manager.status()
        previous_revision = self._connected_account_revisions.get(provider_id)
        if status.revision != previous_revision:
            await self.provider_manager.connected_provider_changed(
                provider_id, connected=status.connected
            )
            self._connected_account_revisions[provider_id] = status.revision
        model_count = len(self.provider_manager.cached_model_ids().get(provider_id, ()))
        return replace(status, model_count=model_count)

    async def start_connected_account_login(
        self,
        provider_id: str,
        mode: ConnectedAccountLoginMode,
    ) -> ConnectedAccountStatus:
        """Start one provider-owned interactive login."""

        return await self._connected_account(provider_id).start_login(mode)

    async def cancel_connected_account_login(
        self, provider_id: str
    ) -> ConnectedAccountStatus:
        """Cancel one pending provider login."""

        return await self._connected_account(provider_id).cancel_login()

    async def disconnect_connected_account(
        self, provider_id: str
    ) -> ConnectedAccountStatus:
        """Disconnect an account and evict only that provider's models."""

        status = await self._connected_account(provider_id).disconnect()
        await self.provider_manager.connected_provider_changed(
            provider_id, connected=False
        )
        self._connected_account_revisions[provider_id] = status.revision
        return status

    def _signal_restart(self) -> bool:
        """Invoke a synchronous signal; failure leaves the saved change pending."""
        callback = self._restart_callback
        if callback is None:
            return False
        try:
            result = callback()
            # Enforce the contract for dynamically supplied callbacks as well.
            # Never execute an async callback that could await runtime.close().
            if inspect.iscoroutine(result):
                result.close()
            if result is not None:
                raise TypeError(
                    "Restart callback must signal synchronously and return None."
                )
        except Exception as exc:
            logger.warning(
                "Config saved but restart signal failed: exc_type={}",
                type(exc).__name__,
            )
            return False
        return True

    async def stop_all(self) -> StopResult | None:
        if self._messaging_workflow is not None:
            outcome = await self._messaging_workflow.stop_all_tasks()
            return StopResult(cancelled_count=outcome.cancelled_count)
        if self._cli_manager is not None:
            await self._cli_manager.stop_all()
            return StopResult(source="cli_manager")
        return None

    async def _commit_admin_update(
        self,
        prepared: PreparedAdminUpdate,
    ) -> JsonObject:
        result = await self._configuration.commit(prepared)
        clear_settings_cache()
        return result

    def _restart_metadata(
        self,
        fields: tuple[str, ...],
        settings: Settings,
        *,
        automatic: bool,
    ) -> JsonObject:
        result: JsonObject = {
            "required": bool(fields),
            "automatic": automatic,
            "admin_url": local_admin_url(settings) if automatic else None,
            "fields": list(fields),
        }
        if automatic:
            result["instance_id"] = self._instance_id
        return result

    async def _start_messaging_if_configured(self) -> None:
        if self.settings.messaging_platform == "none":
            return
        try:

            def load_modules() -> None:
                for name in ("cli.managed", "messaging.session", "messaging.workflow"):
                    importlib.import_module(f"free_claude_code.{name}")
                importlib.import_module(
                    f"free_claude_code.messaging.platforms.{self.settings.messaging_platform}"
                )

            await run_sync_owned(load_modules)
            if self._transcriber_factory is not None:
                self._transcriber = await self._transcriber_factory(self.settings)
            components = messaging_platform_factory.create_messaging_components(
                self.settings.messaging_platform,
                self._messaging_options(),
            )
            if components is not None:
                await self._start_messaging_workflow(components)
                self._messaging_state = "ready"
            else:
                self._messaging_state = "disabled"
        except ImportError as exc:
            self._messaging_state = "failed"
            self._messaging_error = (
                "Messaging could not start. Check its configuration and restart FCC."
            )
            cleaned = await self._cleanup_messaging()
            if self.settings.log_api_error_tracebacks:
                logger.warning("Messaging module import error: {}", exc)
            else:
                logger.warning(
                    "Messaging module import error: exc_type={}",
                    type(exc).__name__,
                )
            if not cleaned:
                raise RuntimeError("Messaging startup cleanup incomplete") from exc
        except Exception as exc:
            self._messaging_state = "failed"
            self._messaging_error = (
                "Messaging could not start. Check its configuration and restart FCC."
            )
            cleaned = await self._cleanup_messaging()
            if self.settings.log_api_error_tracebacks:
                logger.error("Failed to start messaging platform: {}", exc)
                logger.error(traceback.format_exc())
            else:
                logger.error(
                    "Failed to start messaging platform: exc_type={}",
                    type(exc).__name__,
                )
            if not cleaned:
                raise RuntimeError("Messaging startup cleanup incomplete") from exc

    def _messaging_options(self) -> MessagingPlatformOptions:
        settings = self.settings
        return MessagingPlatformOptions(
            telegram_bot_token=settings.telegram_bot_token,
            allowed_telegram_user_id=settings.allowed_telegram_user_id,
            telegram_proxy_url=settings.telegram_proxy_url,
            discord_bot_token=settings.discord_bot_token,
            allowed_discord_channels=settings.allowed_discord_channels,
            transcriber=self._transcriber,
            messaging_rate_limit=settings.messaging_rate_limit,
            messaging_rate_window=settings.messaging_rate_window,
            log_raw_messaging_content=settings.log_raw_messaging_content,
            log_messaging_error_details=settings.log_messaging_error_details,
            log_api_error_tracebacks=settings.log_api_error_tracebacks,
        )

    async def _start_messaging_workflow(
        self,
        components: MessagingPlatformComponents,
    ) -> None:
        import free_claude_code.cli.managed as cli_managed
        import free_claude_code.messaging.session as messaging_session
        import free_claude_code.messaging.workflow as messaging_workflow_module

        settings = self.settings
        self._messaging_runtime = components.runtime
        workspace = (
            os.path.abspath(settings.allowed_dir)
            if settings.allowed_dir
            else os.getcwd()
        )
        await run_sync_owned(partial(os.makedirs, workspace, exist_ok=True))
        data_path = os.path.abspath(messaging_state_dir_path())
        await run_sync_owned(partial(os.makedirs, data_path, exist_ok=True))
        allowed_dirs = [workspace] if settings.allowed_dir else []

        self._cli_manager = cli_managed.ManagedClaudeSessionManager(
            workspace_path=workspace,
            proxy_root_url=local_proxy_root_url(settings),
            allowed_dirs=allowed_dirs,
            auth_token=settings.proxy_auth_token,
            log_raw_cli_diagnostics=settings.log_raw_cli_diagnostics,
            log_messaging_error_details=settings.log_messaging_error_details,
        )
        session_store = await run_sync_owned(
            partial(
                messaging_session.SessionStore,
                storage_path=os.path.join(data_path, "sessions.json"),
                managed_message_cap=settings.max_message_log_entries_per_chat,
            )
        )
        workflow = messaging_workflow_module.MessagingWorkflow(
            platform_name=components.name,
            outbound=components.outbound,
            voice_cancellation=components.voice_cancellation,
            cli_manager=self._cli_manager,
            session_store=session_store,
            debug_platform_edits=settings.debug_platform_edits,
            debug_subagent_stack=settings.debug_subagent_stack,
            log_raw_cli_diagnostics=settings.log_raw_cli_diagnostics,
            log_messaging_error_details=settings.log_messaging_error_details,
        )
        self._messaging_workflow = workflow
        workflow.restore()
        components.runtime.on_message(workflow.handle_message)
        await self._http_ready.wait()
        if self._draining:
            return
        await components.runtime.start()
        await workflow.repair_restored_statuses()
        if components.startup_notice is not None:
            await workflow.publish_startup_notice(components.startup_notice)
        logger.info("{} platform started with messaging workflow", components.name)

    async def _close_owned_resources(self) -> bool:
        if not await best_effort("folder_picker.close", self._folder_picker.close()):
            return False
        if not await self._cleanup_messaging():
            return False
        verbose = self.settings.log_api_error_tracebacks
        if self._code_service is not None and not await best_effort(
            "code_service.close",
            self._code_service.close(),
            log_verbose_errors=verbose,
        ):
            return False
        if not await self._cleanup_transcriber():
            return False
        if not self._provider_manager_closed:
            self._provider_manager_closed = await best_effort(
                "provider_manager.close",
                self.provider_manager.close(),
                log_verbose_errors=verbose,
            )
            if not self._provider_manager_closed:
                return False
        if self._connected_accounts_closed:
            return True
        results = await asyncio.gather(
            *(
                best_effort(
                    f"connected_account.{provider_id}.close",
                    manager.close(),
                    log_verbose_errors=verbose,
                )
                for provider_id, manager in self._connected_accounts.items()
            )
        )
        self._connected_accounts_closed = all(results)
        return self._connected_accounts_closed

    def _connected_account(self, provider_id: str) -> ConnectedAccountPort:
        manager = self._connected_accounts.get(provider_id)
        if manager is None:
            raise ApplicationUnavailableError(
                f"Provider {provider_id!r} does not support connected-account login."
            )
        return manager

    async def _cleanup_messaging(self) -> bool:
        verbose = self.settings.log_api_error_tracebacks
        workflow = self._messaging_workflow
        runtime = self._messaging_runtime
        cli_manager = self._cli_manager

        if runtime is not None:
            quiesced = await best_effort(
                "messaging_runtime.quiesce",
                runtime.quiesce(),
                log_verbose_errors=verbose,
            )
            if not quiesced:
                # Delivery must remain available until ingress is known stopped.
                # Retaining the graph lets the next close retry this exact gate.
                return False

        if workflow is not None:
            closed = await best_effort(
                "messaging_workflow.close",
                workflow.close(),
                log_verbose_errors=verbose,
            )
            if not closed:
                # Active workflow tasks may still need delivery, transcription,
                # CLI sessions, and providers while a later close retries drain.
                return False
            if self._messaging_workflow is workflow:
                self._messaging_workflow = None
            if self._cli_manager is cli_manager:
                self._cli_manager = None
        elif cli_manager is not None:
            drained = await best_effort(
                "cli_manager.stop_all",
                cli_manager.stop_all(),
                log_verbose_errors=verbose,
            )
            if not drained:
                return False
            if self._cli_manager is cli_manager:
                self._cli_manager = None

        if runtime is not None:
            closed = await best_effort(
                "messaging_runtime.close",
                runtime.close(),
                log_verbose_errors=verbose,
            )
            if not closed:
                return False
            if self._messaging_runtime is runtime:
                self._messaging_runtime = None
        return True

    async def _cleanup_transcriber(self) -> bool:
        transcriber = self._transcriber
        if transcriber is None:
            return True
        closed = await best_effort(
            "transcriber.close",
            transcriber.close(),
            log_verbose_errors=self.settings.log_api_error_tracebacks,
        )
        if closed and self._transcriber is transcriber:
            self._transcriber = None
        return closed
