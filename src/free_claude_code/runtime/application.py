"""Single owner for application startup, shutdown, and runtime operations."""

import asyncio
import inspect
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace

from loguru import logger

from free_claude_code.application.code_sessions import CodeService
from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountPort,
    ConnectedAccountStatus,
)
from free_claude_code.application.errors import (
    ApplicationUnavailableError,
)
from free_claude_code.application.model_metadata import ProviderModelRefreshResult
from free_claude_code.application.ports import StopResult
from free_claude_code.config.admin.custom_providers import CustomProviderMutation
from free_claude_code.config.admin.persistence import (
    PreparedAdminUpdate,
)
from free_claude_code.config.admin.state import ConfigInputValue, ValueState
from free_claude_code.config.admin.status import provider_config_status
from free_claude_code.config.loader import clear_settings_cache
from free_claude_code.config.model_refs import parse_provider_type
from free_claude_code.config.server_urls import local_admin_url
from free_claude_code.config.settings import Settings
from free_claude_code.core.async_tasks import run_sync_owned
from free_claude_code.core.json_types import JsonObject
from free_claude_code.messaging.voice import Transcriber
from free_claude_code.providers.credential_validation import (
    CredentialStatus,
    check_credentials,
)

from .configuration import ConfigurationService
from .folder_picker import NativeFolderPicker
from .integration_service import IntegrationService
from .lifecycle import best_effort
from .messaging_service import MessagingService
from .provider_manager import ProviderRuntimeManager
from .retired_chat import remove_retired_chat_history
from .sqlite_database import SQLiteDatabase

RestartCallback = Callable[[], None]


_PROVIDER_CHECK_FAILURE_MESSAGE = (
    "Could not refresh this provider's models. Verify its configuration and access."
)


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
        database: SQLiteDatabase | None = None,
        transcriber_factory: Callable[[Settings], Awaitable[Transcriber | None]]
        | None = None,
        restart_callback: RestartCallback | None = None,
        connected_accounts: Mapping[str, ConnectedAccountPort] | None = None,
    ) -> None:
        self.provider_manager = provider_manager
        self._configuration = configuration
        self._code_service = code_service
        self._database = database
        self._folder_picker = NativeFolderPicker()
        self._restart_callback = restart_callback
        self._connected_accounts = dict(connected_accounts or {})
        self._connected_account_revisions = {
            provider_id: manager.status().revision
            for provider_id, manager in self._connected_accounts.items()
        }
        self._config_lock = asyncio.Lock()
        self._pending_fields: list[str] = []
        self._started = False
        self._instance_id = uuid.uuid4().hex
        self._draining = False
        self._closed = False
        self._provider_manager_closed = False
        self._connected_accounts_closed = False
        self._lifecycle_lock = asyncio.Lock()
        self._startup_tasks: list[asyncio.Task[None]] = []
        self._http_ready = asyncio.Event()
        self._integrations = IntegrationService(
            provider_manager,
            config_lock=self._config_lock,
            is_draining=lambda: self._draining,
            has_pending_restart=lambda: bool(self._pending_fields),
        )
        self._messaging = MessagingService(
            settings=provider_manager.current_settings,
            database=database,
            http_ready=self._http_ready,
            is_draining=lambda: self._draining,
            transcriber=transcriber,
            transcriber_factory=transcriber_factory,
        )

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
                self.provider_manager.set_catalog_changed_callback(
                    self._integrations.catalog_changed
                )
                await self._integrations.start()
                self._startup_tasks.append(
                    asyncio.create_task(
                        run_sync_owned(remove_retired_chat_history),
                        name="fcc-retired-chat-cleanup",
                    )
                )
                self._messaging.start()
                if self._code_service is not None:
                    self._startup_tasks.append(
                        asyncio.create_task(
                            self._code_service.start(), name="fcc-code-startup"
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
        self.provider_manager.set_catalog_changed_callback(None)
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
            self._integrations.cancel_background_tasks()
            self._messaging.cancel_background_tasks()
            for task in self._startup_tasks:
                if not task.done():
                    task.cancel()
            await self._integrations.wait_background_tasks()
            await self._messaging.wait_background_tasks()
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
        custom_provider: CustomProviderMutation | None = None,
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
            prepared = await self._configuration.prepare(
                updates, self.settings, custom_provider
            )
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

    async def vscode_chat_status(self) -> JsonObject:
        return await self._integrations.vscode_chat_status()

    async def connect_vscode_chat(self) -> JsonObject:
        return await self._integrations.connect_vscode_chat()

    async def disconnect_vscode_chat(self) -> JsonObject:
        return await self._integrations.disconnect_vscode_chat()

    async def refresh_vscode_chat(self) -> JsonObject:
        return await self._integrations.refresh_vscode_chat()

    async def claude_vscode_status(self) -> JsonObject:
        return await self._integrations.claude_vscode_status()

    async def connect_claude_vscode(self) -> JsonObject:
        return await self._integrations.connect_claude_vscode()

    async def disconnect_claude_vscode(self) -> JsonObject:
        return await self._integrations.disconnect_claude_vscode()

    async def refresh_claude_vscode(self) -> JsonObject:
        return await self._integrations.refresh_claude_vscode()

    async def refresh_codex_integration(self) -> JsonObject:
        return await self._integrations.refresh_codex_integration()

    async def jetbrains_acp_status(self) -> JsonObject:
        return await self._integrations.jetbrains_acp_status()

    async def connect_jetbrains_acp(self) -> JsonObject:
        return await self._integrations.connect_jetbrains_acp()

    async def disconnect_jetbrains_acp(self) -> JsonObject:
        return await self._integrations.disconnect_jetbrains_acp()

    async def refresh_jetbrains_acp(self) -> JsonObject:
        return await self._integrations.refresh_jetbrains_acp()

    async def claude_desktop_status(self) -> JsonObject:
        return await self._integrations.claude_desktop_status()

    async def connect_claude_desktop(self) -> JsonObject:
        return await self._integrations.connect_claude_desktop()

    async def disconnect_claude_desktop(self) -> JsonObject:
        return await self._integrations.disconnect_claude_desktop()

    async def refresh_claude_desktop(self) -> JsonObject:
        return await self._integrations.refresh_claude_desktop()

    async def codex_integration_status(self) -> JsonObject:
        return await self._integrations.codex_integration_status()

    async def connect_codex(self) -> JsonObject:
        return await self._integrations.connect_codex()

    async def disconnect_codex(self) -> JsonObject:
        return await self._integrations.disconnect_codex()

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
                "messaging": self._messaging.snapshot(),
                "integrations": self._integrations.snapshot(),
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
        return await self._messaging.stop_all()

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

    async def _close_owned_resources(self) -> bool:
        if not await best_effort("folder_picker.close", self._folder_picker.close()):
            return False
        if not await self._messaging.close_delivery():
            return False
        verbose = self.settings.log_api_error_tracebacks
        if self._code_service is not None and not await best_effort(
            "code_service.close",
            self._code_service.close(),
            log_verbose_errors=verbose,
        ):
            return False
        if self._database is not None and not await best_effort(
            "database.close", self._database.close()
        ):
            return False
        if not await self._messaging.close_transcriber():
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
