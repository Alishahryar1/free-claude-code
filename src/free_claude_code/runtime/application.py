"""Single owner for application startup, shutdown, and runtime operations."""

import asyncio
import inspect
import logging
import os
import traceback
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace

from anyio import to_thread
from loguru import logger

import free_claude_code.cli.managed as cli_managed
import free_claude_code.messaging.session as messaging_session
import free_claude_code.messaging.workflow as messaging_workflow_module
from free_claude_code.application.code_sessions import CodeService
from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountPort,
    ConnectedAccountStatus,
)
from free_claude_code.application.errors import (
    ApplicationUnavailableError,
    InvalidRequestError,
)
from free_claude_code.application.model_metadata import ProviderModelRefreshResult
from free_claude_code.application.ports import StopResult
from free_claude_code.cli import vscode
from free_claude_code.config.admin.persistence import (
    PreparedAdminUpdate,
)
from free_claude_code.config.admin.state import ConfigInputValue, ValueState
from free_claude_code.config.admin.status import provider_config_status
from free_claude_code.config.loader import clear_settings_cache
from free_claude_code.config.model_refs import parse_provider_type
from free_claude_code.config.paths import messaging_state_dir_path
from free_claude_code.config.server_urls import local_admin_url, local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject
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

from .configuration import ConfigurationService
from .folder_picker import NativeFolderPicker
from .provider_manager import ProviderRuntimeManager
from .retired_chat import remove_retired_chat_history

RestartCallback = Callable[[], None]

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
        restart_callback: RestartCallback | None = None,
        connected_accounts: Mapping[str, ConnectedAccountPort] | None = None,
    ) -> None:
        self.provider_manager = provider_manager
        self._configuration = configuration
        self._code_service = code_service
        self._folder_picker = NativeFolderPicker()
        self._transcriber = transcriber
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
                await _await_owned_task(
                    asyncio.create_task(to_thread.run_sync(remove_retired_chat_history))
                )
                if self._draining:
                    raise ApplicationUnavailableError(
                        "Application runtime is shutting down."
                    )
                await self.provider_manager.warm_referenced_model_cache()
                self.provider_manager.start_model_list_refresh()
                if self._code_service is not None:
                    await self._code_service.start()
                await self._start_messaging_if_configured()
                if self._draining:
                    raise ApplicationUnavailableError(
                        "Application runtime is shutting down."
                    )
                logging.getLogger("uvicorn.error").info(
                    "Admin UI: %s (local-only)",
                    local_admin_url(self.settings),
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

    def begin_shutdown(self) -> None:
        """Finish indefinite observer responses before the server drains HTTP."""
        self._draining = True
        self._folder_picker.begin_shutdown()
        if self._code_service is not None:
            self._code_service.begin_shutdown()

    async def close(self) -> bool:
        self.begin_shutdown()
        async with self._lifecycle_lock:
            if self._closed:
                return True
            logger.info("Shutdown requested, cleaning up...")
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
        return await self._claude_vscode(None)

    async def connect_claude_vscode(self) -> JsonObject:
        return await self._claude_vscode(True)

    async def disconnect_claude_vscode(self) -> JsonObject:
        return await self._claude_vscode(False)

    async def _claude_vscode(self, connected: bool | None) -> JsonObject:
        async with self._config_lock:
            if self._draining or self._pending_fields:
                raise ApplicationUnavailableError(
                    "Wait for FCC to restart before changing the integration."
                )
            settings = self.settings
            try:
                return await _await_owned_task(
                    asyncio.create_task(
                        to_thread.run_sync(
                            vscode.configure,
                            vscode.settings_path(),
                            local_proxy_root_url(settings),
                            settings.proxy_auth_token,
                            connected,
                        )
                    )
                )
            except ValueError, UnicodeError:
                raise InvalidRequestError(
                    "Could not read VS Code settings. Check the JSON and environment entries."
                ) from None
            except OSError:
                raise ApplicationUnavailableError(
                    "Could not access VS Code settings. Check file permissions and try again."
                ) from None

    async def admin_status(self) -> JsonObject:
        values = await self.admin_values()
        settings = self.settings
        return {
            "status": "stopping" if self._draining else "running",
            "instance_id": self._instance_id,
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
        lease = await self.provider_manager.acquire()
        try:
            provider = lease.resolve_provider(provider_id)
            infos = await provider.list_model_infos()
        except Exception as exc:
            logger.warning(
                "Admin provider check failed: provider={} exc_type={}",
                provider_id,
                type(exc).__name__,
            )
            return {
                "provider_id": provider_id,
                "ok": False,
                "message": _PROVIDER_CHECK_FAILURE_MESSAGE,
            }
        finally:
            await lease.release()
        self.provider_manager.cache_model_infos(provider_id, infos)
        return {
            "provider_id": provider_id,
            "ok": True,
            "models": sorted(info.model_id for info in infos),
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
        try:
            components = messaging_platform_factory.create_messaging_components(
                self.settings.messaging_platform,
                self._messaging_options(),
            )
            if components is not None:
                await self._start_messaging_workflow(components)
        except ImportError as exc:
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
        settings = self.settings
        self._messaging_runtime = components.runtime
        workspace = (
            os.path.abspath(settings.allowed_dir)
            if settings.allowed_dir
            else os.getcwd()
        )
        os.makedirs(workspace, exist_ok=True)
        data_path = os.path.abspath(messaging_state_dir_path())
        os.makedirs(data_path, exist_ok=True)
        allowed_dirs = [workspace] if settings.allowed_dir else []

        self._cli_manager = cli_managed.ManagedClaudeSessionManager(
            workspace_path=workspace,
            proxy_root_url=local_proxy_root_url(settings),
            allowed_dirs=allowed_dirs,
            auth_token=settings.proxy_auth_token,
            log_raw_cli_diagnostics=settings.log_raw_cli_diagnostics,
            log_messaging_error_details=settings.log_messaging_error_details,
        )
        session_store = messaging_session.SessionStore(
            storage_path=os.path.join(data_path, "sessions.json"),
            managed_message_cap=settings.max_message_log_entries_per_chat,
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
