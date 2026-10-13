"""Own messaging startup, delivery resources, and their ordered cleanup gates."""

import asyncio
import importlib
import os
import traceback
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.ports import StopResult
from free_claude_code.config.paths import messaging_state_dir_path
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.core.async_tasks import run_sync_owned
from free_claude_code.core.json_types import JsonObject
from free_claude_code.messaging.platforms import factory as messaging_platform_factory
from free_claude_code.messaging.platforms.factory import MessagingPlatformOptions
from free_claude_code.messaging.platforms.ports import (
    MessagingPlatformComponents,
    MessagingRuntime,
)
from free_claude_code.messaging.voice import Transcriber

from .lifecycle import best_effort
from .sqlite_database import SQLiteDatabase

if TYPE_CHECKING:
    import free_claude_code.cli.managed as cli_managed
    import free_claude_code.messaging.workflow as messaging_workflow_module

    from .messaging_sqlite import SQLiteMessagingStore


class MessagingService:
    """Own one messaging graph, borrowing the database and HTTP readiness signal."""

    def __init__(
        self,
        *,
        settings: Callable[[], Settings],
        database: SQLiteDatabase | None,
        http_ready: asyncio.Event,
        is_draining: Callable[[], bool],
        transcriber: Transcriber | None,
        transcriber_factory: Callable[[Settings], Awaitable[Transcriber | None]] | None,
    ) -> None:
        self._settings = settings
        self._database = database
        self._http_ready = http_ready
        self._is_draining = is_draining
        self._transcriber = transcriber
        self._transcriber_factory = transcriber_factory
        self._messaging_store: SQLiteMessagingStore | None = None
        self._messaging_import_task: asyncio.Task[None] | None = None
        self._messaging_warning: str | None = None
        self._messaging_storage_error: Exception | None = None
        self._messaging_runtime: MessagingRuntime | None = None
        self._messaging_workflow: messaging_workflow_module.MessagingWorkflow | None = (
            None
        )
        self._cli_manager: cli_managed.ManagedClaudeSessionManager | None = None
        self._messaging_state = (
            "disabled" if self.settings.messaging_platform == "none" else "starting"
        )
        self._messaging_error: str | None = None
        self._startup_tasks: list[asyncio.Task[None]] = []

    @property
    def settings(self) -> Settings:
        return self._settings()

    def snapshot(self) -> JsonObject:
        return {
            "state": self._messaging_state,
            "message": self._messaging_error,
            "warning": self._messaging_warning,
        }

    def start(self) -> None:
        if self._is_draining():
            raise ApplicationUnavailableError("Application runtime is shutting down.")
        if self._database is not None:
            self._messaging_state = "starting"
            self._messaging_import_task = asyncio.create_task(
                self._initialize_messaging_storage(), name="fcc-messaging-import"
            )
            self._startup_tasks.append(self._messaging_import_task)
        self._startup_tasks.append(
            asyncio.create_task(
                self._start_messaging_if_configured(), name="fcc-messaging-startup"
            )
        )

    def cancel_background_tasks(self) -> None:
        for task in self._startup_tasks:
            if not task.done():
                task.cancel()

    async def wait_background_tasks(self) -> None:
        results = await asyncio.gather(
            *tuple(self._startup_tasks), return_exceptions=True
        )
        for result in results:
            if isinstance(result, Exception):
                logger.warning(
                    "Background initialization ended with exc_type={}",
                    type(result).__name__,
                )
        self._startup_tasks.clear()

    async def stop_all(self) -> StopResult | None:
        if self._messaging_workflow is not None:
            outcome = await self._messaging_workflow.stop_all_tasks()
            return StopResult(cancelled_count=outcome.cancelled_count)
        if self._cli_manager is not None:
            await self._cli_manager.stop_all()
            return StopResult(source="cli_manager")
        return None

    async def _start_messaging_if_configured(self) -> None:
        if self._messaging_import_task is not None:
            await asyncio.shield(self._messaging_import_task)
        if self.settings.messaging_platform == "none":
            self._messaging_state = "disabled"
            return
        try:

            def load_modules() -> None:
                for name in ("cli.managed", "messaging.workflow"):
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
            cleaned = await self.close_delivery()
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
            cleaned = await self.close_delivery()
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

    async def _initialize_messaging_storage(self) -> None:
        if self._database is None:
            return
        try:
            await self._database.start()
            from .messaging_import import import_legacy
            from .messaging_sqlite import SQLiteMessagingStore

            self._messaging_store = SQLiteMessagingStore(
                self._database,
                managed_message_cap=self.settings.max_message_log_entries_per_chat,
            )
            self._messaging_warning = await import_legacy(
                self._database, Path(messaging_state_dir_path()) / "sessions.json"
            )
            await self._messaging_store.trim()
        except Exception as exc:
            self._messaging_storage_error = exc
            logger.error(
                "Messaging storage initialization failed: {}", type(exc).__name__
            )
            return
        if self._messaging_warning:
            logger.warning("{}", self._messaging_warning)

    async def _start_messaging_workflow(
        self,
        components: MessagingPlatformComponents,
    ) -> None:
        import free_claude_code.cli.managed as cli_managed
        import free_claude_code.messaging.workflow as messaging_workflow_module

        settings = self.settings
        self._messaging_runtime = components.runtime
        workspace = (
            os.path.abspath(settings.allowed_dir)
            if settings.allowed_dir
            else os.getcwd()
        )
        await run_sync_owned(partial(os.makedirs, workspace, exist_ok=True))
        allowed_dirs = [workspace] if settings.allowed_dir else []

        self._cli_manager = cli_managed.ManagedClaudeSessionManager(
            workspace_path=workspace,
            proxy_root_url=local_proxy_root_url(settings),
            allowed_dirs=allowed_dirs,
            auth_token=settings.proxy_auth_token,
            log_raw_cli_diagnostics=settings.log_raw_cli_diagnostics,
            log_messaging_error_details=settings.log_messaging_error_details,
        )
        if self._messaging_import_task is not None:
            await asyncio.shield(self._messaging_import_task)
        if self._messaging_storage_error is not None:
            raise ApplicationUnavailableError(
                "Messaging storage is unavailable."
            ) from self._messaging_storage_error
        if self._messaging_store is None:
            raise ApplicationUnavailableError("Messaging storage is unavailable.")
        session_store = self._messaging_store
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
        await workflow.restore()
        components.runtime.on_message(workflow.handle_message)
        await self._http_ready.wait()
        if self._is_draining():
            return
        await components.runtime.start()
        await workflow.repair_restored_statuses()
        if components.startup_notice is not None:
            await workflow.publish_startup_notice(components.startup_notice)
        logger.info("{} platform started with messaging workflow", components.name)

    async def close_delivery(self) -> bool:
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

    async def close_transcriber(self) -> bool:
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
