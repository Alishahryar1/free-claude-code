"""Application runtime composition and lifecycle ownership."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from loguru import logger

from api.admin_urls import local_admin_url
from api.auto_forge import start as start_auto_forge
from api.auto_forge import stop as stop_auto_forge
from config.settings import Settings, get_settings
from providers.exceptions import ServiceUnavailableError
from providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from cli.manager import CLISessionManager
    from messaging.handler import ClaudeMessageHandler
    from messaging.platforms.base import MessagingPlatform
    from messaging.session import SessionStore

_SHUTDOWN_TIMEOUT_S = 5.0


async def best_effort(
    name: str,
    awaitable: Any,
    timeout_s: float = _SHUTDOWN_TIMEOUT_S,
    *,
    log_verbose_errors: bool = False,
) -> None:
    """Run a shutdown step with timeout; never raise to callers."""
    try:
        await asyncio.wait_for(awaitable, timeout=timeout_s)
    except TimeoutError:
        logger.warning("Shutdown step timed out: {} ({}s)", name, timeout_s)
    except Exception as e:
        if log_verbose_errors:
            logger.warning(
                "Shutdown step failed: {}: {}: {}",
                name,
                type(e).__name__,
                e,
            )
        else:
            logger.warning(
                "Shutdown step failed: {}: exc_type={}",
                name,
                type(e).__name__,
            )


def warn_if_process_auth_token(settings: Settings) -> None:
    """Warn when server auth was implicitly inherited from the shell."""
    if settings.uses_process_anthropic_auth_token():
        logger.warning(
            "ANTHROPIC_AUTH_TOKEN is set in the process environment but not in "
            "a configured .env file. The proxy will require that token. Add "
            "ANTHROPIC_AUTH_TOKEN= to .env to disable proxy auth, or set the "
            "same token in .env to make server auth explicit."
        )


def log_startup_failure(settings: Settings, exc: Exception) -> None:
    """Log startup failures without traceback noise unless verbose diagnostics are enabled."""
    message = startup_failure_message(settings, exc)
    logger.error("Startup failed:\n{}", message)


def startup_failure_message(settings: Settings, exc: Exception) -> str:
    """Return a concise startup failure message for logs and ASGI lifespan failure."""
    if isinstance(exc, ServiceUnavailableError):
        return exc.message.strip() or "Server startup failed."

    if settings.log_api_error_tracebacks:
        return f"{type(exc).__name__}: {exc}"

    return f"Server startup failed: exc_type={type(exc).__name__}"


@dataclass(slots=True)
class AppRuntime:
    """Own optional messaging, CLI, session, and provider runtime resources."""

    app: FastAPI
    settings: Settings
    _provider_registry: ProviderRegistry | None = field(default=None, init=False)
    messaging_platform: MessagingPlatform | None = None
    message_handler: ClaudeMessageHandler | None = None
    cli_manager: CLISessionManager | None = None

    @classmethod
    def for_app(
        cls,
        app: FastAPI,
        settings: Settings | None = None,
    ) -> AppRuntime:
        return cls(app=app, settings=settings or get_settings())

    async def startup(self) -> None:
        logger.info("Starting Claude Code Proxy...")
        admin_url = local_admin_url(self.settings)
        self._provider_registry = ProviderRegistry()
        self.app.state.provider_registry = self._provider_registry
        try:
            warn_if_process_auth_token(self.settings)
            await self._validate_configured_models_best_effort()
            self._provider_registry.start_model_list_refresh(self.settings)
            await self._start_messaging_if_configured()
            start_auto_forge()
            self._publish_state()
            logging.getLogger("uvicorn.error").info(
                "Admin UI: %s (local-only)", admin_url
            )
        except Exception as exc:
            log_startup_failure(self.settings, exc)
            await best_effort(
                "provider_registry.cleanup",
                self._provider_registry.cleanup(),
                log_verbose_errors=self.settings.log_api_error_tracebacks,
            )
            raise

    async def _validate_configured_models_best_effort(self) -> None:
        """Warm validation status without blocking first-run/admin access."""
        if self._provider_registry is None:
            return
        try:
            await self._provider_registry.validate_configured_models(self.settings)
        except ServiceUnavailableError as exc:
            self.app.state.startup_validation_error = exc.message
            logger.warning(
                "Configured provider model validation failed during startup; "
                "server will continue and requests will fail at provider resolution "
                "when config is incomplete. {}",
                exc.message,
            )

    async def shutdown(self) -> None:
        verbose = self.settings.log_api_error_tracebacks
        if self.message_handler is not None:
            try:
                self.message_handler.session_store.flush_pending_save()
            except Exception as e:
                if verbose:
                    logger.warning("Session store flush on shutdown: {}", e)
                else:
                    logger.warning(
                        "Session store flush on shutdown: exc_type={}",
                        type(e).__name__,
                    )

        logger.info("Shutdown requested, cleaning up...")
        await best_effort(
            "auto_forge.stop",
            stop_auto_forge(),
            timeout_s=3.0,
            log_verbose_errors=verbose,
        )
        if self.messaging_platform:
            await best_effort(
                "messaging_platform.stop",
                self.messaging_platform.stop(),
                log_verbose_errors=verbose,
            )
        if self.cli_manager:
            await best_effort(
                "cli_manager.stop_all",
                self.cli_manager.stop_all(),
                log_verbose_errors=verbose,
            )
        if self._provider_registry is not None:
            await best_effort(
                "provider_registry.cleanup",
                self._provider_registry.cleanup(),
                log_verbose_errors=verbose,
            )
        await self._shutdown_limiter()
        logger.info("Server shut down cleanly")

    async def _start_messaging_if_configured(self) -> None:
        try:
            from messaging.platforms.factory import (
                MessagingPlatformOptions,
                create_messaging_platform,
            )

            self.messaging_platform = create_messaging_platform(
                self.settings.messaging_platform,
                MessagingPlatformOptions(
                    telegram_bot_token=self.settings.telegram_bot_token,
                    allowed_telegram_user_id=self.settings.allowed_telegram_user_id,
                    discord_bot_token=self.settings.discord_bot_token,
                    allowed_discord_channels=self.settings.allowed_discord_channels,
                    voice_note_enabled=self.settings.voice_note_enabled,
                    whisper_model=self.settings.whisper_model,
                    whisper_device=self.settings.whisper_device,
                    hf_token=self.settings.hf_token,
                    nvidia_nim_api_key=self.settings.nvidia_nim_api_key,
                    messaging_rate_limit=self.settings.messaging_rate_limit,
                    messaging_rate_window=self.settings.messaging_rate_window,
                    log_raw_messaging_content=self.settings.log_raw_messaging_content,
                    log_api_error_tracebacks=self.settings.log_api_error_tracebacks,
                ),
            )

            if self.messaging_platform:
                await self._start_message_handler()

        except ImportError as e:
            if self.settings.log_api_error_tracebacks:
                logger.warning("Messaging module import error: {}", e)
            else:
                logger.warning(
                    "Messaging module import error: exc_type={}",
                    type(e).__name__,
                )
        except Exception as e:
            if self.settings.log_api_error_tracebacks:
                logger.error("Failed to start messaging platform: {}", e)
                import traceback

                logger.error(traceback.format_exc())
            else:
                logger.error(
                    "Failed to start messaging platform: exc_type={}",
                    type(e).__name__,
                )

    async def _start_message_handler(self) -> None:
        from cli.manager import CLISessionManager
        from messaging.handler import ClaudeMessageHandler
        from messaging.session import SessionStore

        workspace = (
            os.path.abspath(self.settings.allowed_dir)
            if self.settings.allowed_dir
            else os.getcwd()
        )
        os.makedirs(workspace, exist_ok=True)

        data_path = os.path.abspath(self.settings.claude_workspace)
        os.makedirs(data_path, exist_ok=True)

        api_url = f"http://{self.settings.host}:{self.settings.port}/v1"
        allowed_dirs = [workspace] if self.settings.allowed_dir else []
        plans_dir_abs = os.path.abspath(
            os.path.join(self.settings.claude_workspace, "plans")
        )
        plans_directory = os.path.relpath(plans_dir_abs, workspace)
        self.cli_manager = CLISessionManager(
            workspace_path=workspace,
            api_url=api_url,
            allowed_dirs=allowed_dirs,
            plans_directory=plans_directory,
            claude_bin=self.settings.claude_cli_bin,
            auth_token=getattr(self.settings, "anthropic_auth_token", ""),
            log_raw_cli_diagnostics=self.settings.log_raw_cli_diagnostics,
            log_messaging_error_details=self.settings.log_messaging_error_details,
        )

        session_store = SessionStore(
            storage_path=os.path.join(data_path, "sessions.json"),
            message_log_cap=self.settings.max_message_log_entries_per_chat,
        )
        platform = self.messaging_platform
        assert platform is not None
        self.message_handler = ClaudeMessageHandler(
            platform=platform,
            cli_manager=self.cli_manager,
            session_store=session_store,
            debug_platform_edits=self.settings.debug_platform_edits,
            debug_subagent_stack=self.settings.debug_subagent_stack,
            log_raw_messaging_content=self.settings.log_raw_messaging_content,
            log_raw_cli_diagnostics=self.settings.log_raw_cli_diagnostics,
            log_messaging_error_details=self.settings.log_messaging_error_details,
        )
        self._restore_tree_state(session_store)

        platform.on_message(self.message_handler.handle_message)
        await platform.start()
        logger.info(f"{platform.name} platform started with message handler")

    def _restore_tree_state(self, session_store: SessionStore) -> None:
        saved_trees = session_store.get_all_trees()
        if not saved_trees:
            return
        if self.message_handler is None:
            return

        logger.info(f"Restoring {len(saved_trees)} conversation trees...")
        from messaging.trees.queue_manager import TreeQueueManager

        self.message_handler.replace_tree_queue(
            TreeQueueManager.from_dict(
                {
                    "trees": saved_trees,
                    "node_to_tree": session_store.get_node_mapping(),
                },
                queue_update_callback=self.message_handler.update_queue_positions,
                node_started_callback=self.message_handler.mark_node_processing,
            )
        )
        if self.message_handler.tree_queue.cleanup_stale_nodes() > 0:
            tree_data = self.message_handler.tree_queue.to_dict()
            session_store.sync_from_tree_data(
                tree_data["trees"], tree_data["node_to_tree"]
            )

    def _publish_state(self) -> None:
        self.app.state.messaging_platform = self.messaging_platform
        self.app.state.message_handler = self.message_handler
        self.app.state.cli_manager = self.cli_manager

    async def _shutdown_limiter(self) -> None:
        verbose = self.settings.log_api_error_tracebacks
        try:
            from messaging.limiter import MessagingRateLimiter
        except Exception as e:
            if verbose:
                logger.debug(
                    "Rate limiter shutdown skipped (import failed): {}: {}",
                    type(e).__name__,
                    e,
                )
            else:
                logger.debug(
                    "Rate limiter shutdown skipped (import failed): exc_type={}",
                    type(e).__name__,
                )
            return

        await best_effort(
            "MessagingRateLimiter.shutdown_instance",
            MessagingRateLimiter.shutdown_instance(),
            timeout_s=2.0,
            log_verbose_errors=verbose,
        )
