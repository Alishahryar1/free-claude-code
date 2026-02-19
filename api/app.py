"""FastAPI application factory and configuration."""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from config.logging_config import configure_logging
from config.settings import get_settings
from providers.exceptions import ProviderError

from .dependencies import cleanup_provider
from .routes import router

# Opt-in to future behavior for python-telegram-bot
os.environ["PTB_TIMEDELTA"] = "1"

# Configure logging first (before any module logs)
_settings = get_settings()
configure_logging(_settings.log_file)


_SHUTDOWN_TIMEOUT_S = 5.0


async def _best_effort(
    name: str, awaitable, timeout_s: float = _SHUTDOWN_TIMEOUT_S
) -> None:
    """Run a shutdown step with timeout; never raise to callers."""
    try:
        await asyncio.wait_for(awaitable, timeout=timeout_s)
    except TimeoutError:
        logger.warning(f"Shutdown step timed out: {name} ({timeout_s}s)")
    except Exception as e:
        logger.warning(f"Shutdown step failed: {name}: {type(e).__name__}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    logger.info("Starting Claude Code Proxy...")

    # Initialize messaging platform if configured
    messaging_platform = None
    message_handler = None
    cli_manager = None

    try:
        # Use the messaging factory to create the right platform
        from messaging.factory import create_messaging_platform

        messaging_platform = create_messaging_platform(
            platform_type=settings.messaging_platform,
            bot_token=settings.telegram_bot_token,
            allowed_user_id=settings.allowed_telegram_user_id,
            discord_bot_token=settings.discord_bot_token,
            allowed_discord_channels=settings.allowed_discord_channels,
        )

        if messaging_platform:
            from cli.manager import CLISessionManager
            from messaging.handler import ClaudeMessageHandler
            from messaging.session import SessionStore

            # Setup workspace - CLI runs in allowed_dir if set (e.g. project root)
            workspace = (
                os.path.abspath(settings.allowed_dir)
                if settings.allowed_dir
                else os.getcwd()
            )
            os.makedirs(workspace, exist_ok=True)

            # Session data stored in agent_workspace
            data_path = os.path.abspath(settings.claude_workspace)
            os.makedirs(data_path, exist_ok=True)

            api_url = f"http://{settings.host}:{settings.port}/v1"
            allowed_dirs = [workspace] if settings.allowed_dir else []
            plans_dir_abs = os.path.abspath(
                os.path.join(settings.claude_workspace, "plans")
            )
            plans_directory = os.path.relpath(plans_dir_abs, workspace)
            cli_manager = CLISessionManager(
                workspace_path=workspace,
                api_url=api_url,
                allowed_dirs=allowed_dirs,
                plans_directory=plans_directory,
            )

            # Initialize session store
            session_store = SessionStore(
                storage_path=os.path.join(data_path, "sessions.json")
            )

            # Create and register message handler
            message_handler = ClaudeMessageHandler(
                platform=messaging_platform,
                cli_manager=cli_manager,
                session_store=session_store,
            )

            # Restore tree state if available
            saved_trees = session_store.get_all_trees()
            if saved_trees:
                logger.info(f"Restoring {len(saved_trees)} conversation trees...")
                from messaging.tree_queue import TreeQueueManager

                message_handler.tree_queue = TreeQueueManager.from_dict(
                    {
                        "trees": saved_trees,
                        "node_to_tree": session_store.get_node_mapping(),
                    },
                    queue_update_callback=message_handler._update_queue_positions,
                    node_started_callback=message_handler._mark_node_processing,
                )
                # Reconcile restored state - anything PENDING/IN_PROGRESS is lost across restart
                if message_handler.tree_queue.cleanup_stale_nodes() > 0:
                    # Sync back and save
                    tree_data = message_handler.tree_queue.to_dict()
                    session_store.sync_from_tree_data(
                        tree_data["trees"], tree_data["node_to_tree"]
                    )

            # Wire up the handler
            messaging_platform.on_message(message_handler.handle_message)

            # Start the platform
            await messaging_platform.start()
            logger.info(
                f"{messaging_platform.name} platform started with message handler"
            )

    except ImportError as e:
        logger.warning(f"Messaging module import error: {e}")
    except Exception as e:
        logger.error(f"Failed to start messaging platform: {e}")
        import traceback

        logger.error(traceback.format_exc())

    # Store in app state for access in routes
    app.state.messaging_platform = messaging_platform
    app.state.message_handler = message_handler
    app.state.cli_manager = cli_manager

    yield

    # Cleanup
    if message_handler and hasattr(message_handler, "session_store"):
        try:
            message_handler.session_store.flush_pending_save()
        except Exception as e:
            logger.warning(f"Session store flush on shutdown: {e}")
    logger.info("Shutdown requested, cleaning up...")
    if messaging_platform:
        await _best_effort("messaging_platform.stop", messaging_platform.stop())
    if cli_manager:
        await _best_effort("cli_manager.stop_all", cli_manager.stop_all())
    await _best_effort("cleanup_provider", cleanup_provider())

    # Ensure background limiter worker doesn't keep the loop alive.
    try:
        from messaging.limiter import MessagingRateLimiter

        await _best_effort(
            "MessagingRateLimiter.shutdown_instance",
            MessagingRateLimiter.shutdown_instance(),
            timeout_s=2.0,
        )
    except Exception:
        # Limiter may never have been imported/initialized.
        pass

    logger.info("Server shut down cleanly")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Claude Code Proxy",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Register routes
    app.include_router(router)

    # Exception handlers
    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError):
        """Handle provider-specific errors and return Anthropic format."""
        logger.error(f"Provider Error: {exc.error_type} - {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_anthropic_format(),
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Handle general errors and return Anthropic format."""
        logger.error(f"General Error: {exc!s}")
        import traceback

        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "An unexpected error occurred.",
                },
            },
        )

    return app


# Default app instance for uvicorn
app = create_app()
