"""FastAPI application factory and configuration."""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .routes import router
from .dependencies import cleanup_provider
from providers.exceptions import ProviderError
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("server.log", encoding="utf-8", mode="w")],
)
logger = logging.getLogger(__name__)

# Suppress noisy uvicorn logs
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


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
        if settings.telegram_api_id and settings.telegram_api_hash:
            from messaging.telegram import TelegramPlatform
            from messaging.handler import ClaudeMessageHandler
            from messaging.session import SessionStore
            from messaging.rate_limited_platform import RateLimitedPlatform

            from cli.manager import CLISessionManager

            # Setup workspace - CLI runs in allowed_dir if set (e.g. project root)
            workspace = (
                os.path.abspath(settings.allowed_dir)
                if settings.allowed_dir
                else os.getcwd()
            )
            os.makedirs(workspace, exist_ok=True)

            # Session data (Telegram session, app sessions) stored in .agent_workspace
            data_path = os.path.abspath(settings.claude_workspace)
            os.makedirs(data_path, exist_ok=True)

            allowed_dirs = [workspace] if settings.allowed_dir else []
            cli_manager = CLISessionManager(
                workspace_path=workspace,
                api_url="http://localhost:8082/v1",
                allowed_dirs=allowed_dirs,
                max_sessions=settings.max_cli_sessions,
            )

            # Initialize session store
            session_store = SessionStore(
                storage_path=os.path.join(data_path, "sessions.json")
            )

            # Create Telegram platform
            raw_platform = TelegramPlatform(
                session_path=os.path.join(data_path, "claude_bot.session")
            )

            # Wrap with rate limiter
            messaging_platform = RateLimitedPlatform(
                raw_platform, rps=settings.telegram_rps
            )

            # Create and register message handler
            message_handler = ClaudeMessageHandler(
                platform=messaging_platform,
                cli_manager=cli_manager,
                session_store=session_store,
            )

            # Wire up the handler
            messaging_platform.on_message(message_handler.handle_message)

            # Start the platform
            await messaging_platform.start()
            logger.info("Telegram platform started with message handler")

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
    if messaging_platform:
        await messaging_platform.stop()
    if cli_manager:
        await cli_manager.stop_all()
    await cleanup_provider()
    logger.info("Server shutting down...")


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
        logger.error(f"General Error: {str(exc)}")
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
