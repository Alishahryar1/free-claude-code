"""FastAPI application factory and configuration."""

import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.types import Receive, Scope, Send

from config.logging_config import configure_logging
from config.paths import server_log_path
from config.settings import get_settings
from core.trace import extract_claude_session_id_from_headers, trace_event
from providers.exceptions import ProviderError

from .admin_routes import router as admin_router
from .discovery_routes import router as discovery_router
from .routes import router
from .runtime import AppRuntime, startup_failure_message
from .validation_log import summarize_request_validation_body


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    runtime = AppRuntime.for_app(app, settings=get_settings())
    await runtime.startup()

    yield

    await runtime.shutdown()


class GracefulLifespanApp:
    """ASGI wrapper that reports startup failures without Starlette tracebacks."""

    def __init__(self, app: FastAPI):
        self.app = app

    def __getattr__(self, name: str) -> Any:
        return getattr(self.app, name)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "lifespan":
            await self.app(scope, receive, send)
            return
        await self._lifespan(receive, send)

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        settings = get_settings()
        runtime = AppRuntime.for_app(self.app, settings=settings)
        startup_complete = False
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    await runtime.startup()
                except Exception as exc:
                    await send(
                        {
                            "type": "lifespan.startup.failed",
                            "message": startup_failure_message(settings, exc),
                        }
                    )
                    return
                startup_complete = True
                await send({"type": "lifespan.startup.complete"})
                continue

            if message["type"] == "lifespan.shutdown":
                if startup_complete:
                    try:
                        await runtime.shutdown()
                    except Exception as exc:
                        logger.error("Shutdown failed: exc_type={}", type(exc).__name__)
                        await send({"type": "lifespan.shutdown.failed", "message": ""})
                        return
                await send({"type": "lifespan.shutdown.complete"})
                return


def create_app(*, lifespan_enabled: bool = True) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(
        server_log_path(), verbose_third_party=settings.log_raw_api_payloads
    )

    app_kwargs: dict[str, Any] = {
        "title": "Free Claude Code — Anthropic-compatible model gateway",
        "version": "2.4.0",
        "description": (
            "Drop-in Anthropic Messages API proxy that routes Claude Code CLI "
            "and compatible clients to 18 provider backends. See /llms.txt and "
            "/.well-known/agents.json for machine-readable discovery."
        ),
    }
    if lifespan_enabled:
        app_kwargs["lifespan"] = lifespan
    app = FastAPI(**app_kwargs)

    @app.middleware("http")
    async def trace_http_correlation(request: Request, call_next):
        """Attach HTTP identifiers and optional Claude session id to logs."""
        claude_sid = extract_claude_session_id_from_headers(request.headers)
        with logger.contextualize(
            http_method=request.method,
            http_path=request.url.path,
            claude_session_id=claude_sid,
        ):
            response = await call_next(request)
        return response

    # Register routes
    app.include_router(admin_router)
    app.include_router(discovery_router)
    app.include_router(router)

    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Log request shape for 422 debugging without content values."""
        body: Any
        try:
            body = await request.json()
        except Exception as e:
            body = {"_json_error": type(e).__name__}

        message_summary, tool_names = summarize_request_validation_body(body)

        trace_event(
            stage="ingress",
            event="server.request.validation_failed",
            source="api",
            path=request.url.path,
            query=dict(request.query_params),
            error_locs=[list(error.get("loc", ())) for error in exc.errors()],
            error_types=[str(error.get("type", "")) for error in exc.errors()],
            message_summary=message_summary,
            tool_names=tool_names,
        )
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError):
        """Handle provider-specific errors and return Anthropic format."""
        err_settings = get_settings()
        if err_settings.log_api_error_tracebacks:
            logger.error(
                "Provider Error: error_type={} status_code={} message={}",
                exc.error_type,
                exc.status_code,
                exc.message,
            )
        else:
            logger.error(
                "Provider Error: error_type={} status_code={}",
                exc.error_type,
                exc.status_code,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_anthropic_format(),
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Handle general errors and return Anthropic format."""
        settings = get_settings()
        if settings.log_api_error_tracebacks:
            logger.error("General Error: {}", exc)
            logger.error(traceback.format_exc())
        else:
            logger.error(
                "General Error: path={} method={} exc_type={}",
                request.url.path,
                request.method,
                type(exc).__name__,
            )
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


def create_asgi_app() -> GracefulLifespanApp:
    """Create the server ASGI app with graceful lifespan failure reporting."""
    return GracefulLifespanApp(create_app(lifespan_enabled=False))
