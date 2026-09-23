"""Pure FastAPI application factory."""

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from free_claude_code.application.code_sessions import (
    CodeConflictError,
    CodeError,
    CodeNotFoundError,
    CodeUnavailableError,
    CodeValidationError,
)
from free_claude_code.application.errors import ApplicationError
from free_claude_code.core.anthropic import anthropic_error_payload
from free_claude_code.core.diagnostics import (
    redacted_exception_traceback,
    safe_exception_message,
)
from free_claude_code.core.openai_responses import openai_error_payload
from free_claude_code.core.trace import (
    extract_claude_session_id_from_headers,
    trace_event,
)
from free_claude_code.core.version import package_version

from .admin_cache import AdminNoStoreMiddleware, attach_admin_no_store
from .admin_routes import router as admin_router
from .code_sessions_routes import router as code_router
from .ports import ApiServices
from .request_errors import ordinary_application_error_response
from .request_ids import (
    RequestCorrelationMiddleware,
    attach_request_id_headers,
    get_request_id,
)
from .request_lifetime import ClientRequestLifetimeMiddleware
from .routes import router
from .validation_log import summarize_request_validation_body


def create_app(services: ApiServices) -> FastAPI:
    """Create the HTTP adapter around explicitly supplied runtime services."""
    app = FastAPI(title="Claude Code Proxy", version=package_version())
    app.state.services = services
    app.add_middleware(AdminNoStoreMiddleware)
    app.add_middleware(ClientRequestLifetimeMiddleware)
    app.add_middleware(RequestCorrelationMiddleware)

    app.include_router(admin_router)
    app.include_router(code_router)
    app.include_router(router)

    @app.exception_handler(CodeError)
    async def code_error_handler(request: Request, exc: CodeError):
        if isinstance(exc, CodeNotFoundError):
            status_code = 404
        elif isinstance(exc, CodeConflictError):
            status_code = 409
        elif isinstance(exc, CodeValidationError):
            status_code = 400
        elif isinstance(exc, CodeUnavailableError):
            status_code = 503
        else:
            status_code = 500
        response = JSONResponse(
            status_code=status_code,
            content={"detail": str(exc), "code": type(exc).__name__},
        )
        attach_admin_no_store(response, path=request.url.path)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Log request shape for 422 debugging without content values."""
        body: object
        try:
            body = await request.json()
        except Exception as error:
            body = {"_json_error": type(error).__name__}

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

    @app.exception_handler(ApplicationError)
    async def application_error_handler(request: Request, exc: ApplicationError):
        """Serialize defensive application failures in the selected wire protocol."""
        return ordinary_application_error_response(
            exc,
            wire_api=(
                "responses" if request.url.path == "/v1/responses" else "messages"
            ),
            request_id=get_request_id(request),
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Handle general errors and return Anthropic format."""
        request_id = get_request_id(request)
        claude_sid = extract_claude_session_id_from_headers(request.headers)
        settings = services.requests.current_settings()
        with logger.contextualize(
            http_method=request.method,
            http_path=request.url.path,
            claude_session_id=claude_sid,
            request_id=request_id,
        ):
            if settings.log_api_error_tracebacks:
                logger.error("General Error: {}", safe_exception_message(exc))
                logger.error(redacted_exception_traceback(exc))
            else:
                logger.error(
                    "General Error: path={} method={} exc_type={}",
                    request.url.path,
                    request.method,
                    type(exc).__name__,
                )
            message = safe_exception_message(exc)
            if request.url.path == "/v1/responses":
                content = openai_error_payload(message=message, error_type="api_error")
            else:
                content = anthropic_error_payload(
                    error_type="api_error",
                    message=message,
                    request_id=request_id,
                )
            response = JSONResponse(status_code=500, content=content)
        attach_admin_no_store(response, path=request.url.path)
        attach_request_id_headers(
            response,
            request_id=request_id,
            path=request.url.path,
        )
        return response

    return app
