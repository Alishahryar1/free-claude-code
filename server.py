"""
Claude Code Proxy - ASGI Entry Point

Allows running the proxy via:
    uv run uvicorn server:app --host 0.0.0.0 --port 8082
or directly via:
    uv run python server.py
"""

from typing import TYPE_CHECKING

from free_claude_code.cli.process_registry import kill_all_best_effort
from free_claude_code.config.loader import get_settings
from free_claude_code.runtime.bootstrap import build_asgi_app

if TYPE_CHECKING:
    from free_claude_code.runtime.asgi import RuntimeASGIApp


def create_app() -> RuntimeASGIApp:
    """Build and return the configured RuntimeASGIApp."""
    settings = get_settings()
    return build_asgi_app(settings)


app = create_app()

__all__ = ["app", "create_app"]

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    try:
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level="debug",
            timeout_graceful_shutdown=5,
        )
    finally:
        kill_all_best_effort()
