"""Loguru-based structured logging configuration.

All logs are written to server.log as JSON lines for full traceability.
Stdlib logging is intercepted and funneled to loguru.
Context vars (request_id, node_id, chat_id) from contextualize() are
included at top level for easy grep/filter.
"""

import json
import logging
from pathlib import Path

from loguru import logger

_configured = False

# Context keys we promote to top-level JSON for traceability
_CONTEXT_KEYS = ("request_id", "node_id", "chat_id")


def _serialize_with_context(record) -> str:
    """Format record as JSON with context vars at top level.
    Returns a format template; we inject _json into record for output.
    """
    extra = record.get("extra", {})
    out = {
        "time": str(record["time"]),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    for key in _CONTEXT_KEYS:
        if key in extra and extra[key] is not None:
            out[key] = extra[key]
    record["_json"] = json.dumps(out, default=str)
    return "{_json}\n"


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging(log_file: str, *, force: bool = False) -> None:
    """Configure loguru with JSON output to log_file and intercept stdlib logging.

    Idempotent: skips if already configured (e.g. hot reload).
    Use force=True to reconfigure (e.g. in tests with a different log path).
    """
    global _configured
    if _configured and not force:
        return
    _configured = True

    # Remove default loguru handler (writes to stderr)
    logger.remove()

    # Truncate log file on fresh start for clean debugging
    Path(log_file).write_text("")

    # Add file sink: JSON lines, DEBUG level, context vars at top level
    logger.add(
        log_file,
        level="DEBUG",
        format=_serialize_with_context,
        encoding="utf-8",
        mode="a",
        rotation="50 MB",
    )

    # Intercept stdlib logging: route all root logger output to loguru
    intercept = InterceptHandler()
    logging.root.handlers = [intercept]
    logging.root.setLevel(logging.DEBUG)
