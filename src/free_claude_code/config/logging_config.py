"""Loguru-based structured logging configuration.

Structured logs are written as JSON lines to a configurable path (default
``~/.fcc/logs/server.log``). Stdlib logging is intercepted and funneled to loguru.
Loguru's JSON stores exception metadata in ``record.exception`` and context in
``record.extra``. Structured traces live in ``record.extra.trace_payload``.
"""

import logging
import threading
from pathlib import Path

from loguru import logger

_configured = False
_current_path: Path | None = None
_current_level = "INFO"
_current_verbose: bool | None = None
_sink_id: int | None = None

_THIRD_PARTY_LOGGERS = (
    "openai",
    "httpx",
    "httpcore",
    "httpcore.http11",
    "httpcore.connection",
    "telegram",
    "telegram.ext",
)


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging to loguru."""

    def __init__(self) -> None:
        super().__init__()
        self._local = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._local, "active", False):
            # Avoid deadlock when nested stdlib records fire during a loguru emit.
            return
        self._local.active = True
        try:
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
        finally:
            self._local.active = False


def _set_third_party_levels(verbose: bool) -> None:
    level = logging.NOTSET if verbose else logging.WARNING
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(level)


def _add_file_sink(log_file: str | Path, level: str) -> int:
    log_path = Path(log_file)
    return logger.add(
        log_path,
        level=level,
        serialize=True,
        encoding="utf-8",
        mode="a",
        rotation="50 MB",
        retention=5,
        enqueue=True,
    )


def configure_logging(
    log_file: str | Path,
    *,
    force: bool = False,
    verbose_third_party: bool = False,
    level: str = "INFO",
) -> None:
    """Configure loguru with JSON output to log_file and intercept stdlib logging.

    Idempotent: skips if already configured with the same path, level, and verbosity.
    On path or level change, replaces only the file sink without truncating.
    On verbosity change alone, updates only the third-party logger levels.
    Use force=True to rebuild handlers while preserving existing log records.

    When ``verbose_third_party`` is false, managed noisy third-party loggers
    are capped at WARNING unless explicitly configured otherwise.
    """
    global _configured, _current_path, _current_level, _current_verbose, _sink_id

    log_path = Path(log_file).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        _configured
        and not force
        and log_path == _current_path
        and level == _current_level
        and verbose_third_party == _current_verbose
    ):
        return

    if not _configured or force:
        _configured = True

        logger.remove()

        _sink_id = _add_file_sink(log_path, level)

        intercept = InterceptHandler()
        logging.root.handlers = [intercept]
        logging.root.setLevel(logging.DEBUG)

        _set_third_party_levels(verbose_third_party)
    elif log_path != _current_path or level != _current_level:
        if _sink_id is not None:
            logger.remove(_sink_id)
        _sink_id = _add_file_sink(log_path, level)
        if verbose_third_party != _current_verbose:
            _set_third_party_levels(verbose_third_party)
    else:
        _set_third_party_levels(verbose_third_party)

    _current_path = log_path
    _current_level = level
    _current_verbose = verbose_third_party
