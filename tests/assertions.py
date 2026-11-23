"""Assertions shared by tests that expect optional results to exist."""

import logging
from types import TracebackType


def require_present[T](value: T | None) -> T:
    assert value is not None, "Expected a value"
    return value


def record_extra(record: logging.LogRecord) -> dict[str, object]:
    extra = getattr(record, "extra", None)
    assert isinstance(extra, dict), f"Expected Loguru metadata on {record.name}"
    return extra


def require_exception_info(
    info: tuple[type[BaseException], BaseException, TracebackType | None]
    | tuple[None, None, None]
    | bool
    | None,
) -> tuple[type[BaseException], BaseException, TracebackType | None]:
    assert isinstance(info, tuple)
    kind, error, traceback = info
    assert kind is not None and error is not None
    return kind, error, traceback
