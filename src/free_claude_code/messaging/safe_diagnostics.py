"""Helpers for redacting user-derived content from log lines."""


def format_exception_for_log(exc: BaseException, *, log_full_message: bool) -> str:
    """Return exception type and optionally ``str(exc)`` for operator diagnostics."""
    if log_full_message:
        return f"{type(exc).__name__}: {exc}"
    return type(exc).__name__


def text_len_hint(text: str | None) -> int:
    """Length of text for metadata-only logging (0 when missing)."""
    if not text:
        return 0
    return len(text)
