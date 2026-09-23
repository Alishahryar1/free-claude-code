"""Structured DEBUG traces for end-to-end request / CLI / provider logging.

Emitted lines are merged into JSON log rows by ``config.logging_config``.
Conversation and Claude Code prompts are logged verbatim unless values live under
sanitized credential keys (e.g. ``api_key``, ``authorization``). The default
INFO log level excludes these detailed request traces.
"""

import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import Any

from loguru import logger

from free_claude_code.core.async_iterators import try_close_async_iterator

TRACE_PAYLOAD_BINDING = "trace_payload"

_SECRET_VALUE_KEYS = frozenset(
    k.lower()
    for k in (
        "authorization",
        "x-api-key",
        "anthropic-auth-token",
        "api_key",
        "password",
        "secret",
        "token",
        "bearer_token",
        "openapi_token",
        "nvidia-api-key",
    )
)


def sanitize_trace_value(obj: Any) -> Any:
    """Recursively copy JSON-like structures redacting credential-shaped keys."""
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if str(k).lower() in _SECRET_VALUE_KEYS:
                out[str(k)] = "<redacted>"
            else:
                out[str(k)] = sanitize_trace_value(v)
        return out
    if isinstance(obj, tuple | list):
        return [sanitize_trace_value(x) for x in obj]
    return obj


def trace_event(*, stage: str, event: str, source: str, **fields: Any) -> None:
    """Emit one structured DEBUG trace row merged into JSON by the log sink."""
    payload = sanitize_trace_value(
        {
            "stage": stage,
            "event": event,
            "source": source,
            **fields,
        },
    )
    logger.bind(trace_payload=payload).debug("TRACE {}", event)


async def close_stream_input(
    iterator: object,
    *,
    owner: str,
    source: str,
    preserved_error: BaseException | None,
) -> None:
    """Close one transform input and observe cleanup failure without raising it."""
    close_error = await try_close_async_iterator(iterator)
    if close_error is None:
        return
    trace_event(
        stage="lifecycle",
        event="stream.input.close_failed",
        source=source,
        owner=owner,
        close_exc_type=type(close_error).__name__,
        preserved_exc_type=(
            type(preserved_error).__name__ if preserved_error is not None else None
        ),
    )


def extract_claude_session_id_from_headers(headers: Mapping[str, str]) -> str | None:
    """Best-effort session id forwarded by Claude Code / SDK via HTTP."""
    lowered = {str(k).lower(): v for k, v in headers.items() if isinstance(v, str)}
    for key in (
        "anthropic-session-id",
        "x-anthropic-session-id",
        "claude-session-id",
        "x-claude-session-id",
    ):
        candidate = lowered.get(key)
        if candidate:
            return candidate
    return None


async def traced_async_stream(
    agen: AsyncIterator[str],
    *,
    stage: str,
    source: str,
    complete_event: str,
    interrupted_event: str,
    chunk_event: str | None = None,
    chunk_interval: int = 250,
    extra: Mapping[str, Any] | None = None,
) -> AsyncGenerator[str]:
    """Emit TRACE rows when a text stream completes, fails, cancels, or periodically."""
    common = dict(extra or {})
    count = 0
    nbytes = 0
    interrupted = False
    try:
        async for chunk in agen:
            count += 1
            nbytes += len(chunk.encode("utf-8", errors="replace"))
            if chunk_event and chunk_interval > 0 and count % chunk_interval == 0:
                trace_event(
                    stage=stage,
                    event=chunk_event,
                    source=source,
                    stream_chunks_so_far=count,
                    stream_bytes_so_far=nbytes,
                    **common,
                )
            yield chunk
    except GeneratorExit:
        raise
    except asyncio.CancelledError:
        interrupted = True
        trace_event(
            stage=stage,
            event=interrupted_event,
            source=source,
            stream_chunks=count,
            stream_bytes=nbytes,
            outcome="cancelled",
            **common,
        )
        raise
    except BaseExceptionGroup as grp:
        interrupted = True
        trace_event(
            stage=stage,
            event=interrupted_event,
            source=source,
            stream_chunks=count,
            stream_bytes=nbytes,
            outcome="exception_group",
            note=str(grp),
            **common,
        )
        raise
    except Exception as exc:
        interrupted = True
        trace_event(
            stage=stage,
            event=interrupted_event,
            source=source,
            stream_chunks=count,
            stream_bytes=nbytes,
            outcome="error",
            exc_type=type(exc).__name__,
            **common,
        )
        raise
    finally:
        await close_stream_input(
            agen,
            owner="traced_async_stream",
            source=source,
            preserved_error=sys.exception(),
        )

    if not interrupted:
        trace_event(
            stage=stage,
            event=complete_event,
            source=source,
            stream_chunks=count,
            stream_bytes=nbytes,
            outcome="ok",
            **common,
        )


def provider_chat_body_snapshot(body: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitized OpenAI-compat chat body subset for traces (conversation text verbatim)."""
    keys = ("model", "messages", "tools", "tool_choice", "temperature", "max_tokens")
    snap = {k: body[k] for k in keys if k in body and body[k] is not None}
    return sanitize_trace_value(snap)
