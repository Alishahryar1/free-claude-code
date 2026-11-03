"""Bounded, metadata-only policy evidence; normalized usage is not billing proof."""

import asyncio
import copy
import math
import sys
from collections import deque
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from threading import Lock
from time import perf_counter

from free_claude_code.config.routing_policy import MODEL_ID
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.trace import close_stream_input, trace_event

_IDENTIFIERS = {
    "request_id",
    "requested_model",
    "selected_model",
    "provider_id",
    "proposed_model",
    "mode",
    "reason",
    "wire_api",
    "outcome",
    "error_type",
}
_NUMBERS = {
    "attempt_count",
    "first_event_ms",
    "first_output_ms",
    "completion_ms",
    "estimated_cost",
}
_USAGE = {
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
}


class DecisionLog:
    def __init__(self, limit: int = 100):
        self._rows: deque[JsonObject] = deque(maxlen=limit)
        self._lock = Lock()

    def record(self, **fields: object) -> None:
        row: JsonObject = {
            "timestamp": datetime.now(UTC).isoformat(),
            "policy_version": 1,
        }
        for key in _IDENTIFIERS:
            value = fields.get(key)
            row[key] = (
                value if isinstance(value, str) and MODEL_ID.fullmatch(value) else None
            )
        for key in _NUMBERS:
            value = fields.get(key)
            row[key] = (
                value
                if (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                )
                else None
            )
        row["usage_kind"] = "normalized_stream_unverified"
        usage = fields.get("reported_usage")
        numeric_usage: JsonObject = {}
        if isinstance(usage, dict):
            for key, value in usage.items():
                if (
                    key in _USAGE
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                ):
                    numeric_usage[key] = value
        row["reported_usage"] = numeric_usage
        with self._lock:
            self._rows.append(copy.deepcopy(row))
        trace_event(
            stage="routing",
            event="routing_policy.decision",
            source="application",
            **row,
        )

    def recent(self) -> list[JsonObject]:
        with self._lock:
            return copy.deepcopy(list(reversed(self._rows)))


decision_log = DecisionLog()


async def observe_stream(
    stream: AsyncIterator[str], decision: dict[str, object], log: DecisionLog
) -> AsyncIterator[str]:
    started = perf_counter()
    outcome = "ok"
    error_type = None
    buffer = ""
    usage: dict[str, int] = {}
    decision["first_event_ms"] = None
    decision["first_output_ms"] = None
    try:
        async for chunk in stream:
            elapsed = (perf_counter() - started) * 1000
            if decision["first_event_ms"] is None:
                decision["first_event_ms"] = elapsed
            buffer = (buffer + chunk).replace("\r\n", "\n")
            if len(buffer) > 65536:
                buffer = ""
            while "\n\n" in buffer:
                raw, buffer = buffer.split("\n\n", 1)
                for event in parse_sse_text(raw):
                    data = event.data
                    if event.event in {
                        "error",
                        "response.failed",
                        "response.incomplete",
                    }:
                        outcome = "upstream_error"
                    delta = data.get("delta")
                    if (
                        event.event
                        in {
                            "response.output_text.delta",
                            "response.reasoning_text.delta",
                        }
                        or (
                            isinstance(delta, dict)
                            and delta.get("type") in {"text_delta", "thinking_delta"}
                        )
                    ) and decision["first_output_ms"] is None:
                        decision["first_output_ms"] = elapsed
                    container = data.get("message") or data.get("response")
                    values = (
                        container.get("usage")
                        if isinstance(container, dict)
                        else data.get("usage")
                    )
                    if isinstance(values, dict):
                        usage.update(
                            {
                                key: value
                                for key, value in values.items()
                                if key in _USAGE
                                and isinstance(value, int)
                                and not isinstance(value, bool)
                                and value >= 0
                            }
                        )
            yield chunk
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except GeneratorExit:
        outcome = "closed"
        raise
    except Exception as error:
        outcome = "error"
        error_type = type(error).__name__
        raise
    finally:
        try:
            await close_stream_input(
                stream,
                owner="routing_decisions",
                source="application",
                preserved_error=sys.exception(),
            )
        finally:
            log.record(
                **decision,
                outcome=outcome,
                error_type=error_type,
                completion_ms=(perf_counter() - started) * 1000,
                reported_usage=usage,
            )
