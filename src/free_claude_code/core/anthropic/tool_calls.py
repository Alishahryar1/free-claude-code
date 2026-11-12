"""Keep client calls private until the Messages stop reason permits execution."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ..failures import ExecutionFailure, FailureKind
from ..stream_recovery import SourceRecoverySnapshot, StreamFailureContext
from ..tool_call_buffer import OrderedToolCallBuffer

type SourceEvent = tuple[str, dict[str, Any]]
type SourceEmission = tuple[SourceEvent, ...]


@dataclass(slots=True)
class _PendingTool:
    events: list[SourceEvent] = field(default_factory=list)
    stopped: bool = False


class MessagesToolCallBuffer:
    def __init__(self) -> None:
        self._buffer = OrderedToolCallBuffer[SourceEmission | _PendingTool]()
        self._tools: dict[int, _PendingTool] = {}
        self._seen: set[int] = set()
        self._call_ids: set[str] = set()
        self._stop_reason: object = None

    def feed(self, kind: str, payload: dict[str, Any]) -> list[SourceEmission]:
        data = deepcopy(payload)
        data["type"] = kind
        size = len(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        index = data.get("index")
        block = data.get("content_block")
        if (
            kind == "content_block_start"
            and isinstance(block, dict)
            and block.get("type") == "tool_use"
        ):
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index in self._seen
                or self._stop_reason is not None
            ):
                raise _failure("Invalid Messages tool block index or lifecycle")
            self._buffer.budget.add(32)
            self._seen.add(index)
            pending = _PendingTool()
            self._buffer.reserve(index, pending)
            self._tools[index] = pending
        if index in self._tools:
            pending = self._tools[index]
            if pending.stopped:
                raise _failure("Messages tool changed after block stop")
            self._buffer.grow(index, size)
            pending.events.append((kind, data))
            pending.stopped = kind == "content_block_stop"
        else:
            if index in self._seen:
                raise _failure("Messages tool changed after completion")
            delta = data.get("delta")
            if (
                kind == "message_delta"
                and isinstance(delta, dict)
                and delta.get("stop_reason") is not None
            ):
                self._stop_reason = delta["stop_reason"]
                successful = delta["stop_reason"] in {
                    "tool_use",
                    "end_turn",
                    "stop_sequence",
                }
                for key, pending in self._tools.items():
                    if successful and pending.stopped:
                        block = pending.events[0][1]["content_block"]
                        identity = validate_tool_identity(block, self._call_ids)
                        self._buffer.budget.add(len(identity.encode("utf-8")), 0)
                        self._call_ids.add(identity)
                        self._buffer.complete(key)
                    else:
                        self._buffer.discard(key)
                self._tools.clear()
            elif kind in {"message_stop", "error"}:
                if kind == "message_stop" and self._stop_reason is None:
                    raise ExecutionFailure(
                        FailureKind.UPSTREAM,
                        502,
                        "Messages stream ended without a stop reason",
                        False,
                    )
                self._buffer.discard_pending()
                self._tools.clear()
            self._buffer.append(((kind, data),), size=size)
        return self._drain()

    def abandon(self) -> list[SourceEmission]:
        self._buffer.discard_pending()
        self._tools.clear()
        return self._drain()

    def _drain(self) -> list[SourceEmission]:
        return [
            tuple(value.events) if isinstance(value, _PendingTool) else value
            for value in self._buffer.drain()
        ]


def validate_tool_identity(block: dict[str, Any], seen: set[str]) -> str:
    """Validate a complete tool's execution identity for SSE and native JSON."""
    identity = block.get("id")
    name = block.get("name")
    if (
        not isinstance(identity, str)
        or not identity.strip()
        or not isinstance(name, str)
        or not name.strip()
    ):
        raise _failure("Completed Messages tool requires an ID and name")
    if identity in seen:
        raise _failure("Messages tools share an execution ID")
    return identity


def _failure(message: str) -> ExecutionFailure:
    return ExecutionFailure(
        FailureKind.UPSTREAM,
        502,
        message,
        False,
        StreamFailureContext(
            SourceRecoverySnapshot(unsafe_reason="tool_protocol_integrity")
        ),
    )
