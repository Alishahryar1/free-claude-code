"""Retain client tool items until Responses supplies authoritative completion."""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ..failures import ExecutionFailure, FailureKind
from ..stream_recovery import SourceRecoverySnapshot, StreamFailureContext
from ..tool_call_buffer import OrderedToolCallBuffer, RetentionBudget
from .tool_search import is_client_search

type SourceEvent = tuple[str, dict[str, Any]]
type SourceEmission = tuple[SourceEvent, ...]


def is_client_call(item: dict[str, Any]) -> bool:
    kind = item.get("type")
    if kind == "tool_search_call":
        return is_client_search(item)
    return kind in {
        "function_call",
        "custom_tool_call",
        "computer_call",
        "local_shell_call",
        "apply_patch_call",
    } or (
        kind == "shell_call"
        and isinstance(item.get("environment"), dict)
        and item["environment"].get("type") == "local"
    )


@dataclass(slots=True)
class _PendingItem:
    index: int
    item: dict[str, Any]
    events: list[SourceEvent] = field(default_factory=list)
    completed: SourceEmission = ()


class ResponsesToolCallBuffer:
    def __init__(self) -> None:
        self._budget = RetentionBudget()
        self._buffer = OrderedToolCallBuffer[SourceEmission | _PendingItem](
            self._budget
        )
        self._items: dict[str, _PendingItem] = {}
        self._indices: dict[int, str] = {}
        self._fingerprints: dict[str, bytes] = {}
        self._discarded: set[str] = set()
        self._call_ids: dict[str, str] = {}

    def feed(self, kind: str, payload: dict[str, Any]) -> list[SourceEmission]:
        data = deepcopy(payload)
        data["type"] = kind
        if kind in {"error", "response.error"}:
            self._discard_pending()
            self._buffer.append(((kind, data),), size=_size(data))
            return self._drain()
        item = data.get("item")
        item = item if isinstance(item, dict) else {}
        index = data.get("output_index")
        item_id = item.get("id") or data.get("item_id") or self._indices.get(index)
        if (
            isinstance(item_id, str)
            and (item_id in self._items or item_id in self._fingerprints)
            and "output_index" in data
            and (
                not isinstance(index, int)
                or isinstance(index, bool)
                or self._indices.get(index) != item_id
            )
        ):
            raise _failure("Responses tool item changed its output index")
        if is_client_call(item):
            if not isinstance(item_id, str) or not item_id:
                raise _failure("Responses tool item requires an ID")
            self._ensure(item_id, index, item)
        if isinstance(item_id, str) and item_id in self._items:
            pending = self._items[item_id]
            if pending.completed:
                if kind == "response.output_item.done":
                    self._verify(item_id, item)
                    return self._drain()
                raise _failure("Responses tool changed after completion")
            if kind == "response.output_item.done":
                self._finish_item(item_id, item, data)
            else:
                self._buffer.grow(item_id, _size(data))
                pending.events.append((kind, data))
            return self._drain()
        if (
            isinstance(item_id, str)
            and item_id in self._fingerprints
            and is_client_call(item)
        ):
            if kind == "response.output_item.done":
                self._verify(item_id, item)
                return self._drain()
            raise _failure("Responses tool changed after completion")
        if kind.startswith(
            ("response.function_call_arguments.", "response.custom_tool_call_input.")
        ):
            raise _failure("Responses arguments have no unfinished tool item")
        if isinstance(response := data.get("response"), dict):
            terminal = kind in {
                "response.completed",
                "response.incomplete",
                "response.failed",
            }
            output = response.get("output")
            if isinstance(output, list):
                for position, value in enumerate(output):
                    if not isinstance(value, dict) or not is_client_call(value):
                        continue
                    identity = value.get("id")
                    if not isinstance(identity, str) or not identity:
                        raise _failure("Responses snapshot tool requires an ID")
                    eligible = terminal and (
                        value.get("status") == "completed"
                        or (
                            kind == "response.completed"
                            and value.get("status")
                            not in {"in_progress", "incomplete", "failed"}
                        )
                    )
                    if identity in self._fingerprints:
                        if eligible:
                            self._verify(identity, value)
                    elif eligible:
                        self._ensure(identity, position, value)
                        self._finish_item(identity, value)
                response["output"] = [
                    value
                    for value in output
                    if not isinstance(value, dict)
                    or not is_client_call(value)
                    or value.get("id") in self._fingerprints
                ]
            if terminal:
                self._discard_pending()
        self._buffer.append(((kind, data),), size=_size(data))
        return self._drain()

    def _ensure(self, identity: str, index: object, item: dict[str, Any]) -> None:
        if identity in self._fingerprints or identity in self._items:
            return
        if identity in self._discarded:
            raise _failure("Responses reused a discarded tool identity")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise _failure("Responses tool item requires an output index")
        if index in self._indices and self._indices[index] != identity:
            raise _failure("Responses output index belongs to multiple items")
        self._budget.add(len(identity.encode("utf-8")) + 64)
        self._indices[index] = identity
        pending = _PendingItem(index, deepcopy(item))
        self._items[identity] = pending
        self._buffer.reserve(identity, pending, size=_size(item))

    def _finish_item(
        self,
        identity: str,
        item: dict[str, Any],
        done_event: dict[str, Any] | None = None,
    ) -> None:
        pending = self._items[identity]
        if item.get("status") in {"in_progress", "incomplete", "failed"}:
            self._buffer.discard(identity)
            self._discarded.add(identity)
            del self._items[identity]
            return
        call_id = item.get("call_id")
        name = item.get("name")
        if (
            not is_client_call(item)
            or not isinstance(call_id, str)
            or not call_id.strip()
        ):
            raise _failure("Completed Responses client call requires a call ID")
        if item.get("type") in {"function_call", "custom_tool_call"} and (
            not isinstance(name, str) or not name.strip()
        ):
            raise _failure("Completed Responses function requires a name")
        for key in ("type", "call_id", "name", "namespace", "execution", "environment"):
            initial = pending.item.get(key)
            if initial not in (None, "") and initial != item.get(key):
                raise _failure("Responses completion changed the tool identity")
        if call_id in self._call_ids and self._call_ids[call_id] != identity:
            raise _failure("Responses items share an execution call ID")
        self._budget.add(len(call_id.encode("utf-8")), 0)
        self._call_ids[call_id] = identity
        self._buffer.grow(identity, _size(item))
        final = deepcopy(item)
        self._fingerprints[identity] = _fingerprint(final)
        index = pending.index
        field = "input" if final.get("type") == "custom_tool_call" else "arguments"
        initial = {**final, "status": "in_progress"}
        if field in initial:
            initial[field] = ""
        prefix = (
            "response.custom_tool_call_input"
            if field == "input"
            else "response.function_call_arguments"
        )
        common = {"item_id": identity, "output_index": index}
        events = deepcopy(pending.events)
        starts = [data for kind, data in events if kind == "response.output_item.added"]
        if not starts:
            events.insert(
                0,
                (
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": index,
                        "item": initial,
                    },
                ),
            )
        else:
            for start in starts:
                # Name and identity are finalized once. Keep native extensions and
                # argument-fragment association on their original events.
                for key in ("name", "call_id"):
                    if key in final:
                        start["item"][key] = final[key]
        if isinstance(final.get(field), str):
            existing = "".join(
                str(data.get("item", {}).get(field, ""))
                for kind, data in events
                if kind == "response.output_item.added"
            )
            existing += "".join(
                str(data.get("delta", ""))
                for kind, data in events
                if kind == prefix + ".delta"
            )
            if not final[field].startswith(existing):
                raise _failure(
                    "Responses arguments changed between fragments and completion"
                )
            suffix = final[field][len(existing) :]
            insertion = next(
                (
                    position
                    for position, (kind, _) in enumerate(events)
                    if kind == prefix + ".done"
                ),
                len(events),
            )
            if suffix:
                events.insert(
                    insertion,
                    (
                        prefix + ".delta",
                        {"type": prefix + ".delta", **common, "delta": suffix},
                    ),
                )
            if not any(kind == prefix + ".done" for kind, _ in events):
                events.append(
                    (
                        prefix + ".done",
                        {"type": prefix + ".done", **common, field: final[field]},
                    )
                )
            else:
                for kind, data in events:
                    if kind == prefix + ".done":
                        data[field] = final[field]
        events.append(
            (
                "response.output_item.done",
                {
                    **(done_event or {}),
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": final,
                },
            )
        )
        pending.completed = tuple(events)
        self._buffer.complete(identity)

    def _verify(self, identity: str, item: dict[str, Any]) -> None:
        if self._fingerprints[identity] != _fingerprint(item):
            raise _failure("Responses snapshot changed an already completed tool call")

    def _discard_pending(self) -> None:
        for identity, pending in tuple(self._items.items()):
            if not pending.completed:
                self._buffer.discard(identity)
                self._discarded.add(identity)
                del self._items[identity]

    def _drain(self) -> list[SourceEmission]:
        result = []
        for value in self._buffer.drain():
            if isinstance(value, _PendingItem):
                result.append(value.completed)
                del self._items[value.item["id"]]
            else:
                result.append(value)
        return result

    def abandon(self) -> list[SourceEmission]:
        self._discard_pending()
        return self._drain()


def _fingerprint(item: dict[str, Any]) -> bytes:
    content = {
        key: value
        for key, value in item.items()
        if key not in {"status", "completed_at", "created_at", "usage"}
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).digest()


def _size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


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
