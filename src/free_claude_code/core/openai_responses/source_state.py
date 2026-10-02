"""Observe Responses item completion and reconcile snapshots before projection."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.recovery import AttemptFailure
from free_claude_code.core.stream_events import ItemCompletion, StreamEvent
from free_claude_code.core.tool_input import complete_json_object

from .tool_search import is_client_search

_TERMINALS = {"response.completed", "response.incomplete", "response.failed"}
_INPUT_DELTAS = {
    "response.function_call_arguments.delta": "arguments",
    "response.custom_tool_call_input.delta": "input",
}
_INPUT_DONE = {
    "response.function_call_arguments.done": "arguments",
    "response.custom_tool_call_input.done": "input",
}
_TEXT_DELTAS = {
    "response.output_text.delta": ("content_index", "text"),
    "response.refusal.delta": ("content_index", "refusal"),
    "response.reasoning_summary_text.delta": ("summary_index", "text"),
    "response.reasoning_text.delta": ("content_index", "text"),
}


@dataclass(slots=True)
class _Item:
    body: dict[str, Any]
    arguments: list[str] = field(default_factory=list)
    text: dict[tuple[str, int, str], list[str]] = field(default_factory=dict)
    completion: ItemCompletion | None = None

    @property
    def client_call(self) -> bool:
        return self.body.get("type") in {"function_call", "custom_tool_call"} or (
            self.body.get("type") == "tool_search_call" and is_client_search(self.body)
        )


class ResponsesSourceState:
    """One source lifecycle, shared by the native and Messages presenters."""

    def __init__(self) -> None:
        self._items: dict[int, _Item] = {}
        self._indexes: dict[str, int] = {}

    @property
    def native_reasoning_pending(self) -> bool:
        return any(
            item.body.get("type") == "reasoning"
            and item.completion is not ItemCompletion.COMPLETE
            for item in self._items.values()
        )

    @property
    def invalid_input(self) -> bool:
        return any(
            item.client_call and item.completion is not ItemCompletion.COMPLETE
            for item in self._items.values()
        )

    def feed(self, event: StreamEvent) -> list[StreamEvent]:
        kind, payload = event.kind, event.payload
        if kind in _TERMINALS:
            output: list[StreamEvent] = []
            response = payload.get("response", {})
            for position, body in enumerate(response.get("output", [])):
                if not isinstance(body, dict):
                    continue
                index = self._indexes.get(body.get("id"), position)
                output.extend(self._snapshot(index, body, terminal=kind))
            return [*output, event]
        if kind == "response.output_item.added":
            body = payload["item"]
            index = self._item_index(payload, body)
            if index in self._items:
                self._conflict("Duplicate Responses output item.")
            self._items[index] = _Item(deepcopy(body))
            if isinstance(body.get("id"), str):
                self._indexes[body["id"]] = index
            value = body.get(
                "arguments" if body.get("type") == "function_call" else "input"
            )
            if isinstance(value, str) and value:
                self._items[index].arguments.append(value)
            initial = deepcopy(body)
            if body.get("type") == "message":
                initial["content"] = []
            elif body.get("type") == "reasoning":
                initial["summary"] = []
                if isinstance(initial.get("content"), list):
                    initial["content"] = []
            return [
                StreamEvent(kind, {**payload, "output_index": index, "item": initial}),
                *self._text_suffixes(index, self._items[index], body),
            ]
        if kind == "response.output_item.done":
            index = self._item_index(payload, payload["item"])
            return self._snapshot(
                index,
                payload["item"],
                event=StreamEvent(kind, {**payload, "output_index": index}),
            )
        index = payload.get("output_index")
        if not isinstance(index, int):
            return [event]
        item = self._items.get(index)
        if item is None and kind in _TEXT_DELTAS:
            # Supported sparse streams can omit item.added. Keep their observed
            # prefix so the terminal snapshot cannot emit the same text twice.
            identity = payload.get("item_id")
            item = _Item(
                {
                    "type": "reasoning"
                    if kind.startswith("response.reasoning")
                    else "message",
                    "id": identity,
                }
            )
            self._items[index] = item
            if isinstance(identity, str):
                self._indexes[identity] = index
        if item is None:
            return [event]
        if kind in _INPUT_DELTAS:
            delta = payload.get("delta")
            if isinstance(delta, str):
                item.arguments.append(delta)
        elif kind in _INPUT_DONE:
            key = _INPUT_DONE[kind]
            value = payload.get(key)
            if isinstance(value, str):
                return [*self._input_suffix(index, item, key, value), event]
        elif kind in _TEXT_DELTAS:
            index_key, key = _TEXT_DELTAS[kind]
            delta = payload.get("delta")
            if isinstance(delta, str):
                item.text.setdefault(
                    (index_key, payload.get(index_key, 0), key), []
                ).append(delta)
        return [event]

    def _item_index(self, payload: dict[str, Any], body: dict[str, Any]) -> int:
        index = payload.get("output_index")
        if isinstance(index, int) and not isinstance(index, bool):
            return index
        # Some supported native endpoints identify sparse events only by item ID.
        return self._indexes.get(body.get("id"), max(self._items, default=-1) + 1)

    def _snapshot(
        self,
        index: int,
        body: dict[str, Any],
        *,
        event: StreamEvent | None = None,
        terminal: str | None = None,
    ) -> list[StreamEvent]:
        output: list[StreamEvent] = []
        if (
            terminal == "response.failed"
            and body.get("type") in {"function_call", "custom_tool_call"}
            and any(not body.get(key) for key in ("id", "call_id", "name"))
        ):
            # A partial error snapshot cannot complete an invocation. Retain
            # observed state and let the structured request error keep its type.
            return output
        item = self._items.get(index)
        if item is None:
            initial = deepcopy(body)
            if initial.get("type") == "function_call":
                initial["arguments"] = ""
            elif initial.get("type") == "custom_tool_call":
                initial["input"] = ""
            elif initial.get("type") == "message":
                initial["content"] = []
            elif initial.get("type") == "reasoning":
                initial["summary"] = []
            added = StreamEvent(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": initial,
                },
            )
            output.extend(self.feed(added))
            item = self._items[index]
        for key in ("id", "type", "call_id", "name", "namespace"):
            if item.body.get(key) and body.get(key) and item.body[key] != body[key]:
                self._conflict("Responses item identity changed in its final snapshot.")
        if item.body.get("type") in {"function_call", "custom_tool_call"}:
            key = "arguments" if item.body["type"] == "function_call" else "input"
            value = body.get(key)
            if isinstance(value, str):
                output.extend(self._input_suffix(index, item, key, value))
        output.extend(self._text_suffixes(index, item, body))
        completion = self._completion(item, body, terminal)
        if item.completion is ItemCompletion.COMPLETE:
            if completion is not ItemCompletion.COMPLETE:
                self._conflict("Responses contradicted an already finalized item.")
            return output
        item.body = deepcopy(body)
        item.completion = completion
        if event is None:
            event = StreamEvent(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": body,
                },
            )
        output.append(StreamEvent(event.kind, event.payload, completion))
        return output

    def _completion(
        self, item: _Item, body: dict[str, Any], terminal: str | None
    ) -> ItemCompletion:
        status = body.get("status")
        if status not in {None, "completed"}:
            return ItemCompletion.INCOMPLETE
        if (
            terminal in {"response.incomplete", "response.failed"}
            and status != "completed"
            and item.completion is not ItemCompletion.COMPLETE
        ):
            return ItemCompletion.INCOMPLETE
        kind = body.get("type")
        if kind in {"function_call", "custom_tool_call"} and any(
            not isinstance(body.get(key), str) or not body[key]
            for key in ("call_id", "name")
        ):
            return ItemCompletion.INVALID_INPUT
        if kind == "function_call":
            arguments = body.get("arguments")
            if not isinstance(arguments, str) or not complete_json_object(arguments):
                return ItemCompletion.INVALID_INPUT
        elif kind == "custom_tool_call":
            if not isinstance(body.get("input"), str):
                return ItemCompletion.INVALID_INPUT
        elif kind == "tool_search_call" and is_client_search(body):
            if not isinstance(body.get("arguments"), dict):
                return ItemCompletion.INVALID_INPUT
        return ItemCompletion.COMPLETE

    def _input_suffix(
        self, index: int, item: _Item, key: str, value: str
    ) -> list[StreamEvent]:
        observed = "".join(item.arguments)
        if not value.startswith(observed):
            self._conflict(
                "Responses final input conflicts with its streamed arguments."
            )
        suffix = value[len(observed) :]
        if not suffix:
            return []
        item.arguments.append(suffix)
        kind = (
            "response.function_call_arguments.delta"
            if key == "arguments"
            else "response.custom_tool_call_input.delta"
        )
        return [
            StreamEvent(
                kind,
                {
                    "type": kind,
                    "output_index": index,
                    "item_id": item.body.get("id"),
                    "delta": suffix,
                },
            )
        ]

    def _text_suffixes(
        self, index: int, item: _Item, body: dict[str, Any]
    ) -> list[StreamEvent]:
        output: list[StreamEvent] = []
        if body.get("type") not in {"message", "reasoning"}:
            return output
        for container, index_key in (
            ("content", "content_index"),
            ("summary", "summary_index"),
        ):
            parts = body.get(container)
            if not isinstance(parts, list):
                continue
            for position, part in enumerate(parts):
                if not isinstance(part, dict):
                    continue
                key = "refusal" if part.get("type") == "refusal" else "text"
                value = part.get(key)
                if not isinstance(value, str):
                    continue
                fragments = item.text.setdefault((index_key, position, key), [])
                observed = "".join(fragments)
                if not value.startswith(observed):
                    self._conflict(
                        "Responses final text conflicts with its streamed content."
                    )
                suffix = value[len(observed) :]
                if not suffix:
                    continue
                fragments.append(suffix)
                kind = (
                    "response.reasoning_summary_text.delta"
                    if container == "summary"
                    else "response.reasoning_text.delta"
                    if body.get("type") == "reasoning"
                    else (
                        "response.refusal.delta"
                        if key == "refusal"
                        else "response.output_text.delta"
                    )
                )
                output.append(
                    StreamEvent(
                        kind,
                        {
                            "type": kind,
                            "output_index": index,
                            "item_id": item.body.get("id"),
                            index_key: position,
                            "delta": suffix,
                        },
                    )
                )
        return output

    @staticmethod
    def _conflict(message: str) -> None:
        raise AttemptFailure(
            ExecutionFailure(FailureKind.UPSTREAM, 502, message, False),
            blocked_reason=message,
        )
