"""Project Responses events directly to Chat, preserving typed reasoning."""

from copy import deepcopy
from typing import Any
from uuid import uuid4

from ..failures import ExecutionFailure
from ..history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
    protected_reasoning,
)
from ..json_types import JsonObject
from .response import ChatOutput, chat_failure


class ChatResponsesStream:
    terminal_failure_completes_wire = False

    def __init__(
        self,
        *,
        model: str,
        origin: ReplayOrigin,
        reserved_tool_ids: frozenset[str],
        include_usage: bool,
        include_thoughts: bool,
    ) -> None:
        self._output = ChatOutput(model, include_usage=include_usage)
        self._origin = origin
        self._reserved = set(reserved_tool_ids)
        self._include_thoughts = include_thoughts
        self._items: dict[int, dict[str, Any]] = {}
        self._text: dict[tuple[int, str, int], str] = {}
        self._arguments: dict[int, str] = {}
        self._completed: dict[int, dict[str, Any]] = {}
        self._calls = 0
        self._emitted_parts: set[tuple[int, str, int]] = set()

    @property
    def completed(self) -> bool:
        return self._output.completed

    def start(self) -> list[str]:
        return self._output.start()

    def feed(self, event_type: str, payload: JsonObject) -> list[str]:
        data: dict[str, Any] = dict(payload)
        if self.completed:
            raise chat_failure("Responses event arrived after completion")
        index = data.get("output_index", 0)
        if event_type in {
            "response.created",
            "response.in_progress",
            "response.queued",
        }:
            return []
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            item = data.get("item")
            if not isinstance(item, dict):
                raise chat_failure("Responses output item is missing")
            return self._item(index, item, complete=event_type.endswith(".done"))
        if event_type in {
            "response.output_text.delta",
            "response.refusal.delta",
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        }:
            field = self._text_field(event_type)
            position = data.get(
                "summary_index" if field == "summary" else "content_index", 0
            )
            value = data.get("delta")
            if not isinstance(value, str):
                raise chat_failure("Responses text delta must be a string")
            key = (index, field, position)
            self._text[key] = self._text.get(key, "") + value
            return self._emit_text(index, field, value, position)
        if event_type in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }:
            value = data.get("arguments" if event_type.endswith(".done") else "delta")
            if not isinstance(value, str):
                raise chat_failure("Responses function arguments must be a string")
            previous = self._arguments.get(index, "")
            if event_type.endswith(".done"):
                if not value.startswith(previous):
                    raise chat_failure("Responses changed streamed function arguments")
                self._arguments[index] = value
            else:
                self._arguments[index] = previous + value
            return []
        if event_type in {
            "response.output_text.done",
            "response.refusal.done",
            "response.reasoning_text.done",
            "response.reasoning_summary_text.done",
        }:
            field = self._text_field(event_type)
            return self._snapshot_text(
                index,
                field,
                data.get("summary_index" if field == "summary" else "content_index", 0),
                data.get("refusal" if field == "refusal" else "text", ""),
            )
        if event_type in {
            "response.content_part.added",
            "response.content_part.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        }:
            return self._part(
                index,
                data.get("summary_index", data.get("content_index", 0)),
                data.get("part", {}),
            )
        if event_type in {"response.completed", "response.incomplete"}:
            response = data.get("response", {})
            events: list[str] = []
            for position, item in enumerate(response.get("output", [])):
                index = next(
                    (
                        i
                        for i, known in self._items.items()
                        if item.get("id") and known.get("id") == item["id"]
                    ),
                    position,
                )
                events.extend(self._item(index, item, complete=True))
            reason = "tool_calls" if self._calls else "stop"
            if event_type == "response.incomplete":
                incomplete = (response.get("incomplete_details") or {}).get("reason")
                if incomplete not in {"max_output_tokens", "content_filter"}:
                    raise chat_failure(
                        f"Chat cannot represent Responses incomplete reason {incomplete}"
                    )
                reason = (
                    "length" if incomplete == "max_output_tokens" else "content_filter"
                )
            events.extend(self._output.finish(reason, _usage(response.get("usage"))))
            return events
        raise chat_failure(
            f"Chat cannot represent upstream Responses event {event_type}"
        )

    @staticmethod
    def _text_field(kind: str) -> str:
        if "summary" in kind:
            return "summary"
        if "reasoning" in kind:
            return "reasoning"
        return "refusal" if "refusal" in kind else "text"

    def _emit_text(
        self, index: int, field: str, value: str, position: int
    ) -> list[str]:
        key = (index, field, position)
        if value and key not in self._emitted_parts:
            if field in {"reasoning", "summary"} and any(
                owner == index and channel == field
                for owner, channel, _ in self._emitted_parts
            ):
                value = "\n\n" + value
            self._emitted_parts.add(key)
        if field in {"reasoning", "summary"}:
            return (
                self._output.thought(
                    value, summary=field == "summary", source=f"rs_{index}"
                )
                if self._include_thoughts
                else []
            )
        return self._output.delta(
            {"refusal" if field == "refusal" else "content": value}
        )

    def _snapshot_text(
        self, index: int, field: str, position: int, text: str
    ) -> list[str]:
        if not isinstance(text, str):
            raise chat_failure("Responses snapshot text must be a string")
        key = (index, field, position)
        previous = self._text.get(key, "")
        if not text.startswith(previous):
            raise chat_failure("Responses snapshot changed streamed text")
        self._text[key] = text
        return (
            self._emit_text(index, field, text[len(previous) :], position)
            if text != previous
            else []
        )

    def _part(self, index: int, position: int, part: dict[str, Any]) -> list[str]:
        kind = part.get("type")
        if kind not in {"output_text", "refusal", "summary_text", "reasoning_text"}:
            raise chat_failure(f"Chat cannot represent Responses content {kind}")
        if part.get("annotations"):
            raise chat_failure("Chat cannot represent native Responses annotations")
        return self._snapshot_text(
            index,
            self._text_field(kind),
            position,
            part.get("refusal" if kind == "refusal" else "text", ""),
        )

    def _item(self, index: int, item: dict[str, Any], *, complete: bool) -> list[str]:
        if index in self._completed:
            if self._completed[index] != item:
                raise chat_failure("Responses changed a completed output item")
            return []
        kind = item.get("type")
        self._items[index] = {"id": item.get("id"), "type": kind}
        if complete:
            self._completed[index] = deepcopy(item)
        events: list[str] = []
        if kind in {"message", "reasoning"}:
            for field in ("summary", "content"):
                for position, part in enumerate(item.get(field) or []):
                    events.extend(self._part(index, position, part))
            if complete and kind == "reasoning" and protected_reasoning(item):
                events.extend(
                    self._output.replay(
                        encode_replay(ReplayRecord(self._origin, deepcopy(item))),
                        source=f"rs_{index}",
                    )
                )
            return events
        if kind != "function_call":
            raise chat_failure(f"Chat cannot represent Responses output {kind}")
        if not complete or item.get("status") in {
            "in_progress",
            "incomplete",
            "failed",
        }:
            return []
        name = item.get("name", "")
        if not isinstance(name, str) or not name:
            raise chat_failure("Responses function call requires a name")
        arguments = item.get("arguments", self._arguments.get(index, ""))
        if not isinstance(arguments, str) or not arguments.startswith(
            self._arguments.get(index, "")
        ):
            raise chat_failure("Responses changed streamed function arguments")
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in self._reserved:
            call_id = f"call_{uuid4().hex}"
        self._reserved.add(call_id)
        events = self._output.tool(self._calls, call_id, name, arguments)
        self._calls += 1
        return events

    def terminal_failure(
        self, raw_error: Exception, failure: ExecutionFailure
    ) -> list[str]:
        return []

    def failure_payload(self, raw_error: Exception, failure: ExecutionFailure) -> None:
        return None


def _usage(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        **{
            target: value[source]
            for source, target in (
                ("input_tokens", "prompt_tokens"),
                ("output_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
            )
            if source in value
        },
        **(
            {"prompt_tokens_details": deepcopy(value["input_tokens_details"])}
            if "input_tokens_details" in value
            else {}
        ),
        **(
            {"completion_tokens_details": deepcopy(value["output_tokens_details"])}
            if "output_tokens_details" in value
            else {}
        ),
    }
