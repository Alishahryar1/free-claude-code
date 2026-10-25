"""Direct Responses-source projection to Gemini candidates."""

from copy import deepcopy
from typing import Any
from uuid import uuid4

from ..failures import ExecutionFailure
from ..history_replay import ReplayOrigin, ReplayRecord
from ..json_types import JsonObject
from ..openai_tool_names import OpenAIToolNameCodec
from .response import GeminiOutput, conversion_failure, function_arguments


class GeminiResponsesStream:
    terminal_failure_completes_wire = False

    def __init__(
        self,
        *,
        response_id: str,
        model: str,
        include_thoughts: bool,
        origin: ReplayOrigin,
        tool_names: OpenAIToolNameCodec,
        declared_names: frozenset[str],
        reserved_tool_ids: frozenset[str],
    ) -> None:
        self._output = GeminiOutput(
            response_id, model, include_thoughts=include_thoughts
        )
        self._origin = origin
        self._names = tool_names
        self._declared = declared_names
        self._reserved = set(reserved_tool_ids)
        self._items: dict[int, dict[str, Any]] = {}
        self._text: dict[tuple[int, str, int], str] = {}
        self._arguments: dict[int, str] = {}
        self._calls: dict[int, dict[str, Any]] = {}
        self._reasoning: dict[int, JsonObject] = {}
        self._refusal = False

    @property
    def completed(self) -> bool:
        return self._output.completed

    def start(self) -> list[str]:
        return []

    def handoff(self) -> list[str]:
        self._output.records = [
            ReplayRecord(self._origin, record)
            for _, record in sorted(self._reasoning.items())
        ]
        return self._output.handoff()

    def feed(self, event_type: str, payload: JsonObject) -> list[str]:
        # Upstream SDK events have already been decoded to JSON objects.
        data: dict[str, Any] = dict(payload)
        if self.completed:
            raise conversion_failure("Responses event arrived after completion")
        index = data.get("output_index", 0)
        if event_type in {
            "response.created",
            "response.in_progress",
            "response.queued",
        }:
            return []
        if event_type == "response.output_item.added":
            item = data.get("item")
            if not isinstance(item, dict):
                raise conversion_failure("Responses output item is missing")
            self._items[index] = deepcopy(item)
            return self._item(index, item, complete=False)
        if event_type in {
            "response.output_text.delta",
            "response.refusal.delta",
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        }:
            thought = "reasoning" in event_type
            field = "summary" if "summary" in event_type else "content"
            position = data.get(
                "summary_index" if field == "summary" else "content_index", 0
            )
            self._refusal |= "refusal" in event_type
            value = data.get("delta")
            if not isinstance(value, str):
                raise conversion_failure("Responses text delta must be a string")
            key = (index, field, position)
            self._text[key] = self._text.get(key, "") + value
            return self._output.part(
                {"text": value, **({"thought": True} if thought else {})}
            )
        if event_type == "response.function_call_arguments.delta":
            value = data.get("delta")
            if not isinstance(value, str):
                raise conversion_failure("Responses function delta must be a string")
            self._arguments[index] = self._arguments.get(index, "") + value
            return []
        if event_type == "response.function_call_arguments.done":
            value = data.get("arguments")
            if not isinstance(value, str):
                raise conversion_failure(
                    "Responses function arguments must be a string"
                )
            previous = self._arguments.get(index, "")
            if not value.startswith(previous):
                raise conversion_failure(
                    "Responses changed streamed function arguments"
                )
            self._arguments[index] = value
            return []
        if event_type in {
            "response.output_text.done",
            "response.refusal.done",
            "response.reasoning_text.done",
            "response.reasoning_summary_text.done",
        }:
            field = "summary" if "summary" in event_type else "content"
            position = data.get(
                "summary_index" if field == "summary" else "content_index", 0
            )
            self._refusal |= "refusal" in event_type
            return self._snapshot_text(
                index,
                field,
                position,
                data.get("refusal" if "refusal" in event_type else "text", ""),
                thought="reasoning" in event_type,
            )
        if event_type in {
            "response.content_part.added",
            "response.content_part.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        }:
            part = data.get("part", {})
            field = "summary" if "summary" in event_type else "content"
            position = data.get(
                "summary_index" if field == "summary" else "content_index", 0
            )
            return self._part(index, field, position, part)
        if event_type == "response.output_item.done":
            item = data.get("item")
            if not isinstance(item, dict):
                raise conversion_failure("Responses completed item is missing")
            self._items[index] = deepcopy(item)
            return self._item(index, item, complete=True)
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
            reason = "SAFETY" if self._refusal else "STOP"
            if event_type == "response.incomplete":
                incomplete = (response.get("incomplete_details") or {}).get("reason")
                if incomplete not in {"max_output_tokens", "content_filter"}:
                    raise conversion_failure(
                        f"Gemini cannot represent Responses incomplete reason {incomplete}"
                    )
                reason = "MAX_TOKENS" if incomplete == "max_output_tokens" else "SAFETY"
            self._output.records.extend(
                ReplayRecord(self._origin, record)
                for _, record in sorted(self._reasoning.items())
            )
            events.extend(
                self._output.finish(
                    reason,
                    _usage(response.get("usage"), has_reasoning=bool(self._reasoning)),
                )
            )
            return events
        if event_type == "response.output_text.annotation.added":
            raise conversion_failure(
                "Gemini cannot represent upstream native annotations"
            )
        raise conversion_failure(
            f"Gemini cannot represent upstream Responses event {event_type}"
        )

    def _snapshot_text(
        self, index: int, field: str, position: int, text: str, *, thought: bool
    ) -> list[str]:
        if not isinstance(text, str):
            raise conversion_failure("Responses snapshot text must be a string")
        key = (index, field, position)
        previous = self._text.get(key, "")
        if not text.startswith(previous):
            raise conversion_failure("Responses snapshot changed streamed text")
        self._text[key] = text
        suffix = text[len(previous) :]
        return (
            self._output.part(
                {"text": suffix, **({"thought": True} if thought else {})}
            )
            if suffix
            else []
        )

    def _part(
        self, index: int, field: str, position: int, part: dict[str, Any]
    ) -> list[str]:
        kind = part.get("type")
        if kind not in {"output_text", "refusal", "summary_text", "reasoning_text"}:
            raise conversion_failure(
                f"Gemini cannot represent Responses content {kind}"
            )
        if part.get("annotations"):
            raise conversion_failure(
                "Gemini cannot represent upstream native annotations"
            )
        self._refusal |= kind == "refusal"
        return self._snapshot_text(
            index,
            field,
            position,
            part.get("refusal" if kind == "refusal" else "text", ""),
            thought=kind in {"summary_text", "reasoning_text"},
        )

    def _item(self, index: int, item: dict[str, Any], *, complete: bool) -> list[str]:
        kind = item.get("type")
        events: list[str] = []
        if kind in {"message", "reasoning"}:
            for field in ("summary", "content"):
                for position, part in enumerate(item.get(field) or []):
                    events.extend(self._part(index, field, position, part))
            if complete and kind == "reasoning":
                self._reasoning[index] = deepcopy(item)
            return events
        if kind != "function_call":
            raise conversion_failure(f"Gemini cannot represent Responses output {kind}")
        if not complete or item.get("status") in {
            "in_progress",
            "incomplete",
            "failed",
        }:
            return []
        if index in self._calls:
            if self._calls[index] != item:
                raise conversion_failure("Responses changed a completed function call")
            return []
        name = self._names.decode(str(item.get("name", "")))
        if name not in self._declared:
            raise conversion_failure(f"Upstream called an undeclared function: {name}")
        arguments = item.get("arguments", self._arguments.get(index, ""))
        if not isinstance(arguments, str) or not arguments.startswith(
            self._arguments.get(index, "")
        ):
            raise conversion_failure("Responses changed streamed function arguments")
        parsed = function_arguments(arguments)
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in self._reserved:
            call_id = f"call_{uuid4().hex}"
        self._reserved.add(call_id)
        self._calls[index] = deepcopy(item)
        return self._output.part(
            {"functionCall": {"id": call_id, "name": name, "args": parsed}}
        )

    def terminal_failure(
        self, raw_error: Exception, failure: ExecutionFailure
    ) -> list[str]:
        return []

    def failure_payload(self, raw_error: Exception, failure: ExecutionFailure) -> None:
        return None


def _usage(value: object, *, has_reasoning: bool) -> JsonObject:
    if not isinstance(value, dict):
        return {}
    result: JsonObject = {}
    for source, target in (
        ("input_tokens", "promptTokenCount"),
        ("output_tokens", "candidatesTokenCount"),
        ("total_tokens", "totalTokenCount"),
    ):
        count = value.get(source)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            result[target] = count
    for field, source, target in (
        ("input_tokens_details", "cached_tokens", "cachedContentTokenCount"),
        ("output_tokens_details", "reasoning_tokens", "thoughtsTokenCount"),
    ):
        count = (value.get(field) or {}).get(source)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            result[target] = count
    output, thoughts = (
        result.get("candidatesTokenCount"),
        result.get("thoughtsTokenCount"),
    )
    if isinstance(output, int) and isinstance(thoughts, int):
        result["candidatesTokenCount"] = max(0, output - thoughts)
    elif has_reasoning:
        result.pop("candidatesTokenCount", None)
    return result
