"""Project private model turns and local web actions into one Responses stream."""

import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.sse import SSEEvent

from .errors import ResponsesConversionError, openai_error_from_failure
from .models import OpenAIResponsesRequest
from .streaming.event_builders import ResponseEventBuilder
from .web_request import WebSearchSpec, public_action, validate_action

_TERMINALS = {"response.completed", "response.failed", "response.incomplete"}


@dataclass(slots=True)
class WebOutputSlot:
    index: int
    identity: str
    web: bool | None
    item: JsonObject
    private: JsonObject = field(default_factory=dict)
    buffered: list[SSEEvent] = field(default_factory=list)
    annotations: set[int] = field(default_factory=set)


class WebResponsePresenter:
    def __init__(self, request: OpenAIResponsesRequest, spec: WebSearchSpec) -> None:
        self.request = request
        self.spec = spec
        self.events = ResponseEventBuilder()
        self.slots: list[WebOutputSlot] = []
        self.current: dict[int, WebOutputSlot] = {}
        self.response: JsonObject | None = None
        self.terminal: JsonObject | None = None
        self.sources: dict[str, str] = {}
        self._usage: dict[str, Any] = {}
        self._usage_complete = True

    def begin_turn(self) -> None:
        self.current = {}
        self.terminal = None

    def feed(self, event: SSEEvent) -> list[str]:
        kind, data = (
            event.event or str(event.data.get("type", "")),
            deepcopy(event.data),
        )
        if self.terminal is not None:
            raise ResponsesConversionError(
                "Provider emitted data after its terminal response."
            )
        if kind in ("response.created", "response.in_progress"):
            if self.response is not None:
                return []
            response = data.get("response")
            if not isinstance(response, dict):
                raise ResponsesConversionError(
                    "Provider response.created has no response."
                )
            self.response = deepcopy(response)
            return [self.events.response_created(self.payload("in_progress"))]
        if kind in _TERMINALS:
            response = data.get("response")
            if not isinstance(response, dict) or not isinstance(
                response.get("output"), list
            ):
                raise ResponsesConversionError(
                    "Provider terminal has no output snapshot."
                )
            self.terminal = response
            usage = response.get("usage")
            if isinstance(usage, dict):
                _sum_usage(self._usage, usage)
            else:
                self._usage_complete = False
            return []
        if kind == "error":
            raise ResponsesConversionError(
                "Provider returned a stream error during web execution."
            )
        index = data.get("output_index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ResponsesConversionError(
                f"Unexpected model event in local web workflow: {kind}."
            )
        if kind == "response.output_item.added":
            if index in self.current or not isinstance(data.get("item"), dict):
                raise ResponsesConversionError(
                    "Duplicate or malformed provider output item."
                )
            item = data["item"]
            name = item.get("name")
            web = (
                (name == self.spec.name and not item.get("namespace"))
                if item.get("type") == "function_call"
                else False
            )
            if item.get("type") == "function_call" and not name:
                web = None
            prefix = (
                "ws"
                if web
                else {"reasoning": "rs", "message": "msg"}.get(
                    str(item.get("type")), "fc"
                )
            )
            slot = WebOutputSlot(
                len(self.slots), f"{prefix}_{uuid.uuid4().hex}", web, deepcopy(item)
            )
            self.current[index] = slot
            self.slots.append(slot)
            if web is None:
                slot.buffered.append(event)
                return []
            return self._added(slot)
        slot = self.current.get(index)
        if slot is None:
            raise ResponsesConversionError(
                "Provider event references an unknown output slot."
            )
        if slot.web is None:
            slot.buffered.append(event)
            if kind != "response.output_item.done":
                return []
            item = data.get("item")
            if not isinstance(item, dict) or not item.get("name"):
                raise ResponsesConversionError(
                    "Provider completed a function without a name."
                )
            slot.web = item.get("name") == self.spec.name and not item.get("namespace")
            if slot.web:
                slot.identity = f"ws_{uuid.uuid4().hex}"
            slot.item["name"] = item["name"]
            output = self._added(slot)
            pending, slot.buffered = slot.buffered[1:], []
            for deferred in pending:
                output.extend(self.feed(deferred))
            return output
        if kind == "response.output_item.done":
            item = data.get("item")
            if not isinstance(item, dict):
                raise ResponsesConversionError(
                    "Provider completed a malformed output item."
                )
            slot.private = deepcopy(item)
            if slot.web:
                try:
                    args = validate_action(json.loads(str(item.get("arguments", ""))))
                    slot.item["action"] = public_action(args)
                except ValueError, TypeError:
                    pass  # The application returns a bounded argument error as the tool result.
                return []
            slot.item = {**item, "id": slot.identity}
            output = self._annotate_item(slot)
            output.append(self.events.output_item_done(slot.index, slot.item))
            return output
        if slot.web:
            return []
        self._capture_delta(slot, kind, data)
        data["output_index"] = slot.index
        if "item_id" in data:
            data["item_id"] = slot.identity
        output: list[str] = []
        if kind == "response.content_part.done" and isinstance(data.get("part"), dict):
            output.extend(
                self._annotate_part(
                    slot, data["part"], int(data.get("content_index", 0))
                )
            )
        output.append(self.events.emit(kind, cast(JsonObject, data)))
        return output

    def _capture_delta(self, slot: WebOutputSlot, kind: str, data: JsonObject) -> None:
        delta = data.get("delta")
        if not isinstance(delta, str):
            return
        fields = {
            "response.function_call_arguments.delta": "arguments",
            "response.custom_tool_call_input.delta": "input",
        }
        if field_name := fields.get(kind):
            slot.item[field_name] = str(slot.item.get(field_name, "")) + delta
        elif kind in (
            "response.output_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        ):
            summary = kind == "response.reasoning_summary_text.delta"
            field_name = "summary" if summary else "content"
            parts = slot.item.get(field_name)
            if not isinstance(parts, list):
                parts = []
                slot.item[field_name] = parts
            index = data.get("summary_index" if summary else "content_index", 0)
            if not isinstance(index, int) or index < 0:
                raise ResponsesConversionError("Invalid streamed text index.")
            if index > len(parts):
                raise ResponsesConversionError("Noncontiguous streamed text index.")
            if index == len(parts):
                parts.append(
                    {
                        "type": "summary_text"
                        if summary
                        else "output_text"
                        if kind == "response.output_text.delta"
                        else "reasoning_text",
                        "text": "",
                    }
                )
            part = parts[index]
            if isinstance(part, dict):
                part["text"] = str(part.get("text", "")) + delta

    def fail(self, failure: ExecutionFailure) -> list[str]:
        frames = []
        self._usage_complete = False
        for slot in self.slots:
            if slot.item.get("status") not in ("completed", "failed", "incomplete"):
                slot.item["status"] = "incomplete"
                frames.extend(self._annotate_item(slot))
                frames.append(self.events.output_item_done(slot.index, slot.item))
        self.terminal = {
            "status": "failed",
            "error": openai_error_from_failure(failure),
        }
        return frames

    def _added(self, slot: WebOutputSlot) -> list[str]:
        if slot.web:
            slot.item = {
                "id": slot.identity,
                "type": "web_search_call",
                "status": "in_progress",
            }
        else:
            slot.item["id"] = slot.identity
        frames = [self.events.output_item_added(slot.index, slot.item)]
        if slot.web:
            frames.append(
                self.events.emit(
                    "response.web_search_call.in_progress",
                    {"item_id": slot.identity, "output_index": slot.index},
                )
            )
        return frames

    def web_slots(self) -> list[WebOutputSlot]:
        return [slot for slot in self.current.values() if slot.web]

    def private_output(self) -> list[JsonObject]:
        if self.terminal is None:
            raise ResponsesConversionError(
                "Provider ended without a terminal response."
            )
        output = self.terminal["output"]
        if not isinstance(output, list) or any(
            not isinstance(item, dict) for item in output
        ):
            raise ResponsesConversionError("Malformed private output snapshot.")
        for slot in self.current.values():
            if not slot.private:
                raise ResponsesConversionError(
                    "Provider did not finish an output item."
                )
            if slot.private not in output:
                raise ResponsesConversionError(
                    "Provider terminal disagrees with its completed output items."
                )
        return cast(list[JsonObject], deepcopy(output))

    def searching(self, slot: WebOutputSlot) -> str:
        return self.events.emit(
            "response.web_search_call.searching",
            {"item_id": slot.identity, "output_index": slot.index},
        )

    def complete_web(
        self, slot: WebOutputSlot, result: JsonObject, *, status: str | None = None
    ) -> list[str]:
        slot.item["status"] = status or ("failed" if "error" in result else "completed")
        sources = result.get("sources")
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict) and isinstance(source.get("url"), str):
                    self.sources[str(source["url"])] = str(
                        source.get("title") or source["url"]
                    )
            action = slot.item.get("action")
            if self.spec.include_sources and isinstance(action, dict):
                action["sources"] = [
                    {"type": "url", "url": source["url"]}
                    for source in sources
                    if isinstance(source, dict) and isinstance(source.get("url"), str)
                ]
        output = []
        if slot.item["status"] == "completed":
            output.append(
                self.events.emit(
                    "response.web_search_call.completed",
                    {"item_id": slot.identity, "output_index": slot.index},
                )
            )
        output.append(self.events.output_item_done(slot.index, slot.item))
        return output

    def append_replay(self, item: JsonObject) -> list[str]:
        slot = WebOutputSlot(len(self.slots), str(item["id"]), False, item)
        self.slots.append(slot)
        return [
            self.events.output_item_added(slot.index, item),
            self.events.output_item_done(slot.index, item),
        ]

    def _annotate_part(
        self, slot: WebOutputSlot, part: JsonObject, index: int
    ) -> list[str]:
        text = part.get("text")
        if part.get("type") != "output_text" or not isinstance(text, str):
            return []
        annotations: list[JsonObject] = []
        for url, title in self.sources.items():
            needle = f"]({url})"
            start = 0
            while (position := text.find(needle, start)) >= 0:
                offset = position + 2
                annotations.append(
                    {
                        "type": "url_citation",
                        "url": url,
                        "title": title,
                        "start_index": offset,
                        "end_index": offset + len(url),
                    }
                )
                start = offset + len(url) + 1
        annotations.sort(key=lambda annotation: cast(int, annotation["start_index"]))
        part["annotations"] = annotations
        if index in slot.annotations:
            return []
        slot.annotations.add(index)
        return [
            self.events.emit(
                "response.output_text.annotation.added",
                {
                    "item_id": slot.identity,
                    "output_index": slot.index,
                    "content_index": index,
                    "annotation_index": number,
                    "annotation": annotation,
                },
            )
            for number, annotation in enumerate(annotations)
        ]

    def _annotate_item(self, slot: WebOutputSlot) -> list[str]:
        content = slot.item.get("content")
        frames = []
        if isinstance(content, list):
            for index, part in enumerate(content):
                if isinstance(part, dict):
                    frames.extend(self._annotate_part(slot, part, index))
        return frames

    def payload(self, status: str) -> JsonObject:
        if self.response is None:
            raise ResponsesConversionError("Provider did not start a response.")
        result = deepcopy(self.response)
        result.update(
            status=status,
            output=[deepcopy(slot.item) for slot in self.slots],
            tools=self.request.tools or [],
            tool_choice=self.request.tool_choice or "auto",
            usage=deepcopy(self._usage)
            if self._usage_complete and self._usage
            else None,
        )
        if self.terminal is not None:
            for key in ("error", "incomplete_details"):
                result[key] = self.terminal.get(key)
        return result

    def finish(self, *, incomplete: bool = False) -> str:
        status = (
            "incomplete"
            if incomplete
            else str((self.terminal or {}).get("status", "completed"))
        )
        if status not in ("completed", "incomplete", "failed"):
            raise ResponsesConversionError("Invalid provider terminal status.")
        payload = self.payload(status)
        if incomplete:
            payload["incomplete_details"] = {"reason": "max_output_tokens"}
        return self.events.emit(f"response.{status}", {"response": payload})


def _sum_usage(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            target[key] = target.get(key, 0) + value
        elif isinstance(value, dict):
            child = target.setdefault(key, {})
            if isinstance(child, dict):
                _sum_usage(child, value)
