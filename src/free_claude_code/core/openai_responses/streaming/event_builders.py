"""OpenAI Responses SSE event builders."""

from collections.abc import Callable, Iterable
from typing import Any, cast

from free_claude_code.core.json_types import JsonObject

from ..events import format_response_sse_event


class ResponseEventBuilder:
    """Build one ordered Responses event stream."""

    def __init__(
        self,
        transform: Callable[[str, JsonObject], Iterable[tuple[str, JsonObject]]]
        | None = None,
    ) -> None:
        self._next_sequence_number = 0
        self._transform = transform

    def response_created(self, response: dict[str, Any]) -> str:
        return self._format(
            "response.created",
            {"type": "response.created", "response": response},
        )

    def response_completed(self, response: dict[str, Any]) -> str:
        return self._format(
            "response.completed",
            {"type": "response.completed", "response": response},
        )

    def response_incomplete(self, response: dict[str, Any]) -> str:
        return self._format(
            "response.incomplete",
            {"type": "response.incomplete", "response": response},
        )

    def response_failed(self, response: dict[str, Any]) -> str:
        return self._format(
            "response.failed",
            {"type": "response.failed", "response": response},
        )

    def output_item_added(self, output_index: int, item: dict[str, Any]) -> str:
        return self._format(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": item,
            },
        )

    def output_item_done(self, output_index: int, item: dict[str, Any]) -> str:
        return self._format(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item,
            },
        )

    def content_part_added(self, item_id: str, output_index: int) -> str:
        return self._format(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )

    def output_text_delta(self, item_id: str, output_index: int, text: str) -> str:
        return self._format(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            },
        )

    def output_text_done(self, item_id: str, output_index: int, text: str) -> str:
        return self._format(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            },
        )

    def content_part_done(self, item_id: str, output_index: int, text: str) -> str:
        return self._format(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
        )

    def reasoning_text_delta(self, item_id: str, output_index: int, text: str) -> str:
        return self._format(
            "response.reasoning_text.delta",
            {
                "type": "response.reasoning_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            },
        )

    def reasoning_text_done(self, item_id: str, output_index: int, text: str) -> str:
        return self._format(
            "response.reasoning_text.done",
            {
                "type": "response.reasoning_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            },
        )

    def function_call_arguments_delta(
        self, item_id: str, output_index: int, arguments: str
    ) -> str:
        return self._format(
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item_id,
                "output_index": output_index,
                "delta": arguments,
            },
        )

    def function_call_arguments_done(
        self, item_id: str, output_index: int, arguments: str
    ) -> str:
        return self._format(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": item_id,
                "output_index": output_index,
                "arguments": arguments,
            },
        )

    def custom_tool_call_input_delta(
        self, item_id: str, output_index: int, input_text: str
    ) -> str:
        return self._format(
            "response.custom_tool_call_input.delta",
            {
                "type": "response.custom_tool_call_input.delta",
                "item_id": item_id,
                "output_index": output_index,
                "delta": input_text,
            },
        )

    def custom_tool_call_input_done(
        self, item_id: str, output_index: int, input_text: str
    ) -> str:
        return self._format(
            "response.custom_tool_call_input.done",
            {
                "type": "response.custom_tool_call_input.done",
                "item_id": item_id,
                "output_index": output_index,
                "input": input_text,
            },
        )

    def _format(self, event_type: str, data: dict[str, Any]) -> str:
        data["sequence_number"] = self._next_sequence_number
        self._next_sequence_number += 1
        if self._transform is not None:
            return "".join(
                format_response_sse_event(kind, payload)
                for kind, payload in self._transform(event_type, cast(JsonObject, data))
            )
        return format_response_sse_event(event_type, data)
