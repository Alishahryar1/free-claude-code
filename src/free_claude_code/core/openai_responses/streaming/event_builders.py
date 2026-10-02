"""OpenAI Responses SSE event builders."""

from typing import Any

from free_claude_code.core.stream_events import StreamEvent


class ResponseEventBuilder:
    """Build one ordered Responses event stream."""

    def response_created(self, response: dict[str, Any]) -> StreamEvent:
        return self._format(
            "response.created",
            {"type": "response.created", "response": response},
        )

    def response_completed(self, response: dict[str, Any]) -> StreamEvent:
        return self._format(
            "response.completed",
            {"type": "response.completed", "response": response},
        )

    def response_incomplete(self, response: dict[str, Any]) -> StreamEvent:
        return self._format(
            "response.incomplete",
            {"type": "response.incomplete", "response": response},
        )

    def response_failed(self, response: dict[str, Any]) -> StreamEvent:
        return self._format(
            "response.failed",
            {"type": "response.failed", "response": response},
        )

    def output_item_added(self, output_index: int, item: dict[str, Any]) -> StreamEvent:
        return self._format(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": item,
            },
        )

    def output_item_done(self, output_index: int, item: dict[str, Any]) -> StreamEvent:
        return self._format(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item,
            },
        )

    def content_part_added(self, item_id: str, output_index: int) -> StreamEvent:
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

    def output_text_delta(
        self, item_id: str, output_index: int, text: str
    ) -> StreamEvent:
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

    def output_text_done(
        self, item_id: str, output_index: int, text: str
    ) -> StreamEvent:
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

    def content_part_done(
        self, item_id: str, output_index: int, text: str
    ) -> StreamEvent:
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

    def reasoning_text_delta(
        self, item_id: str, output_index: int, text: str
    ) -> StreamEvent:
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

    def reasoning_text_done(
        self, item_id: str, output_index: int, text: str
    ) -> StreamEvent:
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
    ) -> StreamEvent:
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
    ) -> StreamEvent:
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
    ) -> StreamEvent:
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
    ) -> StreamEvent:
        return self._format(
            "response.custom_tool_call_input.done",
            {
                "type": "response.custom_tool_call_input.done",
                "item_id": item_id,
                "output_index": output_index,
                "input": input_text,
            },
        )

    def _format(self, event_type: str, data: dict[str, Any]) -> StreamEvent:
        return StreamEvent(event_type, data)
