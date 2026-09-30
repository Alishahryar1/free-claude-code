"""Block finalization for OpenAI Responses streams."""

from ..items import encrypted_reasoning_item, message_item, reasoning_item
from ..tools import (
    custom_tool_input_text_from_arguments,
)
from .blocks import BlockState, ReasoningBlockState, TextBlockState, ToolBlockState
from .event_builders import ResponseEventBuilder
from .ledger import ResponsesOutputLedger


class ResponseBlockCompleter:
    """Finalize active Responses output blocks."""

    def __init__(
        self,
        ledger: ResponsesOutputLedger,
        *,
        events: ResponseEventBuilder,
    ) -> None:
        self._ledger = ledger
        self._events = events

    def complete_block(self, state: BlockState) -> list[str]:
        if isinstance(state, TextBlockState):
            return self._complete_text_block(state)
        if isinstance(state, ReasoningBlockState):
            return self._complete_reasoning_block(state)
        return self._complete_tool_block(state)

    def _complete_text_block(self, state: TextBlockState) -> list[str]:
        text = "".join(state.text_parts)
        item = message_item(state.item_id, text, "completed")
        self._ledger.commit_output(state.output_index, item)
        return [
            self._events.output_text_done(state.item_id, state.output_index, text),
            self._events.content_part_done(state.item_id, state.output_index, text),
            self._events.output_item_done(state.output_index, item),
        ]

    def _complete_reasoning_block(self, state: ReasoningBlockState) -> list[str]:
        item = _reasoning_output_item(state, status="completed")
        self._ledger.commit_output(state.output_index, item)
        chunks: list[str] = []
        text = "".join(state.text_parts)
        if text:
            self._ledger.add_reasoning_text(text)
            chunks.append(
                self._events.reasoning_text_done(
                    state.item_id, state.output_index, text
                )
            )
        chunks.append(self._events.output_item_done(state.output_index, item))
        return chunks

    def _complete_tool_block(self, state: ToolBlockState) -> list[str]:
        if state.kind == "custom":
            return self._complete_custom_tool_block(state)
        arguments = "".join(state.argument_parts)
        item = tool_item(state, status="completed", arguments=arguments)
        chunks: list[str] = []
        if arguments:
            chunks.append(
                self._events.function_call_arguments_delta(
                    state.item_id, state.output_index, arguments
                )
            )
        chunks.extend(
            [
                self._events.function_call_arguments_done(
                    state.item_id, state.output_index, arguments
                ),
                self._events.output_item_done(state.output_index, item),
            ]
        )
        self._ledger.commit_output(state.output_index, item)
        return chunks

    def _complete_custom_tool_block(self, state: ToolBlockState) -> list[str]:
        input_text = custom_tool_input_text_from_arguments(
            "".join(state.argument_parts)
        )
        item = tool_item(state, status="completed", input_text=input_text)
        self._ledger.commit_output(state.output_index, item)
        chunks: list[str] = []
        if input_text:
            chunks.append(
                self._events.custom_tool_call_input_delta(
                    state.item_id, state.output_index, input_text
                )
            )
        chunks.extend(
            [
                self._events.custom_tool_call_input_done(
                    state.item_id, state.output_index, input_text
                ),
                self._events.output_item_done(state.output_index, item),
            ]
        )
        return chunks


def tool_item(
    state: ToolBlockState,
    *,
    status: str,
    arguments: str = "",
    input_text: str = "",
) -> dict[str, object]:
    if state.kind == "custom":
        item: dict[str, object] = {
            "id": state.item_id,
            "type": "custom_tool_call",
            "status": status,
            "call_id": state.call_id,
            "name": state.name,
            "input": input_text,
        }
    else:
        item = {
            "id": state.item_id,
            "type": "function_call",
            "status": status,
            "call_id": state.call_id,
            "name": state.name,
            "arguments": arguments,
        }
    if state.namespace:
        item["namespace"] = state.namespace
    return item


def reasoning_output_item(
    state: ReasoningBlockState, *, status: str
) -> dict[str, object]:
    if state.encrypted_content is not None:
        item = encrypted_reasoning_item(state.item_id, state.encrypted_content, status)
        if state.text_parts:
            item["content"] = [
                {"type": "reasoning_text", "text": "".join(state.text_parts)}
            ]
        return item
    return reasoning_item(state.item_id, "".join(state.text_parts), status)


def _reasoning_output_item(
    state: ReasoningBlockState, *, status: str
) -> dict[str, object]:
    return reasoning_output_item(state, status=status)
