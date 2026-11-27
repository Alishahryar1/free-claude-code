"""Read the conversation in a Responses request for a foreign provider."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from ..history_replay import tool_history_context
from ..json_types import JsonObject, JsonValue, load_json_value
from ..openai_chat import computer_screenshot_label
from .errors import ResponsesConversionError
from .tool_calls import is_client_call
from .tools import ResponsesToolIdentity, optional_str, required_str

_TOOL_RECORDS = {
    "web_search_call",
    "file_search_call",
    "image_generation_call",
    "code_interpreter_call",
    "mcp_call",
    "mcp_list_tools",
    "computer_call",
    "shell_call",
    "shell_call_output",
    "local_shell_call",
    "local_shell_call_output",
    "apply_patch_call",
    "apply_patch_call_output",
    "tool_search_call",
    "tool_search_output",
}


@dataclass(frozen=True, slots=True)
class InputMessage:
    role: str
    content: str | list[JsonObject]


@dataclass(frozen=True, slots=True)
class ReasoningItem:
    data: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class InputCall:
    call_id: str
    identity: ResponsesToolIdentity
    arguments: JsonValue
    extra_content: JsonValue

    def object_arguments(self) -> JsonObject:
        if self.identity.kind == "custom":
            return {"input": self.arguments}
        value = self.arguments
        if isinstance(value, str):
            try:
                value = load_json_value(value)
            except (ValueError, RecursionError) as error:
                raise ResponsesConversionError(
                    "Function arguments must be valid JSON."
                ) from error
        if not isinstance(value, dict):
            raise ResponsesConversionError("Function arguments must be a JSON object.")
        return value


@dataclass(frozen=True, slots=True)
class InputResult:
    call_id: str
    kind: Literal["function", "custom"]
    output: JsonValue
    is_error: bool | None


@dataclass(frozen=True, slots=True)
class Screenshot:
    content: list[JsonObject]


type HistoryItem = InputMessage | ReasoningItem | InputCall | InputResult | Screenshot


def input_items(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else [] if value is None else [value]


def _content(
    value: JsonValue, *, keep_empty_text: bool = False
) -> str | list[JsonObject]:
    if isinstance(value, str):
        return value
    parts: list[JsonObject] = []
    for part in input_items(value):
        if isinstance(part, str):
            if part or keep_empty_text:
                parts.append({"type": "input_text", "text": part})
            continue
        if not isinstance(part, dict) or not isinstance(kind := part.get("type"), str):
            raise ResponsesConversionError("Responses content blocks require a type")
        if kind in ("input_text", "output_text", "text", "refusal"):
            text = part.get("refusal" if kind == "refusal" else "text")
            if not isinstance(text, str):
                raise ResponsesConversionError(
                    "Responses text content requires a string"
                )
            if text or keep_empty_text:
                parts.append({"type": "input_text", "text": text})
        elif kind == "input_image":
            parts.append(part)
    return parts


def read_history(value: JsonValue) -> Iterator[tuple[int, HistoryItem]]:
    """Keep source positions so tool discovery can restore original wire identities."""
    pending_actions: dict[str, str] = {}
    latest_user: InputMessage | None = None
    for index, raw in enumerate(input_items(value)):
        item = _read_item(raw, pending_actions)
        if isinstance(item, InputMessage) and item.role == "user":
            latest_user = item
        if item is not None:
            yield index, item
    if latest_user is not None and not latest_user.content:
        raise ResponsesConversionError(
            "The latest user message has no translatable content"
        )
    if pending_actions:
        raise ResponsesConversionError(
            "A different protocol cannot finish an unresolved tool action"
        )


def _read_item(raw: JsonValue, pending_actions: dict[str, str]) -> HistoryItem | None:
    if isinstance(raw, str):
        return InputMessage("user", raw)
    if not isinstance(raw, dict):
        raise ResponsesConversionError("Responses input items must be text or objects")
    kind = raw.get("type", "message")
    if (
        "id" in raw
        and raw.get("type") is None
        and "role" not in raw
        and "content" not in raw
    ):
        kind = "item_reference"
    if not isinstance(kind, str):
        raise ResponsesConversionError("Responses input item requires a type")
    if (
        (
            kind
            in (
                "function_call",
                "function_call_output",
                "custom_tool_call",
                "custom_tool_call_output",
            )
            or kind in _TOOL_RECORDS
        )
        and isinstance(caller := raw.get("caller"), dict)
        and caller.get("type") == "program"
    ):
        raise ResponsesConversionError(
            "Cannot translate Responses context from a program caller"
        )
    if kind.endswith("_output") and isinstance(call_id := raw.get("call_id"), str):
        expected = pending_actions.pop(call_id, None)
        if expected is not None and kind != expected:
            raise ResponsesConversionError("Tool output type does not match its call.")
    if kind == "message":
        role = raw.get("role", "user")
        if not isinstance(role, str) or role not in (
            "user",
            "assistant",
            "system",
            "developer",
        ):
            raise ResponsesConversionError("Unsupported Responses message role.")
        return InputMessage(role, _content(raw.get("content")))
    if kind in ("input_text", "output_text", "text", "input_image"):
        return InputMessage("user", _content([raw]))
    if kind == "reasoning":
        return ReasoningItem(raw)
    if kind in ("function_call", "custom_tool_call"):
        custom = kind == "custom_tool_call"
        arguments = raw.get("input" if custom else "arguments")
        if custom and not isinstance(arguments, str):
            raise ResponsesConversionError("Custom tool calls require text input")
        identity = ResponsesToolIdentity(
            "custom" if custom else "function",
            required_str(raw.get("name"), "tool.name"),
            optional_str(raw.get("namespace")),
        )
        return InputCall(
            required_str(raw.get("call_id"), "tool.call_id"),
            identity,
            arguments,
            raw.get("extra_content"),
        )
    if kind in ("function_call_output", "custom_tool_call_output"):
        output = raw.get("output")
        if isinstance(output, list):
            content = _content(output, keep_empty_text=True)
            if output and not content:
                raise ResponsesConversionError(
                    "Tool result has no translatable content"
                )
            output = content
        is_error = raw.get("is_error")
        if is_error is not None and not isinstance(is_error, bool):
            raise ResponsesConversionError("Tool result is_error must be a boolean.")
        return InputResult(
            required_str(raw.get("call_id"), "tool output.call_id"),
            "custom" if kind == "custom_tool_call_output" else "function",
            output,
            is_error,
        )
    if kind == "computer_call_output":
        output = raw.get("output")
        if (
            not isinstance(output, dict)
            or output.get("type") != "computer_screenshot"
            or not output.get("image_url")
        ):
            raise ResponsesConversionError(
                "computer_call_output requires a screenshot with an image URL"
            )
        call_id = required_str(raw.get("call_id"), "computer_call_output.call_id")
        return Screenshot(
            [
                {"type": "input_text", "text": computer_screenshot_label(call_id)},
                {**output, "type": "input_image"},
            ]
        )
    if kind in _TOOL_RECORDS:
        status = raw.get("status")
        if status not in (None, "completed", "failed", "incomplete", "interrupted"):
            raise ResponsesConversionError(
                f"A different protocol cannot continue an active tool {kind!r}"
            )
        if is_client_call(raw) and status in (None, "completed"):
            call_id = required_str(raw.get("call_id"), "tool.call_id")
            if call_id in pending_actions:
                raise ResponsesConversionError("Duplicate pending tool action")
            pending_actions[call_id] = (
                "tool_search_output" if kind == "tool_search_call" else f"{kind}_output"
            )
        return InputMessage("assistant", tool_history_context(raw))
    if kind in (
        "additional_tools",
        "compaction",
        "compaction_trigger",
        "item_reference",
        "configuration_update",
        "mcp_approval_request",
        "mcp_approval_response",
        "program",
        "program_output",
    ):
        raise ResponsesConversionError(
            f"Cannot translate Responses context from {kind}"
        )
    return None
