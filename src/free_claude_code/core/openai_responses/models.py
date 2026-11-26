"""Pydantic models for OpenAI Responses-compatible ingress."""

from collections.abc import Mapping, Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator

from free_claude_code.core.json_types import JsonObject, JsonValue

from .errors import ResponsesConversionError


def latest_user_input(items: Sequence[JsonValue]) -> JsonValue:
    for item in reversed(items):
        if isinstance(item, str):
            return item
        if isinstance(item, Mapping):
            kind = item.get("type", "message")
            if (kind == "message" and item.get("role", "user") == "user") or kind in (
                "input_text",
                "text",
                "input_image",
            ):
                return item
    return None


def is_hosted_history(kind: JsonValue) -> bool:
    return kind in (
        "web_search_call",
        "file_search_call",
        "image_generation_call",
        "code_interpreter_call",
        "mcp_call",
        "mcp_list_tools",
        "computer_call",
        "shell_call",
        "local_shell_call",
        "apply_patch_call",
        "tool_search_call",
    )


class OpenAIResponsesRequest(BaseModel):
    """Permissive subset of the OpenAI Responses API request shape."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: JsonValue = None
    instructions: str | None = None
    tools: list[JsonObject] | None = None
    tool_choice: JsonValue = None
    parallel_tool_calls: bool | None = None
    stream: bool | None = True
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    metadata: JsonObject | None = None
    reasoning: JsonObject | None = None
    previous_response_id: str | None = None
    store: bool | None = None

    def restrict_allowed_tools(self) -> Self:
        choice = self.tool_choice
        if not isinstance(choice, dict) or choice.get("type") != "allowed_tools":
            return self
        mode, allowed = choice.get("mode"), choice.get("tools")
        if mode not in ("auto", "required") or not isinstance(allowed, list):
            raise ResponsesConversionError(
                "allowed_tools requires a mode and tool list"
            )
        if not all(isinstance(tool, dict) for tool in allowed):
            raise ResponsesConversionError("Allowed tools must be objects")

        allowed_entries = allowed

        def permitted(tool: JsonObject, namespace: JsonValue = None) -> bool:
            return any(
                isinstance(entry, dict)
                and (entry.get("type"), entry.get("name"), entry.get("namespace"))
                == (
                    tool.get("type"),
                    tool.get("name"),
                    tool.get("namespace", namespace),
                )
                for entry in allowed_entries
            )

        retained: list[JsonObject] = []
        for tool in self.tools or ():
            if tool.get("type") == "namespace" and isinstance(
                children := tool.get("tools"), list
            ):
                selected = [
                    child
                    for child in children
                    if isinstance(child, dict) and permitted(child, tool.get("name"))
                ]
                if selected:
                    retained.append({**tool, "tools": selected})
            elif permitted(tool):
                retained.append(tool)
        return self.model_copy(update={"tools": retained, "tool_choice": mode})

    @field_validator("max_output_tokens", mode="before")
    @classmethod
    def _reject_boolean_output_limit(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("max_output_tokens must not be a boolean")
        return value
