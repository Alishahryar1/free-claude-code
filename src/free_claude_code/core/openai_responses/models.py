"""Pydantic models for OpenAI Responses-compatible ingress."""

from collections.abc import Mapping
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator

from free_claude_code.core.json_types import JsonObject, JsonValue

from .errors import ResponsesConversionError
from .tool_search import ClientSearchHistory, resolve_client_search_history
from .tools import ResponsesToolIdentity, definition_identity, optional_str


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

    def require_inline_context(self) -> None:
        """Foreign providers can only use context included in the request."""
        extra: Mapping[str, JsonValue] = self.model_extra or {}
        for field, value in (
            ("previous_response_id", self.previous_response_id),
            ("conversation", extra.get("conversation")),
            ("prompt", extra.get("prompt")),
        ):
            if value is not None:
                raise ResponsesConversionError(
                    f"Cannot translate Responses context from {field}"
                )

    def select_direct_tools(self, history: ClientSearchHistory | None = None) -> Self:
        """Apply the caller's restrictions to the declarations available on this turn."""
        choice = self.tool_choice
        allowed: list[tuple[JsonValue, JsonValue, JsonValue]] | None = None
        if isinstance(choice, dict) and choice.get("type") == "allowed_tools":
            mode, entries = choice.get("mode"), choice.get("tools")
            if mode not in ("auto", "required") or not isinstance(entries, list):
                raise ResponsesConversionError(
                    "allowed_tools requires a mode and tool list"
                )
            allowed = []
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ResponsesConversionError("Allowed tools must be objects")
                allowed.append(
                    (entry.get("type"), entry.get("name"), entry.get("namespace"))
                )
            choice = mode

        loaded: set[ResponsesToolIdentity] | None = None

        def permitted(tool: JsonObject, namespace: JsonValue = None) -> bool:
            nonlocal loaded
            kind = tool.get("type")
            source = tool
            if kind in ("function", "custom"):
                nested = tool.get(str(kind))
                if isinstance(nested, dict):
                    source = nested
                format_ = source.get("format")
                if (
                    kind == "custom"
                    and isinstance(format_, dict)
                    and format_.get("type") not in (None, "text")
                ):
                    return False
                callers = tool.get("allowed_callers", source.get("allowed_callers"))
                if callers is not None and (
                    not isinstance(callers, list) or "direct" not in callers
                ):
                    return False
                if tool.get("defer_loading", source.get("defer_loading")):
                    if loaded is None:
                        loaded = (
                            history or resolve_client_search_history(self.input)
                        ).loaded_tools()
                    if definition_identity(tool, optional_str(namespace)) not in loaded:
                        return False
            identity = (
                kind,
                source.get("name"),
                namespace or source.get("namespace") or tool.get("namespace"),
            )
            return allowed is None or identity in allowed

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
        return self.model_copy(update={"tools": retained, "tool_choice": choice})

    @field_validator("max_output_tokens", mode="before")
    @classmethod
    def _reject_boolean_output_limit(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("max_output_tokens must not be a boolean")
        return value
