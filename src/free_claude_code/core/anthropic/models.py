"""Pydantic models for the Anthropic Messages protocol."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from free_claude_code.core.json_types import JsonObject, JsonValue


class _AnthropicBlockBase(BaseModel):
    """Pass through protocol extensions such as ``cache_control``."""

    model_config = ConfigDict(extra="allow")


class ContentBlockText(_AnthropicBlockBase):
    type: Literal["text"]
    text: str


class ContentBlockImage(_AnthropicBlockBase):
    type: Literal["image"]
    source: dict[str, Any]


class ContentBlockDocument(_AnthropicBlockBase):
    """Anthropic document block (e.g. PDF files via the Files API)."""

    type: Literal["document"]
    source: dict[str, Any]


class ContentBlockToolUse(_AnthropicBlockBase):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockToolResult(_AnthropicBlockBase):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[JsonValue] | JsonObject


class ContentBlockThinking(_AnthropicBlockBase):
    type: Literal["thinking"]
    thinking: str
    signature: str | None = None


class ContentBlockRedactedThinking(_AnthropicBlockBase):
    type: Literal["redacted_thinking"]
    data: str


class ContentBlockServerToolUse(_AnthropicBlockBase):
    """Anthropic server-side tool invocation (e.g. ``web_search``, ``web_fetch``)."""

    type: Literal["server_tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockWebSearchToolResult(_AnthropicBlockBase):
    type: Literal["web_search_tool_result"]
    tool_use_id: str
    content: JsonValue


class ContentBlockWebFetchToolResult(_AnthropicBlockBase):
    type: Literal["web_fetch_tool_result"]
    tool_use_id: str
    content: JsonValue


class SystemContent(_AnthropicBlockBase):
    type: Literal["text"]
    text: str


class ExtensionBlock(_AnthropicBlockBase):
    """A tagged block outside the set of known message variants."""

    type: str = Field(min_length=1)

    @field_validator("type")
    @classmethod
    def unknown_type(cls, value: str) -> str:
        if value in {
            "text",
            "image",
            "document",
            "tool_use",
            "tool_result",
            "thinking",
            "redacted_thinking",
            "server_tool_use",
            "web_search_tool_result",
            "web_fetch_tool_result",
        }:
            raise ValueError("Malformed known Messages content block")
        return value


MessageContentBlock = (
    ContentBlockText
    | ContentBlockImage
    | ContentBlockDocument
    | ContentBlockToolUse
    | ContentBlockToolResult
    | ContentBlockThinking
    | ContentBlockRedactedThinking
    | ContentBlockServerToolUse
    | ContentBlockWebSearchToolResult
    | ContentBlockWebFetchToolResult
    | ExtensionBlock
)


class Message(_AnthropicBlockBase):
    role: Literal["user", "assistant"]
    content: str | list[MessageContentBlock]
    reasoning_content: str | None = None


class Tool(_AnthropicBlockBase):
    name: str
    type: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None

    @property
    def allows_direct_call(self) -> bool:
        callers = (self.model_extra or {}).get("allowed_callers")
        return callers is None or (isinstance(callers, list) and "direct" in callers)


class UnnamedTool(_AnthropicBlockBase):
    """Opaque hosted declaration whose protocol supplies no client function name."""

    type: str = Field(min_length=1)

    @model_validator(mode="after")
    def unknown_declaration(self) -> Self:
        if self.type == "custom" or "name" in (self.model_extra or {}):
            raise ValueError("A client tool requires a name")
        return self


class ThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool | None = True
    type: str | None = None
    budget_tokens: int | None = None
    display: Literal["summarized", "omitted", "updates"] | None = None


class MessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    original_model: str | None = Field(default=None, exclude=True)
    resolved_provider_model: str | None = Field(default=None, exclude=True)
    max_tokens: int | None = None
    messages: list[Message]
    system: str | list[SystemContent | ExtensionBlock] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[Tool | UnnamedTool] | None = None
    tool_choice: JsonObject | None = None
    thinking: ThinkingConfig | None = None
    context_management: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    extra_body: dict[str, Any] | None = None
    betas: list[str] | None = Field(default=None, exclude=True)

    @property
    def named_tools(self) -> list[Tool]:
        return [tool for tool in self.tools or () if isinstance(tool, Tool)]

    @property
    def direct_client_tools(self) -> list[Tool]:
        return [
            tool
            for tool in self.named_tools
            if tool.type in (None, "custom") and tool.allows_direct_call
        ]

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _reject_boolean_output_limit(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("max_tokens must not be a boolean")
        return value


class TokenCountRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    original_model: str | None = Field(default=None, exclude=True)
    resolved_provider_model: str | None = Field(default=None, exclude=True)
    messages: list[Message]
    system: str | list[SystemContent | ExtensionBlock] | None = None
    tools: list[Tool | UnnamedTool] | None = None
    thinking: ThinkingConfig | None = None
    tool_choice: JsonObject | None = None
    context_management: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    betas: list[str] | None = Field(default=None, exclude=True)


class TokenCountResponse(BaseModel):
    input_tokens: int


class NativeTokenCountMessage(BaseModel):
    """The outer message shape needed by local estimation, with opaque blocks."""

    model_config = ConfigDict(extra="allow", strict=True)

    role: str
    content: str | list[dict[str, Any]]


class NativeTokenCountRequest(BaseModel):
    """Count native inputs without applying compatibility protocol validation."""

    model_config = ConfigDict(extra="allow", strict=True)

    model: str = Field(min_length=1)
    messages: list[NativeTokenCountMessage]
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: list[
        ContentBlockText
        | ContentBlockToolUse
        | ContentBlockThinking
        | ContentBlockRedactedThinking
        | dict[str, Any]
    ]
    type: Literal["message"] = "message"
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: Usage
