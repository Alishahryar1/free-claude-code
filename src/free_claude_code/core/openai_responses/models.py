"""Pydantic models for OpenAI Responses-compatible ingress."""

from pydantic import BaseModel, ConfigDict, field_validator

from free_claude_code.core.json_types import JsonObject, JsonValue


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

    @field_validator("max_output_tokens", mode="before")
    @classmethod
    def _reject_boolean_output_limit(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("max_output_tokens must not be a boolean")
        return value
