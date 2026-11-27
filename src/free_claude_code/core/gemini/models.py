"""The supported GenerateContent contract, independent of any client harness."""

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from ..anthropic.image_sources import portable_anthropic_image_url
from ..json_types import JsonObject, JsonValue
from ..reasoning import ReasoningEffort, ReasoningPolicy


class GeminiModel(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        alias_generator=to_camel,
        strict=True,
        allow_inf_nan=False,
    )

    def has_extension(self) -> bool:
        extras: Mapping[str, JsonValue] = self.model_extra or {}
        return any(value is not None for value in extras.values())

    @model_validator(mode="before")
    @classmethod
    def protocol_fields(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        declared = set(cls.model_fields)
        for name, field in cls.model_fields.items():
            alias = field.alias or name
            declared.add(alias)
            if alias != name and name in value and alias in value:
                raise ValueError(f"Duplicate protocol field: {alias}")
        # ProtoJSON null means absent for protocol fields. Arbitrary JSON objects
        # are not GeminiModel instances and retain their nulls and original keys.
        return {
            key: item
            for key, item in value.items()
            if item is not None or key not in declared
        }


class InlineData(GeminiModel):
    mime_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str = Field(min_length=1)
    display_name: str | None = None

    @model_validator(mode="after")
    def valid_image(self) -> Self:
        portable_anthropic_image_url(self.source())
        return self

    def source(self) -> JsonObject:
        return {"type": "base64", "media_type": self.mime_type, "data": self.data}

    def url(self) -> str:
        return portable_anthropic_image_url(self.source())


class FunctionCall(GeminiModel):
    id: str | None = Field(default=None, min_length=1)
    name: str = Field(min_length=1)
    args: JsonObject = Field(default_factory=dict)


class MediaPart(GeminiModel):
    inline_data: InlineData | None = None
    file_data: JsonObject | None = None


class FunctionResponsePart(MediaPart):
    @model_validator(mode="after")
    def has_media(self) -> Self:
        if self.inline_data is not None and self.file_data is not None:
            raise ValueError(
                "A function response part must have exactly one media source"
            )
        if (
            self.inline_data is None
            and self.file_data is None
            and not self.has_extension()
        ):
            raise ValueError("A function response part requires media")
        return self


class FunctionResponse(GeminiModel):
    id: str | None = Field(default=None, min_length=1)
    name: str = Field(min_length=1)
    response: JsonObject
    parts: list[FunctionResponsePart] = Field(default_factory=list)


class Part(MediaPart):
    text: str | None = None
    executable_code: JsonObject | None = None
    code_execution_result: JsonObject | None = None
    tool_call: JsonObject | None = None
    tool_response: JsonObject | None = None
    function_call: FunctionCall | None = None
    function_response: FunctionResponse | None = None
    thought: bool = False
    thought_signature: str | None = Field(default=None, min_length=1)

    @property
    def data_variant_count(self) -> int:
        return sum(
            value is not None
            for value in (
                self.text,
                self.inline_data,
                self.file_data,
                self.executable_code,
                self.code_execution_result,
                self.tool_call,
                self.tool_response,
                self.function_call,
                self.function_response,
            )
        )

    @model_validator(mode="after")
    def one_content_variant(self) -> Self:
        count = self.data_variant_count
        if count > 1 or (
            count == 0 and self.thought_signature is None and not self.has_extension()
        ):
            raise ValueError("A part requires one data variant or a thoughtSignature")
        if self.thought and self.text is None and self.thought_signature is None:
            raise ValueError("A thought part requires text or a signature")
        return self


class Content(GeminiModel):
    role: Literal["user", "model"] | None = None
    parts: list[Part] = Field(min_length=1)

    @property
    def effective_role(self) -> Literal["model", "user"]:
        if self.role is not None:
            return self.role
        return (
            "model"
            if any(part.function_call is not None for part in self.parts)
            else "user"
        )


class FunctionDeclaration(GeminiModel):
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: JsonObject | None = None
    parameters_json_schema: JsonObject | None = None

    @model_validator(mode="after")
    def one_schema(self) -> Self:
        if self.parameters is not None and self.parameters_json_schema is not None:
            raise ValueError(
                "parameters and parametersJsonSchema are mutually exclusive"
            )
        return self


class Tool(GeminiModel):
    function_declarations: list[FunctionDeclaration] = Field(
        default_factory=list, min_length=1
    )

    @model_validator(mode="after")
    def has_declaration(self) -> Self:
        if not self.function_declarations and not self.has_extension():
            raise ValueError("A tool requires a declaration")
        return self


class FunctionCallingConfig(GeminiModel):
    mode: Literal["MODE_UNSPECIFIED", "AUTO", "ANY", "NONE"] = "AUTO"
    allowed_function_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def allowed_mode(self) -> Self:
        if self.allowed_function_names and self.mode != "ANY":
            raise ValueError("allowedFunctionNames requires ANY mode")
        return self


class ToolConfig(GeminiModel):
    function_calling_config: FunctionCallingConfig = Field(
        default_factory=FunctionCallingConfig
    )


class ThinkingConfig(GeminiModel):
    include_thoughts: bool = False
    thinking_budget: int | None = Field(default=None, ge=-1)
    thinking_level: (
        Literal["THINKING_LEVEL_UNSPECIFIED", "MINIMAL", "LOW", "MEDIUM", "HIGH"] | None
    ) = None

    @model_validator(mode="after")
    def one_control(self) -> Self:
        if self.thinking_budget is not None and self.thinking_level not in (
            None,
            "THINKING_LEVEL_UNSPECIFIED",
        ):
            raise ValueError("Specify thinkingBudget or thinkingLevel, not both")
        return self

    def policy(self) -> ReasoningPolicy:
        if self.thinking_budget == 0:
            return ReasoningPolicy.off()
        if self.thinking_budget is not None:
            return ReasoningPolicy.on(
                budget_tokens=self.thinking_budget if self.thinking_budget > 0 else None
            )
        if self.thinking_level not in (None, "THINKING_LEVEL_UNSPECIFIED"):
            return ReasoningPolicy.on(
                effort=ReasoningEffort(self.thinking_level.lower())
            )
        return ReasoningPolicy.provider_default()


class TextResponseFormat(GeminiModel):
    mime_type: Literal["TEXT_PLAIN", "APPLICATION_JSON"] = "TEXT_PLAIN"
    schema_: JsonObject | None = Field(default=None, alias="schema")


class ResponseFormat(GeminiModel):
    text: TextResponseFormat | None = None

    @model_validator(mode="after")
    def has_format(self) -> Self:
        if self.text is None and not self.has_extension():
            raise ValueError("responseFormat requires a format")
        return self


class GenerationConfig(GeminiModel):
    candidate_count: Literal[1] = 1
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, gt=0)
    seed: int | None = None
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    stop_sequences: list[str] = Field(default_factory=list)
    thinking_config: ThinkingConfig = Field(default_factory=ThinkingConfig)
    response_mime_type: Literal["text/plain", "application/json"] | None = None
    response_schema: JsonObject | None = None
    response_json_schema: JsonObject | None = None
    response_format: ResponseFormat | None = None
    response_modalities: list[Literal["TEXT"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def output_contract(self) -> Self:
        if self.response_schema is not None and self.response_json_schema is not None:
            raise ValueError("Specify responseSchema or responseJsonSchema, not both")
        if self.response_format is not None and any(
            value is not None
            for value in (
                self.response_mime_type,
                self.response_schema,
                self.response_json_schema,
            )
        ):
            raise ValueError(
                "responseFormat cannot be combined with legacy format fields"
            )
        if (
            self.response_schema is not None or self.response_json_schema is not None
        ) and self.response_mime_type != "application/json":
            raise ValueError("An output schema requires application/json")
        if self.response_format is not None:
            text = self.response_format.text
            if (
                text is not None
                and text.schema_ is not None
                and text.mime_type != "APPLICATION_JSON"
            ):
                raise ValueError("An output schema requires APPLICATION_JSON")
        return self

    @property
    def structured(self) -> bool:
        return self.response_mime_type == "application/json" or (
            self.response_format is not None
            and self.response_format.text is not None
            and self.response_format.text.mime_type == "APPLICATION_JSON"
        )


class GeminiRequest(GeminiModel):
    # Supplied by routing, never taken from the HTTP request body.
    model: str = Field(min_length=1)
    cached_content: str | None = Field(default=None, min_length=1)
    continuation_token: str | None = Field(default=None, min_length=1)
    safety_settings: list[JsonObject] | None = None
    contents: list[Content] = Field(min_length=1)
    system_instruction: Content | None = None
    tools: list[Tool] = Field(default_factory=list)
    tool_config: ToolConfig = Field(default_factory=ToolConfig)
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if not self.model.strip():
            raise ValueError("Model must not be blank")
        if self.system_instruction is not None and any(
            (part.text is None and part.data_variant_count)
            or part.thought
            or part.thought_signature is not None
            for part in self.system_instruction.parts
        ):
            raise ValueError("systemInstruction supports ordinary text only")
        names = [
            declaration.name
            for tool in self.tools
            for declaration in tool.function_declarations
        ]
        if len(set(names)) != len(names):
            raise ValueError("Function declaration names must be unique")
        config = self.tool_config.function_calling_config
        if config.allowed_function_names and not set(names).intersection(
            config.allowed_function_names
        ):
            raise ValueError("allowedFunctionNames must include a declared function")
        if config.mode == "ANY" and not names:
            raise ValueError("ANY function calling requires declared functions")
        return self

    @property
    def include_thoughts(self) -> bool:
        return self.generation_config.thinking_config.include_thoughts

    def reasoning_policy(self) -> ReasoningPolicy:
        return self.generation_config.thinking_config.policy()
