"""NVIDIA NIM settings (fixed values, no env config)."""

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


class NimSettings(BaseModel):
    """Fixed NVIDIA NIM settings (not configurable via env)."""

    temperature: float = Field(
        1.0, ge=0.0, le=2.0, description="Sampling temperature, must be >=0 and <=2."
    )
    top_p: float = Field(
        0.95,
        ge=0.95,
        le=0.95,
        description="FCC's fixed NVIDIA NIM nucleus sampling probability.",
    )
    top_k: int = -1
    max_tokens: int = Field(
        ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ge=1,
        description="Maximum number of tokens in output.",
    )
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    min_p: float = Field(
        0.0, ge=0.0, le=1.0, description="Minimum probability threshold [0,1]."
    )
    repetition_penalty: float = Field(
        1.0, ge=0.0, description="Penalty for repeated tokens. Must be >=0."
    )
    seed: int | None = None
    stop: str | None = None
    parallel_tool_calls: bool = True
    ignore_eos: bool = False
    min_tokens: int = Field(0, ge=0, description="Minimum tokens in the response.")
    chat_template: str | None = None
    request_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k(cls, v, info: ValidationInfo):
        if v is None or v == "":
            return -1
        int_v = int(v)
        if int_v < -1:
            raise ValueError(f"{info.field_name} must be -1 or >= 0")
        return int_v

    @field_validator(
        "temperature",
        "top_p",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        mode="before",
    )
    @classmethod
    def validate_float_fields(cls, v, info: ValidationInfo):
        field_defaults = {
            "temperature": 1.0,
            "top_p": 0.95,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
        if v is None or v == "":
            key = info.field_name or "temperature"
            return field_defaults.get(key, 1.0)
        try:
            val = float(v)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"{info.field_name} must be a float. Got {type(v).__name__}."
            ) from err
        return val

    @field_validator("max_tokens", "min_tokens", mode="before")
    @classmethod
    def validate_int_fields(cls, v, info: ValidationInfo):
        field_defaults = {
            "max_tokens": ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            "min_tokens": 0,
        }
        if v is None or v == "":
            key = info.field_name or "max_tokens"
            return field_defaults.get(key, ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS)
        try:
            val = int(v)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"{info.field_name} must be an int. Got {type(v).__name__}."
            ) from err
        return val

    @field_validator("seed", mode="before")
    @classmethod
    def parse_optional_int(cls, v, info: ValidationInfo):
        if v == "" or v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"{info.field_name} must be an int or empty/None."
            ) from err

    @field_validator("stop", "chat_template", "request_id", mode="before")
    @classmethod
    def parse_optional_str(cls, v, info: ValidationInfo):
        if v == "":
            return None
        if v is not None and not isinstance(v, str):
            return str(v)
        return v
