"""Saved custom endpoint definitions, independent of provider resources."""

import json
from typing import Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .api_keys import ApiKeyList, api_key_values, first_api_key

CUSTOM_PROVIDERS_ENV = "FCC_CUSTOM_PROVIDERS"
REASONING_FORMATS = {
    "openai_chat": (
        "provider_default",
        "openai_effort",
        "limited_effort",
        "reasoning_object",
        "thinking",
        "chat_template",
    ),
    "openai_responses": ("provider_default", "native_responses"),
    "anthropic_messages": ("provider_default", "messages_manual", "messages_adaptive"),
}


class CustomProviderDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    provider_id: str = Field(pattern=r"^custom_[0-9a-f]{32}$")
    display_name: str = Field(min_length=1)
    base_url: str
    api_keys: ApiKeyList = Field(default=(), repr=False)
    api_format: Literal["openai_chat", "openai_responses", "anthropic_messages"] = (
        "openai_chat"
    )
    reasoning_format: str = "provider_default"
    reasoning_history_format: Literal[
        "disabled", "reasoning_content", "reasoning", "think_tags"
    ] = "disabled"
    model_ids: tuple[str, ...] = ()

    @field_validator("display_name", mode="before")
    @classmethod
    def trim_name(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("base_url")
    @classmethod
    def validate_base(cls, value: str) -> str:
        try:
            url = TypeAdapter(AnyHttpUrl).validate_python(value.strip())
        except ValueError:
            raise ValueError("Base URL must be an HTTP or HTTPS API address") from None
        if (
            url.username
            or url.password
            or url.query is not None
            or url.fragment is not None
        ):
            raise ValueError(
                "Base URL cannot contain credentials, a query or a fragment"
            )
        return str(url).rstrip("/")

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_key(cls, value: object) -> object:
        if not isinstance(value, dict) or "api_key" not in value:
            return value
        if "api_keys" in value:
            raise ValueError("Supply api_keys only, not both credential forms")
        values = dict(value)
        values["api_keys"] = values.pop("api_key")
        return values

    @field_validator("model_ids")
    @classmethod
    def normalize_models(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(model.strip() for model in value if model.strip()))

    @model_validator(mode="after")
    def validate_formats(self) -> CustomProviderDefinition:
        if self.reasoning_format not in REASONING_FORMATS[self.api_format]:
            raise ValueError("Reasoning format is not supported by this API format")
        if (
            self.api_format != "openai_chat"
            and self.reasoning_history_format != "disabled"
        ):
            raise ValueError(
                "Reasoning history format applies only to Chat Completions"
            )
        return self

    def private_values(self) -> dict:
        return self.model_dump(mode="json") | {
            "api_keys": api_key_values(self.api_keys),
        }

    def same_inventory(self, other: CustomProviderDefinition) -> bool:
        return self.model_dump(
            exclude={"display_name", "api_keys"}
        ) == other.model_dump(exclude={"display_name", "api_keys"}) and first_api_key(
            self.api_keys
        ) == first_api_key(other.api_keys)


def decode_custom_providers(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value) if value.strip() else []
        except ValueError:
            raise ValueError("Custom providers must be a JSON array") from None
    return value


def encode_custom_providers(definitions: tuple[CustomProviderDefinition, ...]) -> str:
    return json.dumps(
        [item.private_values() for item in definitions],
        ensure_ascii=False,
        separators=(",", ":"),
    )
