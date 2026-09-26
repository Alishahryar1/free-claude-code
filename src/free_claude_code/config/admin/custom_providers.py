"""Keyed custom-provider edits merged into a fresh configuration snapshot."""

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from free_claude_code.config.custom_providers import (
    CUSTOM_PROVIDERS_ENV,
    CustomProviderDefinition,
    encode_custom_providers,
)
from free_claude_code.config.loader import ConfigSource, ManagedConfigSnapshot
from free_claude_code.core.json_types import JsonObject


class CustomProviderMutation(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    action: Literal["create", "update", "delete"]
    provider_id: str | None = None
    values: JsonObject = Field(default_factory=dict, repr=False)


def merge_custom_provider(
    target: dict[str, str],
    snapshot: ManagedConfigSnapshot,
    mutation: CustomProviderMutation,
) -> str:
    if snapshot.sources["custom_providers"] == ConfigSource.PROCESS:
        raise ValueError("Custom providers are managed by the process environment")
    definitions = {
        item.provider_id: item for item in snapshot.settings.custom_providers
    }
    provider_id = mutation.provider_id
    if mutation.action == "create":
        if provider_id is not None:
            raise ValueError("Provider IDs are assigned by FCC")
        provider_id = "custom_" + uuid4().hex
        values = {}
    else:
        if provider_id is None:
            raise ValueError("A provider ID is required")
        if mutation.action == "delete":
            if mutation.values:
                raise ValueError("Removal does not accept field edits")
            references = [
                key
                for key in (
                    "MODEL",
                    "MODEL_FABLE",
                    "MODEL_OPUS",
                    "MODEL_SONNET",
                    "MODEL_HAIKU",
                    "MODEL_FALLBACKS",
                )
                if any(
                    ref.strip().startswith(provider_id + "/")
                    for ref in snapshot.process.get(key, target.get(key, "")).split(",")
                )
            ]
            if references:
                raise ValueError(
                    "Change these Model Config settings before removing the provider: "
                    + ", ".join(references)
                )
            definitions.pop(provider_id, None)
            target[CUSTOM_PROVIDERS_ENV] = encode_custom_providers(
                tuple(definitions.values())
            )
            return provider_id
        if provider_id not in definitions:
            raise ValueError("This custom provider no longer exists")
        values = definitions[provider_id].private_values()
    if "provider_id" in mutation.values:
        raise ValueError("Provider IDs cannot be edited")
    changes = dict(mutation.values)
    if changes.get("api_key") == "********" or (
        isinstance(changes.get("api_key"), str) and not str(changes["api_key"]).strip()
    ):
        changes.pop("api_key")
    try:
        definition = CustomProviderDefinition.model_validate(
            values | changes | {"provider_id": provider_id}
        )
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=False)
        raise ValueError(
            "; ".join(
                f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                for error in errors
            )
        ) from None
    if any(
        item.provider_id != provider_id
        and item.display_name.casefold() == definition.display_name.casefold()
        for item in definitions.values()
    ):
        raise ValueError("A custom provider with this name already exists")
    definitions[provider_id] = definition
    target[CUSTOM_PROVIDERS_ENV] = encode_custom_providers(tuple(definitions.values()))
    return provider_id
