"""Immutable discovery records for Anthropic protocol conversion."""

from dataclasses import dataclass
from typing import Literal

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.anthropic_messages.discovery import messages_model_info
from free_claude_code.providers.anthropic_messages.request_policy import (
    MessagesModelCapabilities,
)


@dataclass(frozen=True, slots=True)
class AnthropicModelRecord:
    info: ProviderModelInfo
    messages: MessagesModelCapabilities


def model_record(item: JsonObject) -> AnthropicModelRecord:
    info = messages_model_info(item, "anthropic")
    capabilities = item.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    thinking = capabilities.get("thinking")
    thinking = thinking if isinstance(thinking, dict) else {}
    types = thinking.get("types")
    types = types if isinstance(types, dict) else {}
    adaptive = types.get("adaptive")
    manual = types.get("enabled")
    adaptive_supported = (
        adaptive.get("supported") if isinstance(adaptive, dict) else None
    )
    manual_supported = manual.get("supported") if isinstance(manual, dict) else None
    mode: Literal["unsupported", "optional", "required"] | None = None
    if adaptive_supported is True:
        mode = "required" if manual_supported is False else "optional"
    elif adaptive_supported is False:
        mode = "unsupported"
    effort = capabilities.get("effort")
    effort = effort if isinstance(effort, dict) else {}
    support = effort.get("supported")
    advertised = {
        name: value.get("supported")
        for name, value in effort.items()
        if name != "supported" and isinstance(value, dict)
    }
    values = (
        tuple(name for name, supported in advertised.items() if supported is True)
        if advertised
        else None
    )
    image = capabilities.get("image_input")
    image_support = image.get("supported") if isinstance(image, dict) else None
    return AnthropicModelRecord(
        info,
        MessagesModelCapabilities(
            max_output_tokens=info.max_output_tokens,
            adaptive_thinking=mode,
            supports_output_effort=support if isinstance(support, bool) else None,
            supported_efforts=values,
            supports_vision=image_support if isinstance(image_support, bool) else None,
        ),
    )
