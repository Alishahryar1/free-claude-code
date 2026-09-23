"""Native Messages wire preparation, without a cross-protocol intermediate."""

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from free_claude_code.core.json_types import JsonObject, JsonValue

from .models import MessagesRequest
from .request_serialization import dump_messages_request

_BETA_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_NATIVE_EXTRA_FIELDS = frozenset({"cache_control", "service_tier"})
_RESERVED_EXTRA_FIELDS = frozenset(
    {
        "model",
        "messages",
        "system",
        "stream",
        "max_tokens",
        "thinking",
        "output_config",
        "tools",
        "tool_choice",
        "betas",
        "extra_body",
        "original_model",
        "resolved_provider_model",
        "api_key",
        "base_url",
        "url",
        "headers",
        "extra_headers",
        "authorization",
        "anthropic-version",
    }
)


class NativeMessagesError(ValueError):
    """A request or event cannot preserve the native Messages contract."""


@dataclass(frozen=True, slots=True)
class NativeMessagesOptions:
    """Provider-resolved wire controls; no model-name inference belongs here."""

    model: str
    max_tokens: int
    thinking: JsonObject | None = None
    output_effort: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise NativeMessagesError("Messages model must not be empty.")
        if (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens <= 0
        ):
            raise NativeMessagesError("Messages max_tokens must be a positive integer.")


@dataclass(frozen=True, slots=True)
class PreparedMessagesRequest:
    body: JsonObject
    betas: tuple[str, ...] = ()


def validate_messages_json(body: JsonObject) -> None:
    """Reject values the HTTP JSON encoder cannot send before opening an attempt."""

    try:
        json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise NativeMessagesError(
            "Messages request must be finite JSON encodable as UTF-8."
        ) from exc


def apply_messages_options(body: JsonObject, options: NativeMessagesOptions) -> None:
    """Apply resolved controls to a new body while preserving format options."""

    body["model"] = options.model
    body["max_tokens"] = options.max_tokens
    body["stream"] = True
    body.pop("thinking", None)
    if options.thinking is not None:
        body["thinking"] = deepcopy(options.thinking)
    raw_output = body.pop("output_config", None)
    output: JsonObject = {}
    if raw_output is not None:
        if not isinstance(raw_output, Mapping):
            raise NativeMessagesError("Messages output_config must be an object.")
        output = dict(raw_output)
        output.pop("effort", None)
    if options.output_effort is not None:
        output["effort"] = options.output_effort
    if output:
        body["output_config"] = output
    choice = body.get("tool_choice")
    if (
        options.thinking is not None
        and options.thinking.get("type") == "enabled"
        and isinstance(choice, Mapping)
        and choice.get("type") in {"any", "tool"}
    ):
        raise NativeMessagesError(
            "Manual Messages thinking cannot force a tool choice."
        )


def build_native_messages_request(
    request: MessagesRequest, *, options: NativeMessagesOptions
) -> PreparedMessagesRequest:
    """Prepare native input, preserving supported nested protocol extensions."""

    if not request.messages:
        raise NativeMessagesError("Messages input must not be empty.")
    body = cast(JsonObject, dump_messages_request(request))
    raw_request = cast(JsonObject, request.model_dump(mode="json", exclude_none=True))
    for name in request.model_extra or ():
        if name not in raw_request:
            continue
        if name not in _NATIVE_EXTRA_FIELDS:
            raise NativeMessagesError(
                f"Unsupported native Messages request field: {name!r}."
            )
        body[name] = deepcopy(raw_request[name])
    system: list[JsonValue] = []
    raw_system = body.pop("system", None)
    if isinstance(raw_system, str):
        system.append({"type": "text", "text": raw_system})
    elif isinstance(raw_system, list):
        system.extend(raw_system)
    messages: list[JsonValue] = []
    for message in request.messages:
        if message.reasoning_content is not None:
            raise NativeMessagesError(
                "Native Messages history requires signed thinking blocks; "
                "reasoning_content cannot be replayed losslessly."
            )
        raw = cast(JsonObject, message.model_dump(mode="json", exclude_none=True))
        if message.role == "system":
            if messages:
                raise NativeMessagesError(
                    "Native Messages supports only leading system messages."
                )
            content = raw["content"]
            if isinstance(content, str):
                system.append({"type": "text", "text": content})
            elif isinstance(content, list):
                if any(
                    not isinstance(block, Mapping) or block.get("type") != "text"
                    for block in content
                ):
                    raise NativeMessagesError("System content must contain only text.")
                system.extend(content)
            continue
        messages.append(raw)
    if not messages:
        raise NativeMessagesError("Messages input must contain a conversational turn.")
    body["messages"] = messages
    if system:
        body["system"] = system
    extra = body.pop("extra_body", None)
    if extra is not None:
        if not isinstance(extra, Mapping):
            raise NativeMessagesError("Messages extra_body must be an object.")
        for name, value in extra.items():
            if name.lower() in _RESERVED_EXTRA_FIELDS or name in body:
                raise NativeMessagesError(
                    f"Messages extra_body cannot override {name!r}."
                )
            body[name] = deepcopy(value)
    apply_messages_options(body, options)
    betas = tuple(request.betas or ())
    if any(_BETA_NAME.fullmatch(beta) is None for beta in betas):
        raise NativeMessagesError("Messages beta names must be valid header tokens.")
    validate_messages_json(body)
    return PreparedMessagesRequest(body=body, betas=tuple(dict.fromkeys(betas)))
