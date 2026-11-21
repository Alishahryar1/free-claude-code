"""Opaque native Messages requests and lifecycle-only SSE relay."""

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from free_claude_code.core.history_replay import (
    ReplayOrigin,
    decode_replay,
    is_replay,
    protected_reasoning,
    resolve_messages_replay,
)
from free_claude_code.core.json_types import JsonObject, JsonValue

from .native import NativeMessagesError, validate_messages_json
from .native_stream import MessagesBlockIndexMap


@dataclass(frozen=True, slots=True)
class NativeMessagesRequest:
    body: JsonObject

    def __post_init__(self) -> None:
        validate_messages_json(self.body)
        if not isinstance(self.body.get("model"), str) or not self.model.strip():
            raise NativeMessagesError("Messages model must not be empty.")
        messages = self.body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise NativeMessagesError("Messages input must be a nonempty array.")
        if "stream" in self.body and not isinstance(self.body["stream"], bool):
            raise NativeMessagesError("Messages stream must be a boolean.")
        object.__setattr__(self, "body", deepcopy(self.body))

    @property
    def model(self) -> str:
        value = self.body["model"]
        assert isinstance(value, str)
        return value

    @property
    def stream(self) -> bool:
        return self.body.get("stream", False) is True

    def with_model(self, model: str) -> NativeMessagesRequest:
        return NativeMessagesRequest({**self.body, "model": model})


def restore_native_history(body: JsonObject, origin: ReplayOrigin) -> JsonObject:
    """Unwrap only FCC-owned state, without changing native history."""
    result = deepcopy(body)
    messages = result.get("messages")
    if not isinstance(messages, list):
        return result
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        content = resolve_messages_replay(content)
        message["content"] = content
        for index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            field = {"thinking": "signature", "redacted_thinking": "data"}.get(
                str(block.get("type"))
            )
            value = block.get(field) if field else None
            if not isinstance(value, str) or not is_replay(value):
                continue
            record = decode_replay(value)
            if not protected_reasoning(record.native):
                # Keep readable identity until the shared destination projection.
                continue
            if not origin.accepts(record.origin):
                raise NativeMessagesError(
                    "Native Messages cannot replay history from a different provider or model."
                )
            content[index] = deepcopy(record.native)
    return result


class NativeMessagesPassthrough:
    """Inspect the envelope and preserve provider-owned event payloads."""

    def __init__(self, public_model: str) -> None:
        self.public_model = public_model
        self.started = False
        self.completed = False
        self._indexes = MessagesBlockIndexMap()

    def feed(self, event_type: str, payload: Mapping[str, JsonValue]) -> str | None:
        if self.completed:
            raise NativeMessagesError("Messages event arrived after message_stop.")
        validate_messages_json(dict(payload))
        if payload.get("type") != event_type:
            raise NativeMessagesError("Messages event has an invalid payload type.")
        if event_type == "ping" and not self.started:
            return None
        body = self._indexes.project(event_type, payload)
        if event_type == "message_start":
            message = body.get("message")
            if self.started or not isinstance(message, dict):
                raise NativeMessagesError("Invalid native message_start.")
            self.started = True
            body["message"] = {**message, "model": self.public_model}
        elif event_type != "ping" and not self.started:
            raise NativeMessagesError(
                "Native Messages event arrived before message_start."
            )
        if event_type == "message_stop":
            self.completed = True
        return f"event: {event_type}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"
