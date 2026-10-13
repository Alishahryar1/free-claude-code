"""Opaque native Messages requests and lifecycle-only SSE relay."""

import json
from copy import deepcopy
from dataclasses import dataclass

from free_claude_code.core.history_replay import (
    ReplayOrigin,
    decode_replay,
    is_replay,
)
from free_claude_code.core.json_types import JsonObject

from .native import NativeMessagesError, validate_messages_json


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

    def feed(self, kind: str, payload: JsonObject) -> str | None:
        if self.completed:
            raise NativeMessagesError("Messages event arrived after message_stop.")
        validate_messages_json(payload)
        if payload.get("type") != kind:
            raise NativeMessagesError("Messages event has an invalid payload type.")
        if kind == "ping" and not self.started:
            return None
        body = dict(payload)
        if kind == "message_start":
            message = body.get("message")
            if self.started or not isinstance(message, dict):
                raise NativeMessagesError("Invalid native message_start.")
            self.started = True
            body["message"] = {**message, "model": self.public_model}
        elif kind != "ping" and not self.started:
            raise NativeMessagesError(
                "Native Messages event arrived before message_start."
            )
        if kind == "message_stop":
            self.completed = True
        return f"event: {kind}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"
