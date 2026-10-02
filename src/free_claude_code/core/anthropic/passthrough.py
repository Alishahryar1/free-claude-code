"""Opaque native Messages requests and history restoration."""

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
