"""Native Messages event validation and identity-preserving relay."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import cast

from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.json_types import JsonObject, JsonValue

from .native import NativeMessagesError

_STOP_REASONS = {
    "end_turn",
    "stop_sequence",
    "tool_use",
    "max_tokens",
    "refusal",
    "pause_turn",
    "model_context_window_exceeded",
}


@dataclass(slots=True)
class _Block:
    body: JsonObject
    parts: list[str] = field(default_factory=list)
    fragments: dict[str, list[str]] = field(default_factory=dict)


class NativeMessagesStreamState:
    """Validate one upstream lifecycle without rewriting protocol identities."""

    def __init__(self) -> None:
        self.started = False
        self.completed = False
        self.stop_reason: str | None = None
        self._blocks: dict[int, _Block] = {}
        self._seen: set[int] = set()
        self._tool_ids: set[str] = set()

    def accept(
        self, event_type: str, payload: Mapping[str, JsonValue]
    ) -> JsonObject | None:
        if self.completed:
            raise NativeMessagesError("Messages event arrived after message_stop.")
        try:
            json.dumps(dict(payload), ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise NativeMessagesError(
                "Native Messages event is not representable JSON."
            ) from exc
        if payload.get("type") != event_type:
            raise NativeMessagesError("Messages event type disagrees with its payload.")
        if event_type == "ping":
            return None
        if event_type == "message_start":
            if self.started:
                raise NativeMessagesError("Duplicate message_start.")
            message = payload.get("message")
            if (
                not isinstance(message, Mapping)
                or not isinstance(message.get("id"), str)
                or not message["id"]
                or message.get("role") != "assistant"
                or message.get("content", []) != []
            ):
                raise NativeMessagesError("Invalid native message_start.")
            self.started = True
            return None
        if not self.started:
            raise NativeMessagesError("Messages event arrived before message_start.")
        if event_type == "message_delta":
            delta = payload.get("delta")
            if not isinstance(delta, Mapping):
                raise NativeMessagesError("Native message_delta requires an object.")
            reason = delta.get("stop_reason")
            if reason is not None:
                if self._blocks:
                    raise NativeMessagesError(
                        "Stop reason arrived with open content blocks."
                    )
                if not isinstance(reason, str) or reason not in _STOP_REASONS:
                    raise NativeMessagesError(
                        "Unsupported native Messages stop reason."
                    )
                if self.stop_reason is not None:
                    raise NativeMessagesError("Duplicate native Messages stop reason.")
                self.stop_reason = reason
            return None
        if event_type == "message_stop":
            if self._blocks or self.stop_reason is None:
                raise NativeMessagesError(
                    "message_stop arrived before content or stop reason completed."
                )
            self.completed = True
            return None
        if event_type not in {
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        }:
            raise NativeMessagesError(
                f"Unsupported native Messages event: {event_type!r}."
            )
        if self.stop_reason is not None:
            raise NativeMessagesError("Content arrived after the message stop reason.")
        index = payload.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise NativeMessagesError(
                "Messages block index must be a nonnegative integer."
            )
        if event_type == "content_block_start":
            if index in self._seen:
                raise NativeMessagesError("Duplicate Messages content block index.")
            body = payload.get("content_block")
            if not isinstance(body, Mapping) or not isinstance(body.get("type"), str):
                raise NativeMessagesError("Invalid native content block.")
            kind = body["type"]
            if kind == "text" and not isinstance(body.get("text"), str):
                raise NativeMessagesError("Native text block requires text.")
            if kind == "thinking" and not isinstance(body.get("thinking"), str):
                raise NativeMessagesError(
                    "Native thinking block requires thinking text."
                )
            if (
                kind == "thinking"
                and "signature" in body
                and not isinstance(body["signature"], str)
            ):
                raise NativeMessagesError("Native thinking signature must be a string.")
            if kind == "redacted_thinking" and (
                not isinstance(body.get("data"), str) or not body["data"]
            ):
                raise NativeMessagesError("Redacted native thinking requires its data.")
            if kind == "tool_use":
                tool_id = body.get("id")
                if (
                    not isinstance(tool_id, str)
                    or not tool_id
                    or tool_id in self._tool_ids
                    or not isinstance(body.get("name"), str)
                    or not body["name"]
                    or not isinstance(body.get("input"), Mapping)
                ):
                    raise NativeMessagesError(
                        "Invalid or duplicate native tool identity/input."
                    )
                self._tool_ids.add(tool_id)
            self._blocks[index] = _Block(dict(body))
            self._seen.add(index)
            return None
        block = self._blocks.get(index)
        if block is None:
            raise NativeMessagesError(
                "Messages content event refers to a closed or unknown block."
            )
        if event_type == "content_block_delta":
            delta = payload.get("delta")
            if not isinstance(delta, Mapping):
                raise NativeMessagesError("Native content delta must be an object.")
            kind = delta.get("type")
            target = (
                {
                    "text_delta": ("text", "text"),
                    "thinking_delta": ("thinking", "thinking"),
                    "signature_delta": ("thinking", "signature"),
                    "input_json_delta": ("tool_use", "partial_json"),
                }.get(kind)
                if isinstance(kind, str)
                else None
            )
            if target is None:
                if kind == "citations_delta" and block.body["type"] == "text":
                    citation = delta.get("citation")
                    citations = block.body.get("citations")
                    if citations is None:
                        citations = []
                    if not isinstance(citation, Mapping) or not isinstance(
                        citations, list
                    ):
                        raise NativeMessagesError("Invalid native citation delta.")
                    block.body["citations"] = [*citations, dict(citation)]
                    return None
                raise NativeMessagesError("Unsupported native content delta.")
            block_type, key = target
            value = delta.get(key)
            if block.body["type"] != block_type or not isinstance(value, str):
                raise NativeMessagesError(
                    "Native content delta does not match its block."
                )
            if key == "partial_json":
                if block.body.get("input"):
                    raise NativeMessagesError(
                        "Tool input mixes eager content and JSON deltas."
                    )
                block.parts.append(value)
            else:
                existing = block.body.get(key, "")
                if not isinstance(existing, str):
                    raise NativeMessagesError("Native content field must be a string.")
                block.fragments.setdefault(key, []).append(value)
            return None
        for key, parts in block.fragments.items():
            initial = block.body.get(key, "")
            if not isinstance(initial, str):
                raise AssertionError("Validated native content fields must be strings.")
            block.body[key] = initial + "".join(parts)
        if block.body["type"] == "thinking" and not block.body.get("signature"):
            raise NativeMessagesError(
                "Completed native thinking requires its signature."
            )
        if block.body["type"] == "tool_use" and block.parts:
            try:
                value = cast(JsonValue, json.loads("".join(block.parts)))
                if not isinstance(value, Mapping):
                    raise NativeMessagesError(
                        "Native tool arguments must decode to an object."
                    )
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise NativeMessagesError(
                    "Native tool arguments are incomplete or invalid JSON."
                ) from exc
            block.body["input"] = dict(value)
        self._blocks.pop(index)
        return block.body


class NativeMessagesRelay:
    """Preserve upstream IDs, indexes, fields and event order for Messages clients."""

    def __init__(
        self, *, public_model: str, replay_origin: ReplayOrigin | None = None
    ) -> None:
        self._public_model = public_model
        self._replay_origin = replay_origin
        self._state = NativeMessagesStreamState()

    @property
    def completed(self) -> bool:
        return self._state.completed

    @property
    def stop_reason(self) -> str | None:
        return self._state.stop_reason

    def feed(self, event_type: str, payload: Mapping[str, JsonValue]) -> str:
        completed = self._state.accept(event_type, payload)
        body = dict(payload)
        if event_type == "message_start":
            message = body.get("message")
            if not isinstance(message, dict):
                raise AssertionError("Validated message_start must contain a message.")
            body["message"] = {**message, "model": self._public_model}
            if (
                self._replay_origin is not None
                and isinstance(message.get("model"), str)
                and message["model"]
            ):
                self._replay_origin = replace(
                    self._replay_origin, model=message["model"]
                )
        prefix = ""
        if self._replay_origin is not None:
            block = body.get("content_block")
            if event_type == "content_block_start" and isinstance(block, dict):
                if block.get("type") == "thinking":
                    body["content_block"] = {**block, "signature": ""}
                elif block.get("type") == "redacted_thinking":
                    body["content_block"] = {
                        **block,
                        "data": encode_replay(ReplayRecord(self._replay_origin, block)),
                    }
            delta = body.get("delta")
            if (
                event_type == "content_block_delta"
                and isinstance(delta, dict)
                and delta.get("type") == "signature_delta"
            ):
                return ""
            if (
                completed is not None
                and completed.get("type") == "thinking"
                and completed.get("signature")
            ):
                signature = encode_replay(ReplayRecord(self._replay_origin, completed))
                prefix = _event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": body["index"],
                        "delta": {"type": "signature_delta", "signature": signature},
                    },
                )
        return prefix + _event(event_type, body)


def _event(kind: str, payload: JsonObject) -> str:
    return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
