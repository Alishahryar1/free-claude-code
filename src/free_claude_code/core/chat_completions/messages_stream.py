"""Project validated native Messages events directly to Chat completions."""

from collections.abc import Mapping
from uuid import uuid4

from ..anthropic.native_stream import NativeMessagesStreamState
from ..history_replay import ReplayOrigin, ReplayRecord, encode_replay
from ..json_types import JsonObject, JsonValue, json_object, json_text
from .response import ChatOutput, chat_failure


def _token_count(value: JsonValue) -> int | float:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    raise chat_failure("Messages usage token count must be a number")


class ChatMessagesStream:
    def __init__(
        self,
        *,
        model: str,
        origin: ReplayOrigin,
        reserved_tool_ids: frozenset[str],
        include_usage: bool,
        include_thoughts: bool,
    ) -> None:
        self._state = NativeMessagesStreamState()
        self._output = ChatOutput(model, include_usage=include_usage)
        self._origin = origin
        self._reserved = set(reserved_tool_ids)
        self._include_thoughts = include_thoughts
        self._usage: JsonObject = {}
        self._tool_index = 0

    @property
    def completed(self) -> bool:
        return self._output.completed

    def feed(self, event_type: str, payload: Mapping[str, JsonValue]) -> list[str]:
        block = self._state.accept(event_type, payload)
        data = payload
        if event_type == "message_start":
            if self._output.include_usage:
                message = json_object(data["message"])
                self._usage.update(json_object(message.get("usage") or {}))
            return self._output.start()
        if event_type == "message_delta":
            if self._output.include_usage:
                self._usage.update(json_object(data.get("usage") or {}))
        elif event_type == "content_block_start":
            content = json_object(data["content_block"])
            kind = json_text(content["type"])
            if kind not in {"text", "thinking", "redacted_thinking", "tool_use"}:
                raise chat_failure(f"Chat cannot represent Messages block {kind}")
            if kind == "text":
                if content.get("citations"):
                    raise chat_failure(
                        "Chat cannot represent native Messages citations"
                    )
                return (
                    self._output.text(json_text(content["text"]))
                    if json_text(content["text"])
                    else []
                )
            if (
                kind == "thinking"
                and self._include_thoughts
                and json_text(content["thinking"])
            ):
                return self._output.thought(
                    json_text(content["thinking"]), source=f"rs_{data['index']}"
                )
        elif event_type == "content_block_delta":
            delta = json_object(data["delta"])
            if delta["type"] == "text_delta":
                return self._output.text(json_text(delta["text"]))
            if delta["type"] == "thinking_delta" and self._include_thoughts:
                return self._output.thought(
                    json_text(delta["thinking"]), source=f"rs_{data['index']}"
                )
            if delta["type"] == "citations_delta":
                raise chat_failure("Chat cannot represent native Messages citations")
        if block is not None:
            if block.body["type"] in {"thinking", "redacted_thinking"}:
                return self._output.replay(
                    encode_replay(ReplayRecord(self._origin, block.body)),
                    source=f"rs_{data['index']}",
                )
            if block.body["type"] == "tool_use":
                name = str(block.body["name"])
                assert block.tool_arguments is not None
                call_id = str(block.body["id"])
                if call_id in self._reserved:
                    call_id = f"call_{uuid4().hex}"
                self._reserved.add(call_id)
                events = self._output.tool(
                    self._tool_index, call_id, name, block.tool_arguments
                )
                self._tool_index += 1
                return events
        if event_type == "message_stop":
            reason = {
                "end_turn": "stop",
                "stop_sequence": "stop",
                "tool_use": "tool_calls",
                "max_tokens": "length",
                "refusal": "content_filter",
            }.get(self._state.stop_reason or "")
            if reason is None:
                raise chat_failure(
                    f"Chat cannot represent Messages stop reason {self._state.stop_reason}"
                )
            if not self._output.include_usage:
                return self._output.finish(reason)
            cached = _token_count(self._usage.get("cache_read_input_tokens", 0))
            prompt = (
                _token_count(self._usage.get("input_tokens", 0))
                + cached
                + _token_count(self._usage.get("cache_creation_input_tokens", 0))
            )
            output = _token_count(self._usage.get("output_tokens", 0))
            return self._output.finish(
                reason,
                {
                    "prompt_tokens": prompt,
                    "completion_tokens": output,
                    "total_tokens": prompt + output,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            )
        return []
