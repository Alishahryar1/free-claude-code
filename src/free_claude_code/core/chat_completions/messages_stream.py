"""Project validated native Messages events directly to Chat completions."""

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from ..anthropic.native_stream import NativeMessagesStreamState
from ..history_replay import ReplayOrigin, ReplayRecord, encode_replay
from ..json_types import JsonValue
from .response import ChatOutput, chat_failure


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
        self._usage: dict[str, int] = {}
        self._tool_index = 0

    @property
    def completed(self) -> bool:
        return self._output.completed

    def feed(self, event_type: str, payload: Mapping[str, JsonValue]) -> list[str]:
        block = self._state.accept(event_type, payload)
        data: dict[str, Any] = dict(payload)
        if event_type == "message_start":
            self._usage.update(data["message"].get("usage") or {})
            return self._output.start()
        if event_type == "message_delta":
            self._usage.update(data.get("usage") or {})
        elif event_type == "content_block_start":
            content = data["content_block"]
            kind = content["type"]
            if kind not in {"text", "thinking", "redacted_thinking", "tool_use"}:
                raise chat_failure(f"Chat cannot represent Messages block {kind}")
            if kind == "text":
                if content.get("citations"):
                    raise chat_failure(
                        "Chat cannot represent native Messages citations"
                    )
                return self._output.text(content["text"]) if content["text"] else []
            if kind == "thinking" and self._include_thoughts and content["thinking"]:
                return self._output.thought(
                    content["thinking"], source=f"rs_{data['index']}"
                )
        elif event_type == "content_block_delta":
            delta = data["delta"]
            if delta["type"] == "text_delta":
                return self._output.text(delta["text"])
            if delta["type"] == "thinking_delta" and self._include_thoughts:
                return self._output.thought(
                    delta["thinking"], source=f"rs_{data['index']}"
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
            prompt = sum(
                self._usage.get(key, 0)
                for key in (
                    "input_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
            )
            output = self._usage.get("output_tokens", 0)
            return self._output.finish(
                reason,
                {
                    "prompt_tokens": prompt,
                    "completion_tokens": output,
                    "total_tokens": prompt + output,
                    "prompt_tokens_details": {
                        "cached_tokens": self._usage.get("cache_read_input_tokens", 0)
                    },
                },
            )
        return []
