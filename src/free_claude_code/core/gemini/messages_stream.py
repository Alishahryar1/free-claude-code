"""Direct Messages-source projection to Gemini candidates."""

from collections.abc import Mapping
from copy import deepcopy
from uuid import uuid4

from ..anthropic.native_stream import NativeMessagesStreamState
from ..history_replay import ReplayOrigin, ReplayRecord
from ..json_types import JsonObject, JsonValue
from ..openai_tool_names import OpenAIToolNameCodec
from .response import GeminiOutput, conversion_failure, function_arguments


class GeminiMessagesStream:
    def __init__(
        self,
        *,
        response_id: str,
        model: str,
        include_thoughts: bool,
        origin: ReplayOrigin,
        tool_names: OpenAIToolNameCodec,
        declared_names: frozenset[str],
        reserved_tool_ids: frozenset[str],
    ) -> None:
        self._state = NativeMessagesStreamState()
        self._output = GeminiOutput(
            response_id, model, include_thoughts=include_thoughts
        )
        self._origin = origin
        self._names = tool_names
        self._declared = declared_names
        self._reserved = set(reserved_tool_ids)
        self._has_reasoning = False
        self._usage: dict[str, int] = {}

    @property
    def completed(self) -> bool:
        return self._output.completed

    def handoff(self) -> list[str]:
        return self._output.handoff()

    def feed(self, event_type: str, payload: Mapping[str, JsonValue]) -> list[str]:
        block = self._state.accept(event_type, payload)
        if event_type == "message_start":
            message = payload.get("message")
            assert isinstance(message, Mapping)
            self._read_usage(message.get("usage"))
            self._usage.pop("output_tokens", None)
        elif event_type == "message_delta":
            self._read_usage(payload.get("usage"))
        elif event_type == "content_block_start":
            content = payload.get("content_block")
            assert isinstance(content, Mapping)
            kind = content.get("type")
            if kind not in {"text", "thinking", "redacted_thinking", "tool_use"}:
                raise conversion_failure(
                    f"Gemini cannot represent upstream Messages block {kind}"
                )
            if kind == "text" and content.get("text"):
                return self._output.part({"text": content["text"]})
            if kind == "thinking" and content.get("thinking"):
                return self._output.part({"text": content["thinking"], "thought": True})
        elif event_type == "content_block_delta":
            delta = payload.get("delta")
            assert isinstance(delta, Mapping)
            if delta.get("type") == "text_delta":
                return self._output.part({"text": delta["text"]})
            if delta.get("type") == "thinking_delta":
                return self._output.part({"text": delta["thinking"], "thought": True})
            if delta.get("type") == "citations_delta":
                raise conversion_failure(
                    "Gemini cannot represent upstream native citations"
                )
        if block is not None:
            if block.body.get("type") in {"thinking", "redacted_thinking"}:
                self._has_reasoning = True
                self._output.capture(ReplayRecord(self._origin, deepcopy(block.body)))
            elif block.body.get("type") == "tool_use":
                name = self._names.decode(str(block.body["name"]))
                if name not in self._declared:
                    raise conversion_failure(
                        f"Upstream called an undeclared function: {name}"
                    )
                assert block.tool_arguments is not None
                call_id = str(block.body["id"])
                if call_id in self._reserved:
                    call_id = f"call_{uuid4().hex}"
                self._reserved.add(call_id)
                return self._output.part(
                    {
                        "functionCall": {
                            "id": call_id,
                            "name": name,
                            "args": function_arguments(block.tool_arguments),
                        }
                    }
                )
        if event_type == "message_stop":
            reason = self._state.stop_reason
            if reason not in {
                "end_turn",
                "stop_sequence",
                "tool_use",
                "max_tokens",
                "refusal",
            }:
                raise conversion_failure(
                    f"Gemini cannot represent Messages stop reason {reason}"
                )
            usage: JsonObject = {}
            if "input_tokens" in self._usage:
                usage["promptTokenCount"] = sum(
                    self._usage.get(key, 0)
                    for key in (
                        "input_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                )
            if "output_tokens" in self._usage and not self._has_reasoning:
                usage["candidatesTokenCount"] = self._usage["output_tokens"]
            if "cache_read_input_tokens" in self._usage:
                usage["cachedContentTokenCount"] = self._usage[
                    "cache_read_input_tokens"
                ]
            if "input_tokens" in self._usage and "output_tokens" in self._usage:
                usage["totalTokenCount"] = sum(
                    self._usage.get(key, 0)
                    for key in (
                        "input_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                        "output_tokens",
                    )
                )
            return self._output.finish(
                "MAX_TOKENS"
                if reason == "max_tokens"
                else "SAFETY"
                if reason == "refusal"
                else "STOP",
                usage,
            )
        return []

    def _read_usage(self, value: JsonValue) -> None:
        if isinstance(value, Mapping):
            self._usage.update(
                {
                    key: count
                    for key, count in value.items()
                    if isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= 0
                }
            )
