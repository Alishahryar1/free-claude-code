"""Present native Anthropic Messages events as one Responses lifecycle."""

import json
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from free_claude_code.core.anthropic.native import NativeMessagesError
from free_claude_code.core.anthropic.native_stream import NativeMessagesStreamState
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.json_types import JsonObject, JsonValue

from .errors import ResponsesConversionError, openai_error_from_failure
from .ids import (
    new_message_item_id,
    new_reasoning_item_id,
    new_response_id,
    new_tool_item_id,
)
from .items import message_item, reasoning_item
from .models import OpenAIResponsesRequest
from .streaming.blocks import ReasoningBlockState, TextBlockState, ToolBlockState
from .streaming.completion import ResponseBlockCompleter, tool_item
from .streaming.event_builders import ResponseEventBuilder
from .streaming.ledger import ResponsesOutputLedger
from .tools import ResponsesToolIdentity


class _NativeUsage:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._cache_creation: JsonObject | None = None
        self._thinking_tokens: int | None = None
        self._final_output_seen = False

    def update(self, value: JsonValue, *, final: bool = False) -> None:
        if final:
            self._thinking_tokens = None
            output = value.get("output_tokens") if isinstance(value, Mapping) else None
            self._final_output_seen = (
                isinstance(output, int) and not isinstance(output, bool) and output >= 0
            )
        if not isinstance(value, Mapping):
            return
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "output_tokens",
        ):
            count = value.get(key)
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                self._counts[key] = count
        creation = value.get("cache_creation")
        if isinstance(creation, Mapping):
            self._cache_creation = {
                key: count
                for key, count in creation.items()
                if isinstance(count, int) and not isinstance(count, bool) and count >= 0
            }
        details = value.get("output_tokens_details")
        if isinstance(details, Mapping):
            thinking = details.get("thinking_tokens")
            if (
                isinstance(thinking, int)
                and not isinstance(thinking, bool)
                and thinking >= 0
            ):
                self._thinking_tokens = thinking

    def payload(self, *, require_final: bool = False) -> JsonObject | None:
        if require_final and not self._final_output_seen:
            return None
        if "input_tokens" not in self._counts or "output_tokens" not in self._counts:
            return None
        cached = self._counts.get("cache_read_input_tokens", 0)
        created = self._counts.get("cache_creation_input_tokens", 0)
        input_tokens = self._counts["input_tokens"] + cached + created
        output_tokens = self._counts["output_tokens"]
        details: JsonObject = {}
        if "cache_read_input_tokens" in self._counts:
            details["cached_tokens"] = cached
        if "cache_creation_input_tokens" in self._counts:
            details["cache_creation_tokens"] = created
        if self._cache_creation is not None:
            details["cache_creation"] = dict(self._cache_creation)
        usage: JsonObject = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if details:
            usage["input_tokens_details"] = details
        if self._thinking_tokens is not None and self._thinking_tokens <= output_tokens:
            usage["output_tokens_details"] = {"reasoning_tokens": self._thinking_tokens}
        return usage


class AnthropicToResponsesStream:
    """Keep native block order and signed replay without using a Chat presenter."""

    def __init__(
        self,
        request: OpenAIResponsesRequest,
        *,
        public_model: str,
        tool_identities: Mapping[str, ResponsesToolIdentity],
        replay_origin: ReplayOrigin,
    ) -> None:
        self._request = request
        self._public_model = public_model
        self._identities = tool_identities
        self._replay_origin = replay_origin
        self._response_id = new_response_id()
        self._created_at = int(time.time())
        self._native = NativeMessagesStreamState()
        self._ledger = ResponsesOutputLedger()
        self._events = ResponseEventBuilder()
        self._usage = _NativeUsage()
        self._terminal = False
        self._started = False
        self._completer = ResponseBlockCompleter(
            self._ledger,
            events=self._events,
            on_invalid_function_call=self._invalid_function,
        )

    @property
    def completed(self) -> bool:
        return self._terminal

    def start(self) -> list[str]:
        return []

    def _payload(
        self, status: str, *, failure: ExecutionFailure | None = None
    ) -> JsonObject:
        payload: JsonObject = {
            "id": self._response_id,
            "object": "response",
            "created_at": self._created_at,
            "status": status,
            "model": self._public_model,
            "output": cast(JsonValue, self._ledger.output()),
            "tools": self._request.tools or [],
            "tool_choice": self._request.tool_choice or "auto",
            "parallel_tool_calls": self._request.parallel_tool_calls
            if self._request.parallel_tool_calls is not None
            else True,
            "usage": self._usage.payload(require_final=status != "in_progress"),
            "error": cast(JsonValue, openai_error_from_failure(failure))
            if failure is not None
            else None,
            "incomplete_details": {"reason": "max_output_tokens"}
            if status == "incomplete"
            else None,
        }
        for key, value in (
            ("instructions", self._request.instructions),
            ("max_output_tokens", self._request.max_output_tokens),
            ("temperature", self._request.temperature),
            ("top_p", self._request.top_p),
            ("metadata", self._request.metadata),
            ("reasoning", self._request.reasoning),
        ):
            if value is not None:
                payload[key] = value
        return payload

    def feed(self, event_type: str, payload: Mapping[str, JsonValue]) -> list[str]:
        if self._terminal:
            raise NativeMessagesError(
                "Native event arrived after the Responses terminal event."
            )
        completed = self._native.accept(event_type, payload)
        if event_type == "ping":
            return []
        if event_type == "message_start":
            message = payload["message"]
            if not isinstance(message, Mapping):
                raise AssertionError("Validated message_start must contain a message.")
            self._usage.update(message.get("usage"))
            if isinstance(message.get("model"), str) and message["model"]:
                self._replay_origin = replace(
                    self._replay_origin, model=message["model"]
                )
            self._started = True
            return [self._events.response_created(self._payload("in_progress"))]
        if event_type == "message_delta":
            delta = payload.get("delta")
            self._usage.update(
                payload.get("usage"),
                final=isinstance(delta, Mapping)
                and delta.get("stop_reason") is not None,
            )
            return []
        if event_type == "message_stop":
            reason = self._native.stop_reason
            if reason not in {
                "end_turn",
                "stop_sequence",
                "tool_use",
                "max_tokens",
                "refusal",
            }:
                raise NativeMessagesError(
                    f"Responses cannot represent native stop reason {reason!r}."
                )
            self._terminal = True
            if reason == "max_tokens":
                return [self._events.response_incomplete(self._payload("incomplete"))]
            return [self._events.response_completed(self._payload("completed"))]
        index = payload["index"]
        if not isinstance(index, int):
            raise AssertionError("Validated native content event must have an index.")
        if event_type == "content_block_start":
            block = payload["content_block"]
            if not isinstance(block, Mapping):
                raise AssertionError("Validated content_block_start must have a block.")
            return self._start_block(index, block)
        if event_type == "content_block_delta":
            delta = payload["delta"]
            if not isinstance(delta, Mapping):
                raise AssertionError("Validated content delta must be an object.")
            return self._delta(index, delta)
        if event_type == "content_block_stop":
            if completed is None:
                raise AssertionError("Validated content stop must return its block.")
            return self._finish_block(index, completed)
        raise NativeMessagesError(f"Unsupported native event {event_type!r}.")

    def _start_block(self, index: int, block: Mapping[str, JsonValue]) -> list[str]:
        slot = self._ledger.reserve_output_slot()
        kind = block.get("type")
        if kind == "text":
            if any(
                key not in {"type", "text", "citations"} for key in block
            ) or block.get("citations"):
                raise NativeMessagesError(
                    "Responses cannot represent native text extensions."
                )
            text = block["text"]
            if not isinstance(text, str):
                raise AssertionError("Validated native text must be a string.")
            state = TextBlockState(index, slot, new_message_item_id())
            self._ledger.set_active_block(state)
            events = [
                self._events.output_item_added(
                    slot, message_item(state.item_id, "", "in_progress")
                ),
                self._events.content_part_added(state.item_id, slot),
            ]
            if text:
                state.text_parts.append(text)
                events.append(self._events.output_text_delta(state.item_id, slot, text))
            return events
        if kind in {"thinking", "redacted_thinking"}:
            state = ReasoningBlockState(index, slot, new_reasoning_item_id())
            self._ledger.set_active_block(state)
            text = block.get("thinking", "")
            if not isinstance(text, str):
                raise NativeMessagesError("Native thinking text must be a string.")
            events = [
                self._events.output_item_added(
                    slot, reasoning_item(state.item_id, "", "in_progress")
                )
            ]
            if text:
                state.text_parts.append(text)
                events.append(
                    self._events.reasoning_text_delta(state.item_id, slot, text)
                )
            return events
        if kind != "tool_use":
            raise NativeMessagesError(
                f"Responses cannot represent native block type {kind!r}."
            )
        if any(key not in {"type", "id", "name", "input", "caller"} for key in block):
            raise NativeMessagesError(
                "Responses cannot represent native tool extensions."
            )
        caller = block.get("caller")
        if caller is not None and caller != {"type": "direct"}:
            raise NativeMessagesError(
                "Responses cannot represent a server-managed tool caller."
            )
        name = block["name"]
        call_id = block["id"]
        if not isinstance(name, str) or not isinstance(call_id, str):
            raise AssertionError("Validated native tools must have string identities.")
        identity = self._identities.get(name)
        if identity is None:
            raise NativeMessagesError("Native output used an unknown tool name.")
        tool = ToolBlockState(
            index,
            slot,
            new_tool_item_id(identity.kind),
            call_id,
            identity.kind,
            identity.name,
            namespace=identity.namespace,
        )
        self._ledger.set_active_block(tool)
        return [
            self._events.output_item_added(slot, tool_item(tool, status="in_progress"))
        ]

    def _delta(self, index: int, delta: Mapping[str, JsonValue]) -> list[str]:
        state = self._ledger.active_block(index)
        kind = delta.get("type")
        if isinstance(state, TextBlockState) and kind == "text_delta":
            text = delta["text"]
            if not isinstance(text, str):
                raise AssertionError("Validated text delta must be a string.")
            state.text_parts.append(text)
            return [
                self._events.output_text_delta(state.item_id, state.output_index, text)
            ]
        if isinstance(state, ReasoningBlockState) and kind == "thinking_delta":
            text = delta["thinking"]
            if not isinstance(text, str):
                raise AssertionError("Validated thinking delta must be a string.")
            state.text_parts.append(text)
            return [
                self._events.reasoning_text_delta(
                    state.item_id, state.output_index, text
                )
            ]
        if isinstance(state, ReasoningBlockState) and kind == "signature_delta":
            return []
        if isinstance(state, ToolBlockState) and kind == "input_json_delta":
            # The completer emits validated arguments once, after native block stop.
            return []
        raise NativeMessagesError(
            "Responses cannot represent the native content delta."
        )

    def _finish_block(self, index: int, block: JsonObject) -> list[str]:
        state = self._ledger.active_block(index)
        if state is None:
            raise NativeMessagesError(
                "Native block has no corresponding Responses item."
            )
        if isinstance(state, ReasoningBlockState):
            state.encrypted_content = encode_replay(
                ReplayRecord(self._replay_origin, block)
            )
        elif isinstance(state, ToolBlockState):
            arguments = block.get("input")
            if not isinstance(arguments, Mapping):
                raise NativeMessagesError("Native tool input must be an object.")
            if state.kind == "custom" and (
                set(arguments) != {"input"}
                or not isinstance(arguments.get("input"), str)
            ):
                raise NativeMessagesError(
                    "Native custom tool input must contain exactly one text input."
                )
            state.argument_parts.append(
                json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            )
        events = self._completer.complete_block(state)
        self._ledger.pop_active_block(index)
        return events

    @staticmethod
    def _invalid_function(
        _state: ToolBlockState, error: ResponsesConversionError
    ) -> list[str]:
        raise NativeMessagesError("Native tool arguments are invalid.") from error

    def terminal_failure(self, failure: ExecutionFailure) -> list[str]:
        """Terminate once without completing partial tools or inventing arguments."""

        if self._terminal:
            return []
        events: list[str] = []
        if not self._started:
            self._started = True
            events.append(self._events.response_created(self._payload("in_progress")))
        for state in self._ledger.pop_active_blocks_by_output_order():
            if isinstance(state, TextBlockState):
                item = message_item(
                    state.item_id, "".join(state.text_parts), "incomplete"
                )
            elif isinstance(state, ReasoningBlockState):
                item = reasoning_item(
                    state.item_id, "".join(state.text_parts), "incomplete"
                )
            else:
                item = tool_item(state, status="incomplete")
            self._ledger.commit_output(state.output_index, item)
        self._terminal = True
        events.append(
            self._events.response_failed(self._payload("failed", failure=failure))
        )
        return events
