"""Translate upstream OpenAI Responses events into Anthropic Messages SSE."""

from dataclasses import dataclass
from typing import Any

from free_claude_code.core.anthropic.streaming import AnthropicStreamLedger
from free_claude_code.core.anthropic.usage import anthropic_input_usage_fields
from free_claude_code.core.history_replay import readable_reasoning
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec


class ResponsesStreamFailure(RuntimeError):
    """An upstream Responses stream reported a terminal failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        body: dict[str, Any] | None = None,
        event_type: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.body = body
        self.event_type = event_type
        self.payload = payload


@dataclass(slots=True)
class _ToolState:
    tool_index: int
    call_id: str
    name: str
    started: bool = False
    received_delta: bool = False
    stopped: bool = False


class ResponsesProviderStream:
    """Stateful one-way adapter for one upstream response."""

    def __init__(
        self,
        *,
        message_id: str,
        model: str,
        input_tokens: int,
        log_raw_events: bool = False,
        tool_names: OpenAIToolNameCodec | None = None,
        pad_empty: bool = True,
    ) -> None:
        self.ledger = AnthropicStreamLedger(
            message_id,
            model,
            input_tokens,
            log_raw_events=log_raw_events,
        )
        self.completed = False
        self._pad_empty = pad_empty
        self.generated_output = False
        self._tool_names = tool_names or OpenAIToolNameCodec.from_names(())
        self._tools: dict[str, _ToolState] = {}
        self._thinking_item_id: str | None = None
        self._readable_parts: dict[tuple[str, bool], str] = {}

    def start(self) -> list[str]:
        """Return the Anthropic message_start event."""

        return [self.ledger.message_start()]

    def feed(self, event_type: str, data: dict[str, Any]) -> list[str]:
        """Consume one Responses event and return zero or more Anthropic events."""

        if self.completed:
            return []
        if event_type == "response.output_item.added":
            return self._item_added(data)
        if event_type in {
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        }:
            return self._reasoning_delta(
                data, summary=event_type == "response.reasoning_summary_text.delta"
            )
        if event_type == "response.output_text.delta":
            return self._text_delta(data)
        if event_type == "response.function_call_arguments.delta":
            return self._tool_delta(data)
        if event_type == "response.output_item.done":
            return self._item_done(data)
        if event_type in {"response.completed", "response.incomplete"}:
            return self._finish(data, incomplete=event_type == "response.incomplete")
        if event_type in {"response.failed", "error", "response.error"}:
            raise responses_stream_failure_from_event(event_type, data)
        return []

    def _item_added(self, data: dict[str, Any]) -> list[str]:
        item = data.get("item")
        if not isinstance(item, dict):
            return []
        item_id = _string(item.get("id"))
        if item.get("type") == "function_call" and item_id:
            self._tools[item_id] = _ToolState(
                tool_index=len(self._tools),
                call_id=_string(item.get("call_id")) or item_id,
                name=self._tool_names.decode(_string(item.get("name"))),
            )
        if item.get("type") == "message" and self.ledger.blocks.text_started:
            return [self.ledger.stop_text_block()]
        if item.get("type") == "reasoning":
            return self._item_readable(item)
        return []

    def _reasoning_delta(
        self, data: dict[str, Any], *, summary: bool = False
    ) -> list[str]:
        delta = data.get("delta")
        if not isinstance(delta, str) or not delta:
            return []
        events = list(self.ledger.ensure_thinking_block())
        self._thinking_item_id = _string(data.get("item_id"))
        key = (self._thinking_item_id, summary)
        self._readable_parts[key] = self._readable_parts.get(key, "") + delta
        events.append(self.ledger.emit_thinking_delta(delta))
        self.generated_output = True
        return events

    def _item_readable(self, item: dict[str, Any]) -> list[str]:
        item_id = _string(item.get("id"))
        events: list[str] = []
        for text, summary in readable_reasoning(item):
            if text in {
                value
                for (source_id, _), value in self._readable_parts.items()
                if source_id == item_id
            }:
                continue
            key = (item_id, summary)
            previous = self._readable_parts.get(key, "")
            remaining = text.removeprefix(previous)
            if (
                self.ledger.blocks.thinking_started
                and item_id == self._thinking_item_id
            ):
                events.append(self.ledger.emit_thinking_delta(remaining))
                self._readable_parts[key] = previous + remaining
            else:
                events.extend(
                    self._reasoning_delta(
                        {"item_id": item_id, "delta": remaining}, summary=summary
                    )
                )
        return events

    def _text_delta(self, data: dict[str, Any]) -> list[str]:
        delta = data.get("delta")
        if not isinstance(delta, str) or not delta:
            return []
        events = (
            [] if self.ledger.blocks.text_started else [self.ledger.start_text_block()]
        )
        events.append(self.ledger.emit_text_delta(delta))
        self.generated_output = True
        return events

    def _tool_delta(self, data: dict[str, Any]) -> list[str]:
        item_id = _string(data.get("item_id"))
        delta = data.get("delta")
        if not item_id or not isinstance(delta, str):
            return []
        state = self._tools.get(item_id)
        if state is None:
            state = _ToolState(
                tool_index=len(self._tools),
                call_id=item_id,
                name="",
            )
            self._tools[item_id] = state
        events = [] if state.started else list(self.ledger.close_content_blocks())
        events.extend(self._ensure_tool_started(state))
        if delta:
            events.append(self.ledger.emit_tool_delta(state.tool_index, delta))
            state.received_delta = True
            self.generated_output = True
        return events

    def _item_done(self, data: dict[str, Any]) -> list[str]:
        item = data.get("item")
        if not isinstance(item, dict):
            return []
        item_type = item.get("type")
        item_id = _string(item.get("id"))
        if item_type == "function_call":
            state = self._tools.get(item_id)
            if state is None:
                state = _ToolState(
                    tool_index=len(self._tools),
                    call_id=_string(item.get("call_id")) or item_id,
                    name=self._tool_names.decode(_string(item.get("name"))),
                )
                self._tools[item_id] = state
            if not state.name:
                state.name = self._tool_names.decode(_string(item.get("name")))
            events = [] if state.started else list(self.ledger.close_content_blocks())
            events.extend(self._ensure_tool_started(state))
            arguments = item.get("arguments")
            if not state.received_delta and isinstance(arguments, str) and arguments:
                events.append(self.ledger.emit_tool_delta(state.tool_index, arguments))
                self.generated_output = True
            if not state.stopped:
                events.append(self.ledger.stop_tool_block(state.tool_index))
                state.stopped = True
                self.ledger.blocks.tool_states[state.tool_index].started = False
            return events
        if item_type == "reasoning":
            encrypted = item.get("encrypted_content")
            events = self._item_readable(item)
            if (
                self.ledger.blocks.thinking_started
                and item_id == self._thinking_item_id
            ):
                if isinstance(encrypted, str) and encrypted:
                    events.append(
                        self.ledger.content_block_delta(
                            self.ledger.blocks.thinking_index,
                            "signature_delta",
                            encrypted,
                        )
                    )
                events.append(self.ledger.stop_thinking_block())
                self._thinking_item_id = None
                return events
            if isinstance(encrypted, str) and encrypted:
                index = self.ledger.blocks.allocate_index()
                events.append(
                    self.ledger.content_block_start(
                        index, "redacted_thinking", data=encrypted
                    )
                )
                events.append(self.ledger.content_block_stop(index))
                self.generated_output = True
                return events
            return events
        return []

    def _ensure_tool_started(self, state: _ToolState) -> list[str]:
        if state.started:
            return []
        state.started = True
        self.generated_output = True
        return [
            self.ledger.start_tool_block(
                state.tool_index,
                state.call_id,
                state.name,
            )
        ]

    def _finish(self, data: dict[str, Any], *, incomplete: bool) -> list[str]:
        response = data.get("response")
        response = response if isinstance(response, dict) else {}
        events = []
        if self.ledger.blocks.thinking_started:
            events.append(self.ledger.stop_thinking_block())
        for state in self._tools.values():
            if state.started and not state.stopped:
                events.append(self.ledger.stop_tool_block(state.tool_index))
                state.stopped = True
        if self.ledger.blocks.text_started:
            events.append(self.ledger.stop_text_block())
        if self._pad_empty and not self.ledger.has_content_block():
            events.extend(self.ledger.ensure_text_block())
            events.append(self.ledger.emit_text_delta(" "))
            events.append(self.ledger.stop_text_block())
        usage = response.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        input_tokens = _integer(usage.get("input_tokens"))
        output_tokens = _integer(usage.get("output_tokens"))
        details = usage.get("input_tokens_details")
        details = details if isinstance(details, dict) else {}
        cached_tokens = _integer(details.get("cached_tokens"))
        cache_write_tokens = _integer(details.get("cache_write_tokens"))
        usage_fields = anthropic_input_usage_fields(
            input_tokens,
            cache_read_tokens=cached_tokens,
            cache_creation_tokens=cache_write_tokens,
        )
        stop_reason = "max_tokens" if incomplete else "end_turn"
        events.append(
            self.ledger.message_delta(
                self.ledger.final_stop_reason(stop_reason),
                output_tokens
                if output_tokens is not None
                else self.ledger.estimate_output_tokens(),
                input_tokens=input_tokens,
                usage_fields=usage_fields,
            )
        )
        events.append(self.ledger.message_stop())
        self.completed = True
        return events


def responses_stream_failure_from_event(
    event_type: str,
    data: dict[str, Any],
) -> ResponsesStreamFailure:
    """Retain one native failure event for provider-owned retry decisions."""

    response = data.get("response")
    response = response if isinstance(response, dict) else {}
    error = response.get("error", data.get("error"))
    if not isinstance(error, dict):
        error = data if event_type == "error" else {}
    message = error.get("message")
    code = error.get("code") or error.get("type")
    return ResponsesStreamFailure(
        message if isinstance(message, str) and message else "OpenAI response failed.",
        code=code if isinstance(code, str) else None,
        body=dict(error),
        event_type=event_type,
        payload=data,
    )


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
