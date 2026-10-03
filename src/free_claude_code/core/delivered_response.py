"""Bounded, request-local evidence of output released to a stream consumer."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import simplejson

from .openai_responses import is_client_search
from .token_estimation import estimate_text_tokens

RECOVERY_RETENTION_BYTES = 1_048_576

CLIENT_CALL_TYPES = frozenset(
    {
        "function_call",
        "custom_tool_call",
        "computer_call",
        "local_shell_call",
        "apply_patch_call",
    }
)


def client_call(item: dict[str, Any]) -> bool:
    kind = item.get("type")
    if kind == "tool_search_call":
        return is_client_search(item)
    return kind in CLIENT_CALL_TYPES or (
        kind == "shell_call"
        and isinstance(item.get("environment"), dict)
        and item["environment"].get("type") == "local"
    )


@dataclass(frozen=True, slots=True)
class DeliveredPrefix:
    text: str
    thinking: str
    has_calls: bool
    eligible: bool


class DeliveredResponse:
    """Track this response only; the harness continues to own conversation history."""

    def __init__(self, wire_api: Literal["messages", "responses"]) -> None:
        self.wire_api = wire_api
        self.items: dict[int, dict[str, Any]] = {}
        self.closed: set[int] = set()
        self._text_closed: set[tuple[int, str, int]] = set()
        self._part_closed: set[tuple[int, str, int]] = set()
        self.header: dict[str, Any] = {}
        self.sequence = -1
        self.unsafe_reason: str | None = None
        self._bytes = 0
        self._limited = False
        self._limited_output_tokens = 0
        self.progress = 0

    def _retain(self, value: object) -> bool:
        if self._limited:
            return False
        self._bytes += len(
            simplejson.dumps(value, use_decimal=True, ensure_ascii=False).encode(
                "utf-8"
            )
        )
        if self._bytes > RECOVERY_RETENTION_BYTES:
            self._limited_output_tokens = self.output_tokens()
            self._limited = True
            self.unsafe_reason = "retention_limit"
            self.items.clear()
            self.closed.clear()
            self._text_closed.clear()
            self._part_closed.clear()
            return False
        return True

    def observe(self, data: dict[str, Any]) -> None:
        kind = data.get("type", "")
        if not isinstance(kind, str):
            self.unsafe_reason = "unknown_event"
            return
        sequence = data.get("sequence_number")
        if isinstance(sequence, int):
            self.sequence = max(self.sequence, sequence)
        if kind in {"message_start", "response.created"} and not self.header:
            self.header = deepcopy(data.get("message", data.get("response", {})))
        delta = data.get("delta", "")
        text = (
            delta
            if isinstance(delta, str)
            else "".join(
                delta.get(field, "") for field in ("text", "thinking", "partial_json")
            )
            if isinstance(delta, dict)
            else ""
        )
        block = data.get("content_block", {})
        if kind == "content_block_start":
            text += block.get("text", "") + block.get("thinking", "")
        if (
            text
            or data.get("part", {}).get("refusal")
            or (kind == "response.refusal.done" and data.get("refusal"))
            or (kind == "content_block_start" and block.get("type") == "tool_use")
            or (
                kind == "response.output_item.done"
                and client_call(data.get("item", {}))
            )
        ):
            self.progress += 1
        if not self._limited:
            if self.wire_api == "messages":
                self._messages(kind, data)
            else:
                self._responses(kind, data)
        if self._limited:
            self._limited_output_tokens += estimate_text_tokens(text)

    def _messages(self, kind: str, data: dict[str, Any]) -> None:
        index = data.get("index")
        if not isinstance(index, int):
            if kind not in {
                "message_start",
                "message_delta",
                "message_stop",
                "ping",
                "error",
            }:
                self.unsafe_reason = "unknown_event"
            return
        if kind == "content_block_start":
            block = data.get("content_block", {})
            if not self._retain(block):
                return
            self.items[index] = deepcopy(block)
            if block.get("type") not in {
                "text",
                "thinking",
                "redacted_thinking",
                "tool_use",
            }:
                self.unsafe_reason = "native_content"
            if block.get("signature") or block.get("type") == "redacted_thinking":
                self.unsafe_reason = "opaque_reasoning"
        elif kind == "content_block_delta":
            delta = data.get("delta", {})
            if not self._retain(delta):
                return
            block = self.items.get(index)
            if block is None:
                self.unsafe_reason = "missing_block"
                return
            field = {
                "text_delta": "text",
                "thinking_delta": "thinking",
                "input_json_delta": "partial_json",
            }.get(delta.get("type"))
            if field:
                block[field] = block.get(field, "") + delta.get(field, "")
            else:
                self.unsafe_reason = "opaque_delta"
                if delta.get("type") == "signature_delta":
                    block["signature"] = block.get("signature", "") + delta.get(
                        "signature", ""
                    )
        elif kind == "content_block_stop":
            self.closed.add(index)

    def _responses(self, kind: str, data: dict[str, Any]) -> None:
        index = data.get("output_index")
        if not isinstance(index, int):
            if kind not in {
                "response.created",
                "response.in_progress",
                "response.completed",
                "response.incomplete",
                "response.failed",
                "response.error",
                "error",
                "ping",
            }:
                self.unsafe_reason = "unknown_event"
            return
        if kind in {"response.output_item.added", "response.output_item.done"}:
            item = data.get("item", {})
            if not self._retain(item):
                return
            self.items[index] = deepcopy(item)
            if item.get("type") not in {"message", "reasoning"} and not client_call(
                item
            ):
                self.unsafe_reason = "native_content"
            if item.get("encrypted_content"):
                self.unsafe_reason = "opaque_reasoning"
            if any(part.get("type") == "refusal" for part in item.get("content", [])):
                self.unsafe_reason = "refusal"
            if kind.endswith(".done"):
                self.closed.add(index)
        elif kind in {
            "response.content_part.added",
            "response.reasoning_summary_part.added",
            "response.content_part.done",
            "response.reasoning_summary_part.done",
        }:
            part = data.get("part", {})
            if not self._retain(part) or (
                kind.endswith(".done")
                and not self._retain(
                    (
                        kind,
                        index,
                        data.get("content_index", data.get("summary_index", 0)),
                    )
                )
            ):
                return
            item = self.items.get(index)
            if item is None:
                self.unsafe_reason = "missing_item"
                return
            field = "summary" if "reasoning" in kind else "content"
            position = data.get(
                "summary_index" if field == "summary" else "content_index", 0
            )
            parts = item.setdefault(field, [])
            while len(parts) <= position:
                parts.append({})
            parts[position] = deepcopy(part)
            if kind.endswith(".done"):
                self._part_closed.add((index, field, position))
            if part.get("type") == "refusal":
                self.unsafe_reason = "refusal"
            elif part.get("type") not in {
                "output_text",
                "summary_text",
                "reasoning_text",
            }:
                self.unsafe_reason = "native_content"
        elif kind in {
            "response.output_text.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
            "response.refusal.delta",
            "response.output_text.done",
            "response.reasoning_summary_text.done",
            "response.reasoning_text.done",
            "response.refusal.done",
        }:
            done = kind.endswith(".done")
            text_field = "refusal" if "refusal" in kind else "text"
            delta = data.get(text_field if done else "delta", "")
            if not self._retain(delta) or (
                done
                and not self._retain(
                    (
                        kind,
                        index,
                        data.get("content_index", data.get("summary_index", 0)),
                    )
                )
            ):
                return
            item = self.items.get(index)
            if item is None:
                self.unsafe_reason = "missing_item"
                return
            field = "summary" if "reasoning_summary" in kind else "content"
            position = data.get(
                "summary_index" if field == "summary" else "content_index", 0
            )
            parts = item.setdefault(field, [])
            while len(parts) <= position:
                parts.append(
                    {
                        "type": "summary_text"
                        if field == "summary"
                        else "reasoning_text"
                        if "reasoning" in kind
                        else "refusal"
                        if text_field == "refusal"
                        else "output_text",
                        text_field: "",
                    }
                )
            parts[position][text_field] = (
                delta if done else parts[position].get(text_field, "") + delta
            )
            if done:
                self._text_closed.add((index, field, position))
            if text_field == "refusal":
                self.unsafe_reason = "refusal"
        elif kind.endswith(".delta") and kind not in {
            "response.function_call_arguments.delta",
            "response.custom_tool_call_input.delta",
        }:
            self.unsafe_reason = "native_delta"
        elif kind not in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
        }:
            self.unsafe_reason = "unknown_event"

    def snapshot(self) -> DeliveredPrefix:
        text: list[str] = []
        thinking: list[str] = []
        has_calls = False
        for index, item in sorted(self.items.items()):
            kind = item.get("type")
            if self.wire_api == "messages":
                if kind == "text":
                    text.append(item.get("text", ""))
                elif kind == "thinking":
                    thinking.append(item.get("thinking", ""))
                elif kind == "tool_use" and index in self.closed:
                    has_calls = True
            elif kind == "message":
                text.extend(part.get("text", "") for part in item.get("content", []))
            elif kind == "reasoning":
                thinking.extend(
                    part.get("text", "")
                    for part in [*item.get("summary", []), *item.get("content", [])]
                )
            elif client_call(item) and index in self.closed:
                has_calls = True
        return DeliveredPrefix(
            "".join(text), "".join(thinking), has_calls, self.unsafe_reason is None
        )

    def output_tokens(self, output: list[dict[str, Any]] | None = None) -> int:
        if output is None and self._limited:
            return self._limited_output_tokens
        return estimate_text_tokens(
            simplejson.dumps(
                list(self.items.values()) if output is None else output,
                use_decimal=True,
                ensure_ascii=False,
            )
        )

    @property
    def can_handoff(self) -> bool:
        if self.unsafe_reason not in {None, "opaque_reasoning", "opaque_delta"}:
            return False
        return all(
            index in self.closed
            for index, item in self.items.items()
            if item.get("type") in {"thinking", "redacted_thinking", "reasoning"}
        )

    def closing_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for index, item in sorted(self.items.items()):
            if index in self.closed:
                continue
            if self.wire_api == "messages":
                events.append({"type": "content_block_stop", "index": index})
            else:
                for field in ("content", "summary"):
                    for position, part in enumerate(item.get(field, [])):
                        key = (index, field, position)
                        if key in self._part_closed:
                            continue
                        summary = field == "summary"
                        identity = {
                            "item_id": item["id"],
                            "output_index": index,
                            "summary_index" if summary else "content_index": position,
                        }
                        if key not in self._text_closed:
                            text_kind = (
                                "response.reasoning_summary_text.done"
                                if summary
                                else "response.reasoning_text.done"
                                if part.get("type") == "reasoning_text"
                                else "response.output_text.done"
                            )
                            events.append(
                                {
                                    "type": text_kind,
                                    **identity,
                                    "text": part.get("text", ""),
                                }
                            )
                        if part.get("type") != "reasoning_text":
                            events.append(
                                {
                                    "type": "response.reasoning_summary_part.done"
                                    if summary
                                    else "response.content_part.done",
                                    **identity,
                                    "part": deepcopy(part),
                                }
                            )
                events.append(
                    {
                        "type": "response.output_item.done",
                        "output_index": index,
                        "item": {**deepcopy(item), "status": "completed"},
                    }
                )
        return events

    def handoff_events(self) -> list[dict[str, Any]]:
        """Signal handoff; the public writer fills output and usage after closure."""
        if self.wire_api == "messages":
            return [
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {},
                },
                {"type": "message_stop"},
            ]
        response = {
            **deepcopy(self.header),
            "status": "completed",
        }
        return [{"type": "response.completed", "response": response}]

    def usage(self, output: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        initial = deepcopy(self.header.get("usage") or {})
        input_tokens = initial.get("input_tokens", 0)
        output_tokens = self.output_tokens(output)
        return {
            **initial,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
