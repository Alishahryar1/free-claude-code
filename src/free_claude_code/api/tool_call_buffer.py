"""Withhold incomplete client tool calls at the public SSE delivery boundary."""

from collections.abc import Mapping
from typing import Literal, cast

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.openai_responses import is_client_search

from .stream_frames import frame_data, replace_frame_data

type CallKey = int | str

_UNFINISHED = frozenset({"in_progress", "incomplete", "failed"})
_FAILURES = frozenset({"error", "response.error", "response.failed"})
_RESPONSES_TERMINALS = frozenset({"response.completed", "response.incomplete"})
_ARGUMENT_EVENTS = frozenset(
    {
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
    }
)


def _client_call(item: Mapping[str, object]) -> bool:
    kind = item.get("type")
    if kind in {
        "function_call",
        "custom_tool_call",
        "computer_call",
        "local_shell_call",
        "apply_patch_call",
    }:
        return True
    if kind == "tool_search_call":
        return is_client_search(cast(Mapping[str, JsonValue], item))
    environment = item.get("environment")
    return (
        kind == "shell_call"
        and isinstance(environment, Mapping)
        and environment.get("type") == "local"
    )


class ToolCallBuffer:
    """Select complete groups in source order, without owning public delivery."""

    def __init__(self, wire_api: Literal["messages", "responses"]) -> None:
        self._wire_api = wire_api
        self._frames: list[str] = []
        self._calls: set[CallKey] = set()
        self._pending: set[CallKey] = set()
        self._delivered: set[CallKey] = set()
        self._discarded: set[CallKey] = set()
        self._ids: dict[str, CallKey] = {}
        self._indices: dict[int, CallKey] = {}

    def feed(self, frame: str) -> list[str]:
        events = parse_sse_text(frame)
        if not events:
            return self._publish(frame)
        event = events[0]
        data = cast(Mapping[str, object], event.data)
        kind = event.event or data.get("type")
        if kind in _FAILURES:
            self._discard_group()
            return [self._filter_snapshot(frame, failed=True)]
        if self._wire_api == "messages":
            index = data.get("index")
            block = data.get("content_block")
            if isinstance(index, int) and not isinstance(index, bool):
                if (
                    kind == "content_block_start"
                    and isinstance(block, Mapping)
                    and block.get("type") == "tool_use"
                ):
                    self._begin_call(index)
                elif kind == "content_block_stop":
                    self._pending.discard(index)
            if kind == "message_stop" and self._pending:
                self._discard_group()
        else:
            item = data.get("item")
            item = item if isinstance(item, Mapping) else {}
            if _client_call(item) or kind in _ARGUMENT_EVENTS:
                key = self._key(data, item)
                if key is not None:
                    if kind == "response.output_item.done":
                        self._calls.add(key)
                        if item.get("status") in _UNFINISHED:
                            self._pending.add(key)
                        else:
                            self._pending.discard(key)
                    else:
                        self._begin_call(key)
            if kind in _RESPONSES_TERMINALS:
                response = data.get("response")
                output = (
                    response.get("output") if isinstance(response, Mapping) else None
                )
                if isinstance(response, Mapping) and isinstance(output, list):
                    for value in output:
                        if isinstance(value, Mapping) and _client_call(value):
                            key = self._key({}, value)
                            if not self._withheld_item(value, response.get("status")):
                                self._pending.discard(key)
                if self._pending:
                    self._discard_group()
            if isinstance(data.get("response"), Mapping):
                frame = self._filter_snapshot(frame)
        return self._publish(frame)

    def _begin_call(self, key: CallKey) -> None:
        if key not in self._calls:
            self._calls.add(key)
            self._pending.add(key)

    def _key(
        self, data: Mapping[str, object], item: Mapping[str, object]
    ) -> CallKey | None:
        ids = [
            value
            for value in (data.get("item_id"), item.get("id"), item.get("call_id"))
            if isinstance(value, str) and value
        ]
        index = data.get("output_index")
        index = (
            index if isinstance(index, int) and not isinstance(index, bool) else None
        )
        key = next((self._ids[value] for value in ids if value in self._ids), None)
        if key is None and index is not None:
            key = self._indices.get(index, index)
        if key is None and ids:
            key = ids[0]
        if key is not None:
            for value in ids:
                self._ids[value] = key
            if index is not None:
                self._indices[index] = key
        return key

    def _publish(self, frame: str) -> list[str]:
        if not self._calls:
            return [frame]
        self._frames.append(frame)
        if self._pending:
            return []
        frames, self._frames = self._frames, []
        self._delivered.update(self._calls)
        self._calls.clear()
        return frames

    def _discard_group(self) -> None:
        self._discarded.update(self._calls)
        self._frames.clear()
        self._calls.clear()
        self._pending.clear()

    def _filter_snapshot(self, frame: str, *, failed: bool = False) -> str:
        # Parse only frames we may edit using decimal-aware JSON; all other
        # payloads, including argument strings and SSE framing, stay verbatim.
        value = frame_data(frame)
        if value is None:
            return frame
        response = value.get("response")
        if not isinstance(response, dict):
            return frame
        output = response.get("output")
        if not isinstance(output, list):
            return frame
        retained = [
            item
            for item in output
            if not (
                isinstance(item, Mapping)
                and _client_call(item)
                and self._withheld_item(item, response.get("status"), failed=failed)
            )
        ]
        if len(retained) == len(output):
            return frame
        response["output"] = retained
        return replace_frame_data(frame, value)

    def _withheld_item(
        self,
        item: Mapping[str, object],
        response_status: object,
        *,
        failed: bool = False,
    ) -> bool:
        key = self._key({}, item)
        status = item.get("status")
        return (
            key in self._discarded
            or status in _UNFINISHED
            or (failed and key not in self._delivered)
            or (
                status is None
                and response_status != "completed"
                and key not in self._delivered
            )
        )
