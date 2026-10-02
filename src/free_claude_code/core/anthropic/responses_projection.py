"""Map Responses source items into the request's Messages writer."""

from collections.abc import Iterator
from typing import Any, cast

from free_claude_code.core.history_replay import (
    ReplayOrigin,
    preserve_responses_reasoning,
)
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.core.stream_events import ItemCompletion, StreamEvent
from free_claude_code.core.stream_observations import ResponsesObservation

from .projection import MessagesWriter


class ResponsesMessagesProjection:
    """Own source bindings only; content and public lifecycle belong to the writer."""

    def __init__(
        self, writer: MessagesWriter, tool_names: OpenAIToolNameCodec | None
    ) -> None:
        self._writer = writer
        self._tool_names = tool_names or OpenAIToolNameCodec.from_names(())
        self._blocks: dict[tuple[object, ...], int] = {}
        self._open: set[int] = set()

    def _event(
        self,
        kind: str,
        payload: dict[str, Any],
        completion: ItemCompletion | None = None,
    ) -> list[StreamEvent]:
        return self._writer.accept_event(
            StreamEvent(kind, {"type": kind, **payload}, completion)
        )

    def _start(
        self, key: tuple[object, ...], body: dict[str, Any]
    ) -> list[StreamEvent]:
        if key in self._blocks:
            return []
        index = self._writer.allocate_block_index()
        self._blocks[key] = index
        self._open.add(index)
        return self._event(
            "content_block_start", {"index": index, "content_block": body}
        )

    def _delta(self, index: int, kind: str, value: str) -> list[StreamEvent]:
        field = {
            "text_delta": "text",
            "thinking_delta": "thinking",
            "signature_delta": "signature",
            "input_json_delta": "partial_json",
        }[kind]
        return self._event(
            "content_block_delta",
            {"index": index, "delta": {"type": kind, field: value}},
        )

    def _stop(
        self, index: int, completion: ItemCompletion | None = None
    ) -> list[StreamEvent]:
        if index not in self._open:
            return []
        self._open.remove(index)
        return self._event("content_block_stop", {"index": index}, completion)

    def feed(
        self, observation: ResponsesObservation, origin: ReplayOrigin
    ) -> Iterator[StreamEvent]:
        if observation.events:
            yield from self._writer.start_message()
        for event in observation.events:
            kind = event.kind
            data = cast(
                dict[str, Any], preserve_responses_reasoning(event.payload, origin)
            )
            identity = data.get("output_index", data.get("item_id"))
            item = data.get("item", {})
            if (
                kind == "response.output_item.added"
                and item.get("type") == "function_call"
            ):
                key = (identity, "tool")
                yield from (
                    self._start(
                        key,
                        {
                            "type": "tool_use",
                            "id": item.get("call_id") or item["id"],
                            "name": self._tool_names.decode(item.get("name", "")),
                            "input": {},
                        },
                    )
                )
                if arguments := item.get("arguments"):
                    yield from (
                        self._delta(self._blocks[key], "input_json_delta", arguments)
                    )
            elif kind == "response.function_call_arguments.delta":
                index = self._blocks[(identity, "tool")]
                yield from (self._delta(index, "input_json_delta", data["delta"]))
            elif kind in {
                "response.output_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            }:
                thinking = kind != "response.output_text.delta"
                key = (
                    identity,
                    "thinking" if thinking else "text",
                    data.get("content_index", data.get("summary_index", 0)),
                )
                yield from (
                    self._start(
                        key,
                        {"type": "thinking", "thinking": ""}
                        if thinking
                        else {"type": "text", "text": ""},
                    )
                )
                yield from (
                    self._delta(
                        self._blocks[key],
                        "thinking_delta" if thinking else "text_delta",
                        data["delta"],
                    )
                )
            elif kind == "response.output_item.done":
                indexes = [
                    index for key, index in self._blocks.items() if key[0] == identity
                ]
                if item.get("type") == "reasoning" and (
                    encrypted := item.get("encrypted_content")
                ):
                    thinking = [index for index in indexes if index in self._open]
                    if thinking:
                        yield from (
                            self._delta(thinking[-1], "signature_delta", encrypted)
                        )
                    else:
                        key = (identity, "opaque")
                        yield from (
                            self._start(
                                key, {"type": "redacted_thinking", "data": encrypted}
                            )
                        )
                        indexes.append(self._blocks[key])
                for index in indexes:
                    yield from (self._stop(index, event.item_completion))
            elif kind in {"response.completed", "response.incomplete"}:
                for index in tuple(self._open):
                    # A source terminal can close text, but cannot finalize a tool.
                    if all(
                        key[1] != "tool"
                        for key, value in self._blocks.items()
                        if value == index
                    ):
                        yield from (self._stop(index))
                if not self._blocks:
                    key = ("empty", "text")
                    yield from (self._start(key, {"type": "text", "text": " "}))
                    yield from (self._stop(self._blocks[key]))
                self._writer.responses_terminal(
                    data.get("response", {}), incomplete=kind == "response.incomplete"
                )
