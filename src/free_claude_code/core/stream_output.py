"""Presentation packets carrying source-established call completion evidence."""

from collections.abc import Iterable
from dataclasses import dataclass

from .anthropic.streaming.decoder import AnthropicSSEDecoder
from .json_types import json_index, json_object, json_objects, optional_string
from .openai_responses import is_client_call
from .stream_recovery import CompletedCall as CompletedCall


@dataclass(frozen=True, slots=True)
class StreamOutput:
    frames: tuple[str, ...]
    completed_calls: tuple[CompletedCall, ...] = ()


def completed_output(frames: Iterable[str]) -> StreamOutput:
    """Address calls in an already source-complete presentation, without retaining bodies.

    Only source owners may use this constructor. Serialization events are used to
    address the destination call, never to decide whether the source finished it.
    """
    packet = tuple(frame for frame in frames if frame)
    return StreamOutput(packet, call_references(packet))


def call_references(frames: Iterable[str]) -> tuple[CompletedCall, ...]:
    decoder = AnthropicSSEDecoder()
    result: list[CompletedCall] = []
    for frame in frames:
        for event in decoder.feed(frame):
            data = event.data
            if data.get("type") == "content_block_start":
                block = json_object(data.get("content_block", {}))
                if block.get("type") == "tool_use":
                    result.append(
                        CompletedCall(
                            "messages",
                            (json_index(data["index"]),),
                            optional_string(block.get("id")),
                        )
                    )
            elif data.get("type") == "response.output_item.done":
                item = json_object(data.get("item", {}))
                if is_client_call(item):
                    result.append(
                        CompletedCall(
                            "responses",
                            (json_index(data["output_index"]),),
                            optional_string(item.get("call_id")),
                            optional_string(item.get("id")),
                        )
                    )
            for choice in json_objects(data.get("choices", [])):
                delta = json_object(choice.get("delta") or {})
                for call in json_objects(delta.get("tool_calls") or []):
                    function = json_object(
                        call.get("function") or call.get("custom") or {}
                    )
                    if function.get("name"):
                        result.append(
                            CompletedCall(
                                "chat",
                                (
                                    json_index(choice["index"]),
                                    json_index(call["index"]),
                                ),
                                optional_string(call.get("id")),
                            )
                        )
            for candidate in json_objects(data.get("candidates", [])):
                candidate_index = candidate.get("index", 0)
                content = json_object(candidate.get("content", {}))
                for index, part in enumerate(json_objects(content.get("parts", []))):
                    if "functionCall" in part:
                        result.append(
                            CompletedCall(
                                "gemini",
                                # Native Gemini also accepts 0.0 for its sole candidate.
                                (
                                    0
                                    if candidate_index == 0
                                    else json_index(candidate_index),
                                    index,
                                ),
                                optional_string(
                                    json_object(part["functionCall"]).get("id")
                                ),
                            )
                        )
    return tuple(dict.fromkeys(result))
