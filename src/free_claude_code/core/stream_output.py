"""Presentation packets carrying source-established call completion evidence."""

from collections.abc import Iterable
from dataclasses import dataclass

from .anthropic.streaming.decoder import AnthropicSSEDecoder
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
            if not isinstance(data, dict):
                continue
            if data.get("type") == "content_block_start":
                block = data.get("content_block", {})
                if block.get("type") == "tool_use":
                    result.append(
                        CompletedCall("messages", (data["index"],), block.get("id"))
                    )
            elif data.get("type") == "response.output_item.done" and is_client_call(
                data.get("item", {})
            ):
                result.append(
                    CompletedCall(
                        "responses",
                        (data["output_index"],),
                        data["item"].get("call_id"),
                        data["item"].get("id"),
                    )
                )
            for choice in data.get("choices", []):
                result.extend(
                    CompletedCall(
                        "chat", (choice["index"], call["index"]), call.get("id")
                    )
                    for call in ((choice.get("delta") or {}).get("tool_calls") or [])
                    if (call.get("function") or call.get("custom") or {}).get("name")
                )
            for candidate in data.get("candidates", []):
                for index, part in enumerate(
                    candidate.get("content", {}).get("parts", [])
                ):
                    if "functionCall" in part:
                        result.append(
                            CompletedCall(
                                "gemini",
                                (candidate.get("index", 0), index),
                                part["functionCall"].get("id"),
                            )
                        )
    return tuple(dict.fromkeys(result))
