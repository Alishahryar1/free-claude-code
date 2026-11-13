"""Collect one replaceable private response without claiming public delivery."""

import json
import sys
from collections.abc import AsyncIterator
from typing import Literal

from free_claude_code.core.anthropic.errors import anthropic_failure_from_error
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.tool_call_buffer import RetentionBudget
from free_claude_code.core.trace import close_stream_input

type WireProtocol = Literal["messages", "responses", "chat", "gemini"]


async def collect_response(
    body: AsyncIterator[StreamOutput],
    *,
    protocol: WireProtocol,
    native_json: bool = False,
) -> tuple[str, ...]:
    state = StreamDeliveryState()
    revision = state.attempt_revision
    frames: list[str] = []
    budget = RetentionBudget()
    with bind_stream_delivery(state):
        try:
            while True:
                try:
                    packet = await anext(body)
                except StopAsyncIteration:
                    if revision != state.attempt_revision:
                        frames.clear()
                    break
                if revision != state.attempt_revision:
                    revision = state.attempt_revision
                    frames.clear()
                    budget = RetentionBudget()
                budget.add(
                    sum(len(frame.encode("utf-8")) for frame in packet.frames),
                    max(1, len(packet.frames)),
                )
                frames.extend(packet.frames)
            _validate(frames, protocol=protocol, native_json=native_json)
            return tuple(frames)
        finally:
            await close_stream_input(
                body,
                owner="private_response",
                source="application",
                preserved_error=sys.exception(),
            )


async def collected_frames(
    body: AsyncIterator[StreamOutput], *, protocol: WireProtocol
) -> AsyncIterator[str]:
    for frame in await collect_response(body, protocol=protocol):
        yield frame


def _validate(frames: list[str], *, protocol: WireProtocol, native_json: bool) -> None:
    if native_json:
        data = json.loads("".join(frames))
        if (
            not isinstance(data, dict)
            or data.get("type") != "message"
            or data.get("stop_reason") is None
        ):
            raise _failure("Messages JSON response ended without a completed message")
        return
    decoder = AnthropicSSEDecoder()
    terminal = False
    message_reason = False
    choices: dict[int, bool] = {}
    chat_done = False
    for chunk in frames:
        for raw in decoder.feed_frames(chunk):
            if raw.strip() == "data: [DONE]":
                chat_done = True
                continue
            for event in AnthropicSSEDecoder().feed(raw):
                data = event.data
                kind = data.get("type")
                if kind in {"error", "response.error", "response.failed"} or data.get(
                    "error"
                ):
                    error = (
                        data.get("error") or data.get("response", {}).get("error") or {}
                    )
                    if protocol == "messages":
                        raise anthropic_failure_from_error(error)
                    raise _failure(
                        str(error.get("message", "Provider response failed"))
                    )
                if protocol == "messages":
                    if kind == "message_delta":
                        message_reason |= (
                            data.get("delta", {}).get("stop_reason") is not None
                        )
                    terminal |= kind == "message_stop" and message_reason
                elif protocol == "responses":
                    terminal |= kind in {"response.completed", "response.incomplete"}
                elif protocol == "gemini":
                    terminal |= any(
                        candidate.get("finishReason")
                        for candidate in data.get("candidates", [])
                    )
                else:
                    for choice in data.get("choices", []):
                        index = choice.get("index", 0)
                        choices[index] = (
                            choices.get(index, False)
                            or choice.get("finish_reason") is not None
                        )
    if protocol == "chat":
        terminal = chat_done and bool(choices) and all(choices.values())
    if not terminal or decoder.finish_frames():
        raise _failure("Provider response ended without a complete protocol terminal")


def _failure(message: str) -> ExecutionFailure:
    return ExecutionFailure(FailureKind.UPSTREAM, 502, message, False)
