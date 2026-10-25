"""Direct Gemini presentation of the shared Chat stream assembler."""

from copy import deepcopy

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.gemini.replay import CallReplay
from free_claude_code.core.gemini.response import (
    GeminiOutput,
    conversion_failure,
    function_arguments,
)
from free_claude_code.core.history_replay import ReplayRecord, decode_replay, is_replay
from free_claude_code.core.json_types import JsonObject

from .stream_output import ChatStreamOutput, ChatStreamUsage, ChatToolState


class GeminiChatStreamOutput(ChatStreamOutput):
    pad_empty = False

    def __init__(
        self,
        *,
        response_id: str,
        model: str,
        include_thoughts: bool,
        tool_names: frozenset[str],
        input_tokens: int,
    ) -> None:
        super().__init__(input_tokens=input_tokens)
        self._gemini = GeminiOutput(
            response_id, model, include_thoughts=include_thoughts
        )
        self._tool_names = tool_names

    def capture_reasoning_record(self, native: JsonObject) -> bool:
        assert self.replay_origin is not None
        self._gemini.capture(ReplayRecord(self.replay_origin, deepcopy(native)))
        return True

    def _start_events(self) -> list[str]:
        return []

    def _start_reasoning_block(self, source: int) -> list[str]:
        return []

    def _emit_reasoning_delta(
        self, content: str, *, summary: bool = False, source: int = 0
    ) -> str:
        return "".join(self._gemini.part({"text": content, "thought": True}))

    def _emit_opaque_reasoning(self, data: str) -> list[str]:
        if is_replay(data):
            self._gemini.capture(decode_replay(data))
        else:
            raise conversion_failure(
                "Upstream standalone Chat signature lacks a replayable native reasoning record"
            )
        return []

    def _attach_reasoning_replay(self, data: str, source: int) -> list[str]:
        return self._emit_opaque_reasoning(data)

    def _stop_reasoning_block(self, source: int) -> list[str]:
        return []

    def _start_text_block(self) -> list[str]:
        return []

    def _emit_text_delta(self, content: str) -> str:
        return "".join(self._gemini.part({"text": content}))

    def _stop_text_block(self) -> list[str]:
        return []

    def _start_tool_block(self, tool_index: int, state: ChatToolState) -> str:
        return ""

    def _emit_tool_delta(
        self, tool_index: int, state: ChatToolState, partial_json: str
    ) -> str:
        return ""

    def _stop_tool_block(self, tool_index: int, state: ChatToolState) -> list[str]:
        if state.name not in self._tool_names:
            raise conversion_failure(
                f"Upstream called an undeclared function: {state.name}"
            )
        arguments = function_arguments(state.content)
        if state.extra_content:
            assert self.replay_origin is not None
            self._gemini.calls.append(
                CallReplay(
                    self.replay_origin, state.tool_id, deepcopy(state.extra_content)
                )
            )
        return self._gemini.part(
            {
                "functionCall": {
                    "id": state.tool_id,
                    "name": state.name,
                    "args": arguments,
                }
            }
        )

    def _finish_success(self, *, stop_reason: str, usage: ChatStreamUsage) -> list[str]:
        reported = usage.reported
        metadata: JsonObject = {}
        for source, target in (
            ("prompt_tokens", "promptTokenCount"),
            ("completion_tokens", "candidatesTokenCount"),
            ("total_tokens", "totalTokenCount"),
            ("cached_tokens", "cachedContentTokenCount"),
            ("reasoning_tokens", "thoughtsTokenCount"),
        ):
            if source in reported:
                metadata[target] = reported[source]
        if "completion_tokens" in reported and "reasoning_tokens" in reported:
            metadata["candidatesTokenCount"] = max(
                0, reported["completion_tokens"] - reported["reasoning_tokens"]
            )
        elif self.accumulated_reasoning or self._gemini.records:
            metadata.pop("candidatesTokenCount", None)
        reason = (
            "MAX_TOKENS"
            if usage.finish_reason == "length"
            else "SAFETY"
            if usage.finish_reason == "content_filter"
            else "STOP"
        )
        return self._gemini.finish(reason, metadata)

    def _finish_failure(self, failure: ExecutionFailure) -> list[str]:
        return self._gemini.failure(failure.status_code, failure.message)
