"""Chat-client writer for semantic events from a translated upstream."""

from free_claude_code.core.chat_completions.response import ChatOutput, chat_frame
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.openai_responses import openai_error_from_failure

from .stream_output import ChatStreamOutput, ChatStreamUsage, ChatToolState


class NativeChatStreamOutput(ChatStreamOutput):
    pad_empty = False

    def __init__(
        self, *, model: str, input_tokens: int = 0, include_usage: bool = False
    ) -> None:
        super().__init__(input_tokens=input_tokens)
        self.chat = ChatOutput(model, include_usage=include_usage)

    def _start_events(self) -> list[str]:
        return self.chat.start()

    def _start_reasoning_block(self, source: int) -> list[str]:
        return []

    def _emit_reasoning_delta(
        self,
        content: str,
        *,
        summary: bool = False,
        source: int = 0,
        replay_owner: int | None = None,
    ) -> str:
        return "".join(self.chat.thought(content, summary=summary))

    def _emit_opaque_reasoning(self, data: str) -> list[str]:
        return self.chat.replay(data)

    def _attach_reasoning_replay(self, data: str, source: int) -> list[str]:
        return self.chat.replay(data)

    def _stop_reasoning_block(self, source: int) -> list[str]:
        return []

    def _start_text_block(self) -> list[str]:
        return []

    def _emit_text_delta(self, content: str) -> str:
        return "".join(self.chat.text(content))

    def _stop_text_block(self) -> list[str]:
        return []

    def _start_tool_block(self, tool_index: int, state: ChatToolState) -> str:
        return "".join(self.chat.tool(tool_index, state.tool_id, state.name, ""))

    def _emit_tool_delta(
        self, tool_index: int, state: ChatToolState, partial_json: str
    ) -> str:
        return "".join(
            self.chat.delta(
                {
                    "tool_calls": [
                        {"index": tool_index, "function": {"arguments": partial_json}}
                    ]
                }
            )
        )

    def _stop_tool_block(self, tool_index: int, state: ChatToolState) -> list[str]:
        return []

    def _finish_success(self, *, stop_reason: str, usage: ChatStreamUsage) -> list[str]:
        reason = {
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "refusal": "content_filter",
        }.get(stop_reason, "stop")
        return self.chat.finish(
            reason,
            {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
                "prompt_tokens_details": {"cached_tokens": usage.cached_tokens},
                "completion_tokens_details": {
                    "reasoning_tokens": usage.reasoning_tokens
                },
            },
        )

    def _finish_failure(self, failure: ExecutionFailure) -> list[str]:
        return [chat_frame({"error": openai_error_from_failure(failure)})]
