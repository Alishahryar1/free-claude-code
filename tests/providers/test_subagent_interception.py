import json

import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.chat_completions.tool_calls import (
    restore_tool_argument_aliases,
)
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    build_responses_chat_request,
)
from free_claude_code.providers.openai_chat.stream_output import (
    AnthropicChatStreamOutput,
    ChatStreamOutput,
    ChatStreamUsage,
    CompletedToolCall,
    ResponsesChatStreamOutput,
)


@pytest.fixture(params=["messages", "responses"])
def output(request) -> ChatStreamOutput:
    if request.param == "messages":
        return AnthropicChatStreamOutput(
            message_id="msg_test", model="test-model", input_tokens=1
        )
    prepared = build_responses_chat_request(
        OpenAIResponsesRequest(model="test-model", input="hello"),
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    )
    return ResponsesChatStreamOutput(prepared.tool_adapter, input_tokens=1)


def _argument_deltas(frames: list[str]) -> list[str]:
    parts = []
    for event in parse_sse_text("".join(frames)):
        if event.event == "response.function_call_arguments.delta":
            parts.append(event.data["delta"])
        elif event.data.get("delta", {}).get("type") == "input_json_delta":
            parts.append(event.data["delta"]["partial_json"])
    return parts


@pytest.mark.parametrize("name", ["Task", "ordinary_tool"])
@pytest.mark.parametrize(
    "arguments",
    [
        '{"run_in_background":true,"prompt":"inspect"}',
        "{}",
        '{"run_in_background":null}',
        "[]",
        "not json",
        '{"broken":',
        "",
    ],
)
def test_tool_arguments_stream_without_name_specific_rewrites(output, name, arguments):
    frames = output.start_events()
    frames.extend(
        output.emit_complete_tool(0, CompletedToolCall("call_task", name, arguments))
    )
    frames.extend(
        output.finish_success(
            stop_reason="tool_use",
            usage=ChatStreamUsage(input_tokens=1, output_tokens=1),
        )
    )
    events = parse_sse_text("".join(frames))
    assert events[-1].event == (
        "response.completed"
        if isinstance(output, ResponsesChatStreamOutput)
        else "message_stop"
    )
    assert "".join(_argument_deltas(frames)) == arguments


def test_task_argument_aliases_are_restored_recursively(output):
    arguments = restore_tool_argument_aliases(
        '{"run_in_background":true,"wire_prompt":"inspect","nested":[{"wire_prompt":"child"}]}',
        {"wire_prompt": "prompt"},
    )
    assert arguments is not None
    frames = output.emit_complete_tool(
        0, CompletedToolCall("call_task", "Task", arguments)
    )
    assert json.loads("".join(_argument_deltas(frames))) == {
        "run_in_background": True,
        "prompt": "inspect",
        "nested": [{"prompt": "child"}],
    }
