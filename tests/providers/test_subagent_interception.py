import copy
import json

import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    build_responses_chat_request,
)
from free_claude_code.providers.openai_chat.stream_output import (
    AnthropicChatStreamOutput,
    ChatStreamOutput,
    ChatStreamUsage,
    ResponsesChatStreamOutput,
)
from free_claude_code.providers.openai_chat.tool_calls import (
    OpenAIToolCallAssembler,
    iter_heuristic_tool_use_events,
)


@pytest.fixture(params=["messages", "responses"])
def output(request) -> ChatStreamOutput:
    if request.param == "messages":
        return AnthropicChatStreamOutput(
            message_id="msg_test", model="test-model", input_tokens=1
        )
    prepared = build_responses_chat_request(
        OpenAIResponsesRequest(model="test-model", input="hello"),
        reasoning_replay=ReasoningReplayMode.DISABLED,
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


@pytest.mark.parametrize(
    "arguments",
    [
        {"run_in_background": True},
        {"run_in_background": False},
        {"run_in_background": None},
        {"prompt": "inspect"},
    ],
)
def test_heuristic_task_preserves_arguments(output, arguments):
    tool_use = {
        "type": "tool_use",
        "id": "call_task",
        "name": "Task",
        "input": {**arguments, "nested": {"items": [1, 2]}},
    }
    original = copy.deepcopy(tool_use)
    frames = list(iter_heuristic_tool_use_events(output, tool_use))
    assert json.loads("".join(_argument_deltas(frames))) == original["input"]
    assert tool_use == original


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
    assembler = OpenAIToolCallAssembler(reserved_tool_ids=())
    frames = output.start_events()
    split = len(arguments) // 2
    for part, tool_name in [(arguments[:split], name), (arguments[split:], None)]:
        emitted = list(
            assembler.process_tool_call(
                {
                    "index": 0,
                    "id": "call_task",
                    "function": {"name": tool_name, "arguments": part},
                },
                output,
            )
        )
        if isinstance(output, AnthropicChatStreamOutput):
            assert _argument_deltas(emitted) == ([part] if part else [])
        frames.extend(emitted)
    frames.extend(
        output.finish_success(
            stop_reason="tool_use",
            usage=ChatStreamUsage(input_tokens=1, output_tokens=1),
        )
    )
    assert output.tool_states[0].content == arguments
    assert output.tool_states[0].tool_id == "call_task"
    events = parse_sse_text("".join(frames))
    if isinstance(output, ResponsesChatStreamOutput):
        valid = arguments in (
            '{"run_in_background":true,"prompt":"inspect"}',
            "{}",
            '{"run_in_background":null}',
        )
        assert any(e.event == "response.completed" for e in events) == valid
        assert any(e.event == "response.failed" for e in events) != valid
        if valid:
            assert json.loads("".join(_argument_deltas(frames))) == json.loads(
                arguments
            )
        else:
            assert not _argument_deltas(frames)
    else:
        assert events[-1].event == "message_stop"
        assert "".join(_argument_deltas(frames)) == arguments


def test_task_argument_aliases_are_restored_recursively(output):
    assembler = OpenAIToolCallAssembler(reserved_tool_ids=())
    buffers = {}
    frames = []
    for name, part in [
        ("Task", '{"run_in_background":true,"wire_prompt":"inspect",'),
        (None, '"nested":[{"wire_prompt":"child"}]}'),
    ]:
        frames.extend(
            assembler.process_tool_call(
                {
                    "index": 0,
                    "id": "call_task",
                    "function": {"name": name, "arguments": part},
                },
                output,
                tool_argument_aliases={"Task": {"wire_prompt": "prompt"}},
                tool_argument_alias_buffers=buffers,
            )
        )
    frames.extend(
        assembler.flush_tool_argument_alias_buffers(
            output, {"Task": {"wire_prompt": "prompt"}}, buffers
        )
    )
    frames.extend(output.close_all_blocks())
    assert json.loads("".join(_argument_deltas(frames))) == {
        "run_in_background": True,
        "prompt": "inspect",
        "nested": [{"prompt": "child"}],
    }
    assert not buffers
