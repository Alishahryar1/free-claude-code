import json

import pytest

from free_claude_code.core.chat_observations import ChatChange, ChatStreamUsage
from free_claude_code.providers.openai_chat.source_state import ChatSourceState
from free_claude_code.providers.openai_chat.tool_calls import (
    OpenAIToolCallAssembler,
)


@pytest.fixture
def output() -> ChatSourceState:
    return ChatSourceState(input_tokens=1)


def _argument_deltas(frames: list[ChatChange]) -> list[str]:
    return [change.text for change in frames if change.kind == "tool.delta"]


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
    assert frames[-1].kind == "complete"
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
