from typing import cast

import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.chat_observations import ChatStreamUsage
from free_claude_code.core.openai_responses import build_responses_chat_request
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest
from free_claude_code.core.stream_events import StreamEvent
from tests.protocol_stream_support import ChatSourceHarness


def _parse_frame(frame: StreamEvent) -> tuple[str, dict[str, object]]:
    return frame.kind, frame.payload


def _parse_frames(frames: list[StreamEvent]) -> list[tuple[str, dict[str, object]]]:
    return [_parse_frame(event) for event in frames]


def _object_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _responses_output(
    request: OpenAIResponsesRequest, *, input_tokens: int
) -> ChatSourceHarness:
    prepared = build_responses_chat_request(
        request, reasoning_replay=ReasoningReplayMode.DISABLED
    )
    return ChatSourceHarness(prepared.tool_adapter, input_tokens=input_tokens)


def _finished_responses_usage(usage: ChatStreamUsage) -> dict[str, object]:
    output = _responses_output(
        OpenAIResponsesRequest.model_validate(
            {"model": "public-model", "input": "Hello"}
        ),
        input_tokens=1,
    )
    frames = output.project(
        [
            *output.start_events(),
            *output.finish_success(stop_reason="stop", usage=usage),
        ]
    )
    final = _object_dict(_parse_frame(frames[-1])[1]["response"])
    return _object_dict(final["usage"])


def test_anthropic_chat_output_preserves_existing_wire_lifecycle() -> None:
    output = ChatSourceHarness(
        model="public-model",
        input_tokens=7,
    )

    frames = [*output.start_events()]
    frames.extend(output.ensure_reasoning_block())
    frames.append(output.emit_reasoning_delta("think"))
    frames.extend(output.ensure_text_block())
    frames.append(output.emit_text_delta("answer"))
    frames.extend(output.close_content_blocks())
    output.ensure_tool_state(0)
    output.register_tool_name(0, "lookup")
    frames.append(output.start_tool_block(0, "call_1", "lookup"))
    frames.extend(output.emit_tool_delta(0, '{"q":"fcc"}'))
    frames.extend(output.close_all_blocks())
    frames.extend(
        output.finish_success(
            stop_reason="tool_use",
            usage=ChatStreamUsage(input_tokens=9, output_tokens=4),
        )
    )

    events = [_parse_frame(frame) for frame in output.project(frames)]
    assert [event_type for event_type, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert str(_object_dict(events[0][1]["message"])["id"]).startswith("msg_")
    assert events[-2][1]["usage"] == {"input_tokens": 9, "output_tokens": 4}
    assert output.accumulated_reasoning == "think"
    assert output.accumulated_text == "answer"
    tool_block = output.tool_block_for_tool_index(0)
    assert tool_block is not None
    assert tool_block.content == '{"q":"fcc"}'


def test_responses_chat_output_emits_one_native_lifecycle_with_exact_usage() -> None:
    output = _responses_output(
        OpenAIResponsesRequest.model_validate(
            {
                "model": "public-model",
                "input": "Hello",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "mcp",
                        "tools": [
                            {
                                "type": "function",
                                "name": "lookup",
                                "parameters": {"type": "object"},
                            }
                        ],
                    }
                ],
            }
        ),
        input_tokens=7,
    )

    frames = [*output.start_events()]
    frames.extend(output.ensure_reasoning_block())
    frames.append(output.emit_reasoning_delta("think"))
    frames.extend(output.ensure_text_block())
    frames.append(output.emit_text_delta("answer"))
    frames.extend(output.close_content_blocks())
    output.ensure_tool_state(0)
    output.register_tool_name(0, "mcp__lookup")
    frames.append(output.start_tool_block(0, "call_1", "mcp__lookup"))
    frames.extend(output.emit_tool_delta(0, '{"q":"fcc"}'))
    frames.extend(output.close_all_blocks())
    frames.extend(
        output.finish_success(
            stop_reason="tool_calls",
            usage=ChatStreamUsage(
                input_tokens=20,
                output_tokens=8,
                cached_tokens=5,
                cache_write_tokens=4,
                reasoning_tokens=3,
            ),
        )
    )

    events = _parse_frames(list(output.project(frames)))
    event_types = [event_type for event_type, _ in events]
    assert event_types.count("response.created") == 1
    assert event_types.count("response.completed") == 1
    assert not any(event_type.startswith("message_") for event_type in event_types)
    assert event_types.index("response.reasoning_text.delta") < event_types.index(
        "response.output_text.delta"
    )
    assert event_types.index("response.output_text.delta") < event_types.index(
        "response.function_call_arguments.done"
    )

    final = _object_dict(events[-1][1]["response"])
    assert final["model"] == "public-model"
    assert final["status"] == "completed"
    assert final["usage"] == {
        "input_tokens": 20,
        "input_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 4},
        "output_tokens": 8,
        "output_tokens_details": {"reasoning_tokens": 3},
        "total_tokens": 28,
    }
    final_output = final["output"]
    assert isinstance(final_output, list)
    tool = next(
        item
        for value in final_output
        if isinstance(value, dict)
        and (item := _object_dict(value))["type"] == "function_call"
    )
    assert tool == {
        "id": tool["id"],
        "type": "function_call",
        "status": "completed",
        "call_id": "call_1",
        "name": "lookup",
        "namespace": "mcp",
        "arguments": '{"q":"fcc"}',
    }


def test_responses_chat_output_preserves_explicit_zero_cache_write() -> None:
    usage = _finished_responses_usage(
        ChatStreamUsage(
            input_tokens=20,
            output_tokens=8,
            cached_tokens=5,
            cache_write_tokens=0,
        )
    )

    assert usage["input_tokens_details"] == {
        "cached_tokens": 5,
        "cache_write_tokens": 0,
    }


@pytest.mark.parametrize(
    "cache_write_tokens",
    [None, -1, 16, True, "4"],
    ids=["absent", "negative", "over-remaining-total", "boolean", "string"],
)
def test_responses_chat_output_omits_untrustworthy_cache_write_without_losing_read(
    cache_write_tokens: object,
) -> None:
    usage = _finished_responses_usage(
        ChatStreamUsage(
            input_tokens=20,
            output_tokens=8,
            cached_tokens=5,
            cache_write_tokens=cast(int | None, cache_write_tokens),
        )
    )

    assert usage == {
        "input_tokens": 20,
        "input_tokens_details": {"cached_tokens": 5},
        "output_tokens": 8,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 28,
    }


def test_responses_chat_output_ignores_overflowing_read_without_losing_write() -> None:
    usage = _finished_responses_usage(
        ChatStreamUsage(
            input_tokens=20,
            output_tokens=8,
            cached_tokens=21,
            cache_write_tokens=5,
        )
    )

    assert usage == {
        "input_tokens": 20,
        "input_tokens_details": {
            "cached_tokens": 0,
            "cache_write_tokens": 5,
        },
        "output_tokens": 8,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 28,
    }


def test_responses_chat_output_preserves_options_on_incomplete_terminal() -> None:
    output = _responses_output(
        OpenAIResponsesRequest.model_validate(
            {
                "model": "public-model",
                "input": "Hello",
                "temperature": 0.25,
                "top_p": 0.75,
                "max_output_tokens": 12,
            }
        ),
        input_tokens=1,
    )

    frames = [*output.start_events(), *output.ensure_text_block()]
    frames.append(output.emit_text_delta("partial"))
    frames.extend(
        output.finish_success(
            stop_reason="max_tokens",
            usage=ChatStreamUsage(input_tokens=4, output_tokens=12),
        )
    )

    events = _parse_frames(list(output.project(frames)))
    assert [event_type for event_type, _ in events].count("response.created") == 1
    assert [event_type for event_type, _ in events].count("response.incomplete") == 1
    final = _object_dict(events[-1][1]["response"])
    assert final["status"] == "incomplete"
    assert final["incomplete_details"] == {"reason": "max_output_tokens"}
    assert final["parallel_tool_calls"] is True
    assert final["tool_choice"] == "auto"
    assert final["temperature"] == 0.25
    assert final["top_p"] == 0.75
    assert final["max_output_tokens"] == 12


def test_responses_chat_output_preserves_custom_tool_free_form_input() -> None:
    output = _responses_output(
        OpenAIResponsesRequest.model_validate(
            {
                "model": "public-model",
                "input": "Patch the file",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "format": {"type": "text"},
                    }
                ],
            }
        ),
        input_tokens=1,
    )

    frames = [*output.start_events()]
    frames.append(output.start_tool_block(0, "call_patch", "apply_patch"))
    frames.extend(output.emit_tool_delta(0, "*** Begin Patch\n*** End Patch"))
    frames.extend(
        output.finish_success(
            stop_reason="tool_calls",
            usage=ChatStreamUsage(input_tokens=3, output_tokens=2),
        )
    )

    events = _parse_frames(list(output.project(frames)))
    final = _object_dict(events[-1][1]["response"])
    final_output = final["output"]
    assert isinstance(final_output, list)
    item_id = _object_dict(final_output[0])["id"]
    assert isinstance(item_id, str) and item_id.startswith("ctc_")
    assert [kind for kind, _ in events] == [
        "response.created",
        "response.output_item.added",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event["item_id"] for _, event in events if "item_id" in event] == [
        item_id,
        item_id,
    ]
    assert [
        _object_dict(event["item"])["id"] for _, event in events if "item" in event
    ] == [item_id, item_id]
    assert final_output == [
        {
            "id": item_id,
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": "call_patch",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch",
        }
    ]


def test_responses_chat_output_completes_malformed_function_call_once() -> None:
    output = _responses_output(
        OpenAIResponsesRequest.model_validate(
            {
                "model": "public-model",
                "input": "Call lookup",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        ),
        input_tokens=1,
    )

    frames = [*output.start_events()]
    frames.append(output.start_tool_block(0, "call_bad", "lookup"))
    frames.extend(output.emit_tool_delta(0, '{"q":'))
    frames.extend(
        output.finish_success(
            stop_reason="tool_calls",
            usage=ChatStreamUsage(input_tokens=3, output_tokens=1),
        )
    )
    frames.extend(
        output.finish_success(
            stop_reason="tool_calls",
            usage=ChatStreamUsage(input_tokens=3, output_tokens=1),
        )
    )

    events = _parse_frames(list(output.project(frames)))
    event_types = [event_type for event_type, _ in events]
    assert event_types.count("response.completed") == 1
    assert "response.failed" not in event_types
    final = _object_dict(events[-1][1]["response"])
    assert final["status"] == "completed"
    assert final["error"] is None
    final_output = final["output"]
    assert isinstance(final_output, list)
    assert len(final_output) == 1
    item = _object_dict(final_output[0])
    assert item["arguments"] == '{"q":'
    assert item["call_id"] == "call_bad"


def test_projection_failure_keeps_checkpoint_equal_to_delivered_prefix() -> None:
    from free_claude_code.core.history_replay import ReplayOrigin
    from free_claude_code.core.openai_responses.errors import ResponsesConversionError
    from free_claude_code.core.recovery import AttemptFailure
    from free_claude_code.core.stream_observations import (
        ChatObservation,
        DecodedStreamEvent,
    )

    source = _responses_output(
        OpenAIResponsesRequest(
            model="public",
            input="hi",
            tools=[
                {
                    "type": "namespace",
                    "name": namespace,
                    "tools": [
                        {
                            "type": "function",
                            "name": "lookup",
                            "parameters": {"type": "object"},
                        }
                    ],
                }
                for namespace in ("a", "b")
            ],
        ),
        input_tokens=1,
    )
    changes = (
        *source.start_events(),
        *source.ensure_text_block(),
        source.emit_text_delta("Visible prefix."),
        *source.close_content_blocks(),
        source.start_tool_block(0, "call", "lookup"),
    )
    delivered = []
    with pytest.raises((ResponsesConversionError, AttemptFailure), match="Ambiguous"):
        for event in source.writer.feed(
            DecodedStreamEvent(
                ReplayOrigin("test", "chat", "", "", "m"),
                StreamEvent("chat.completion.chunk", {}),
                observation=ChatObservation(changes, source.tools),
            )
        ):
            assert event.kind != "response.completed"
            delivered.append(event)
    text = "".join(
        event.payload["delta"]
        for event in delivered
        if event.kind == "response.output_text.delta"
    )
    assert text == source.writer.checkpoint.text == "Visible prefix."
