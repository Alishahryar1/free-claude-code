"""Output limits must not publish an unfinished client invocation or generate again."""

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from tests.providers.test_history_transports import _harness
from tests.providers.test_openai_responses_transport import _completed_response


def _limited_call(protocol: str) -> list[dict]:
    if protocol == "chat":
        return [
            {
                "id": "chat_test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "unfinished_call",
                                    "type": "function",
                                    "function": {
                                        "name": "Write",
                                        "arguments": '{"content":"partial',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "length",
                    }
                ],
            }
        ]
    item = {
        "id": "fc_partial",
        "type": "function_call",
        "call_id": "unfinished_call",
        "name": "Write",
        "arguments": "",
        "status": "in_progress",
    }
    response = _completed_response()
    return [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                **response,
                "status": "in_progress",
                "output": [],
            },
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": item,
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "output_index": 0,
            "item_id": item["id"],
            "delta": '{"content":"partial',
        },
        {
            "type": "response.incomplete",
            "sequence_number": 3,
            "response": {
                **response,
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {**item, "arguments": '{"content":"partial', "status": "incomplete"}
                ],
            },
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_output_limit_keeps_unfinished_tool_private_without_another_call(
    protocol, wire
):
    async with _harness(protocol, lambda _: (200, _limited_call(protocol))) as (
        send,
        bodies,
        _,
    ):
        body = "".join(
            [
                event
                async for event in send(wire, [{"role": "user", "content": "Write it"}])
            ]
        )
    assert len(bodies) == 1
    assert "unfinished_call" not in body
    events = parse_sse_text(body)
    if wire == "messages":
        delta = next(event.data for event in events if event.event == "message_delta")
        assert delta["delta"]["stop_reason"] == "max_tokens"
        assert events[-1].event == "message_stop"
    else:
        assert events[-1].event == "response.incomplete"
        assert (
            events[-1].data["response"]["incomplete_details"]["reason"]
            == "max_output_tokens"
        )
