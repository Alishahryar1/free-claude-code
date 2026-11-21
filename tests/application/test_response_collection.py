import pytest

from free_claude_code.application.response_collection import collect_response
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.stream_delivery import current_stream_delivery
from free_claude_code.core.stream_output import StreamOutput


@pytest.mark.asyncio
async def test_private_replacement_discards_previous_attempt():
    async def body():
        state = current_stream_delivery()
        assert state is not None
        state.begin_attempt()
        yield StreamOutput(
            ('data: {"choices":[{"index":0,"delta":{"content":"abandoned"}}]}\n\n',)
        )
        assert not state.content_released
        state.begin_attempt()
        yield StreamOutput(
            (
                'data: {"choices":[{"index":0,"delta":{"content":"replacement"},"finish_reason":"stop"}]}\n\n',
                "data: [DONE]\n\n",
            )
        )

    frames = await collect_response(body(), protocol="chat")
    assert "abandoned" not in "".join(frames)
    assert "replacement" in "".join(frames)


@pytest.mark.asyncio
async def test_eof_after_new_attempt_does_not_return_old_draft():
    async def body():
        state = current_stream_delivery()
        assert state is not None
        state.begin_attempt()
        yield StreamOutput(
            (
                'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n',
                'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            )
        )
        state.begin_attempt()

    with pytest.raises(ExecutionFailure, match="without"):
        await collect_response(body(), protocol="messages")
