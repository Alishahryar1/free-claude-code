"""Consume provider packets through the real public delivery boundary."""

from collections.abc import AsyncIterator
from typing import Literal

from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    anthropic_sse_streaming_response,
    openai_responses_sse_streaming_response,
)
from free_claude_code.core.stream_output import StreamOutput


async def delivered(
    body: AsyncIterator[StreamOutput], wire: Literal["messages", "responses"]
) -> str:
    def fail(error: BaseException):
        raise error

    if wire == "messages":
        response = await anthropic_sse_streaming_response(
            body, pre_start_error_response=fail, request_id="public-test"
        )
    else:
        response = await openai_responses_sse_streaming_response(
            body, headers={}, pre_start_error_response=fail, request_id="public-test"
        )
    assert isinstance(response, ManagedStreamingResponse)
    try:
        return "".join([str(frame) async for frame in response.body_iterator])
    finally:
        await response.aclose()
