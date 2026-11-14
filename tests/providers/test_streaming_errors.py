"""Tests for streaming error handling in providers/nvidia_nim/client.py."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import httpx2
import openai
import pytest

from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    _first_chunk_streaming_response,
)
from free_claude_code.application.response_collection import collect_response
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import (
    UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS,
    ProviderOperationKind,
)
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.openai_chat.stream_output import (
    AnthropicChatStreamOutput,
)
from free_claude_code.providers.openai_chat.transport import (
    _OpenAIChatStreamRunner,
    _reserved_anthropic_tool_ids,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    REASONING_OFF,
    SDKStreamDouble,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    sdk_chat_chunk,
)


class AsyncStreamMock(SDKStreamDouble):
    """Async iterable mock that yields chunks then optionally raises."""

    def __init__(self, chunks, error=None):
        self._chunks = chunks
        self._error = error
        super().__init__(self._aiter())

    async def _aiter(self):
        for chunk in self._chunks:
            yield chunk
        if self._error:
            raise self._error


class ClosableAsyncStreamMock(AsyncStreamMock):
    """Async stream mock that records cleanup."""

    def __init__(self, chunks, error=None, *, close_error=None):
        super().__init__(chunks, error=error)
        self.closed = False
        self.close_calls = 0
        self._close_error = close_error

    async def aclose(self):
        self.close_calls += 1
        self.closed = True
        if self._close_error is not None:
            raise self._close_error

    async def close(self):
        await super().close()
        await self.aclose()


class BlockingClosableAsyncStreamMock(SDKStreamDouble):
    """Async stream that blocks until its consumer is cancelled."""

    def __init__(self, *, close_error=None):
        self.entered = asyncio.Event()
        self.close_calls = 0
        self._close_error = close_error
        super().__init__(self._aiter(), close=self.aclose)

    async def _aiter(self):
        self.entered.set()
        await asyncio.Event().wait()
        yield

    async def aclose(self):
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


def _make_provider():
    """Create a provider instance for testing."""
    config = make_provider_config(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
    )
    return NvidiaNimProvider(
        config,
        admission=immediate_admission(),
    )


def _make_anthropic_output() -> AnthropicChatStreamOutput:
    return AnthropicChatStreamOutput(
        message_id="msg_test",
        model="test-model",
        input_tokens=0,
    )


def _make_request(model: str = "test-model", stream: bool = True, **overrides: object):
    """Create a concrete request matching the original streaming-test defaults."""
    request_overrides: dict[str, object] = {
        "messages": [],
        "max_tokens": 4096,
        "temperature": None,
        "top_p": None,
        "system": None,
        "tools": None,
        "extra_body": None,
        "thinking": None,
        "stream": stream,
    }
    request_overrides.update(overrides)
    return make_messages_request(model, **request_overrides)


def _make_stream_runner(
    provider: NvidiaNimProvider,
    *,
    request=None,
    request_id: str | None = None,
) -> _OpenAIChatStreamRunner:
    concrete_request = request or _make_request()
    return _OpenAIChatStreamRunner(
        provider._chat,
        body=provider._chat._build_request_body(concrete_request),
        tool_names=OpenAIToolNameCodec.from_request(concrete_request),
        reserved_tool_ids=_reserved_anthropic_tool_ids(concrete_request),
        output_factory=_make_anthropic_output,
        input_tokens=0,
        request_id=request_id,
        response_model=concrete_request.model,
        reasoning=DEFAULT_REASONING_POLICY,
    )


def _make_chunk(
    content=None, finish_reason=None, tool_calls=None, reasoning_content=None
):
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

    calls = [
        tool
        if isinstance(tool, dict)
        else {
            "index": tool.index,
            "id": tool.id,
            "function": {
                "name": tool.function.name,
                "arguments": tool.function.arguments,
            },
        }
        for tool in tool_calls or []
    ]
    return ChatCompletionChunk.model_construct(
        id="chat_test",
        object="chat.completion.chunk",
        created=0,
        model="test-model",
        choices=[
            Choice.model_construct(
                index=0,
                delta=ChoiceDelta.model_validate(
                    {
                        "content": content,
                        "tool_calls": calls or None,
                        "reasoning_content": reasoning_content,
                    }
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _make_context_length_bad_request() -> openai.BadRequestError:
    response = httpx2.Response(
        status_code=400,
        request=httpx2.Request(
            "POST", "https://test.api.nvidia.com/v1/chat/completions"
        ),
    )
    return openai.BadRequestError(
        "maximum context reached",
        response=response,
        body={"error": {"code": "context_length_exceeded"}},
    )


def _make_usage_chunk(*, prompt_tokens: int, completion_tokens: int):
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletionChunk

    return ChatCompletionChunk.model_construct(
        id="chat_test",
        object="chat.completion.chunk",
        created=0,
        model="test-model",
        choices=[],
        usage=CompletionUsage.model_construct(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


def _make_tool_calls_chunk(
    *, name: str | None, arguments: str, tool_id: str | None, index: int = 0
):
    """Single OpenAI-style tool_calls delta (starts a native streamed tool block)."""
    tc = MagicMock()
    tc.index = index
    tc.id = tool_id
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments
    tc.function = fn
    return _make_chunk(tool_calls=[tc])


async def _collect_stream(
    provider,
    request,
    *,
    reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
):
    """Collect all SSE events from a stream."""
    from free_claude_code.application.response_collection import collect_response

    return list(
        await collect_response(
            provider.stream_messages(request, reasoning=reasoning), protocol="messages"
        )
    )


async def _collect_stream_error(provider, request, **kwargs) -> ExecutionFailure:
    with pytest.raises(ExecutionFailure) as exc_info:
        [
            frame
            async for e in provider.stream_messages(request, **kwargs)
            for frame in e.frames
        ]
    return exc_info.value


async def _collect_stream_and_error(
    provider, request, **kwargs
) -> tuple[list[str], ExecutionFailure]:
    events: list[str] = []

    def reject(error):
        raise error

    response = None
    with pytest.raises(ExecutionFailure) as exc_info:
        response = await _first_chunk_streaming_response(
            provider.stream_messages(request, **kwargs),
            wire_api="messages",
            headers={},
            pre_start_error_response=reject,
            terminal_frame=None,
            terminal_failure_observer=None,
        )
        assert isinstance(response, ManagedStreamingResponse)
        try:
            async for event in response.body_iterator:
                events.extend((str(event),))
        finally:
            await response.aclose()
    return events, exc_info.value


def _tool_use_starts(events: list[str]) -> list[dict]:
    return [
        event.data["content_block"]
        for event in parse_sse_text("".join(events))
        if event.event == "content_block_start"
        and event.data.get("content_block", {}).get("type") == "tool_use"
    ]


def _assert_no_content_deltas_after_error_text(
    events: list[str], error_substr: str
) -> None:
    """After the error text delta, only block close + message tail events may follow."""
    parsed = parse_sse_text("".join(events))
    first_error_idx = None
    for i, ev in enumerate(parsed):
        if ev.event != "content_block_delta":
            continue
        delta = ev.data.get("delta", {})
        if delta.get("type") == "text_delta" and error_substr in str(
            delta.get("text", "")
        ):
            first_error_idx = i
            break
    assert first_error_idx is not None, (error_substr, "".join(events))
    for ev in parsed[first_error_idx + 1 :]:
        assert ev.event in ("content_block_stop", "message_delta", "message_stop"), (
            ev.event,
            ev.data,
        )


def _assert_error_not_in_text_deltas_after_tool(
    events: list[str], error_substr: str
) -> None:
    """Transport errors after a native tool call must not use assistant text_delta (issue #206)."""
    blob = "".join(events)
    for ev in parse_sse_text(blob):
        if ev.event != "content_block_delta":
            continue
        delta = ev.data.get("delta", {})
        if delta.get("type") == "text_delta" and error_substr in str(
            delta.get("text", "")
        ):
            raise AssertionError(
                f"error leaked as text_delta after tool_use: {ev.data!r} full={blob!r}"
            )


class TestStreamingExceptionHandling:
    @pytest.mark.asyncio
    async def test_stream_normalization_failure_closes_raw_stream(self):
        provider = _make_provider()
        stream = ClosableAsyncStreamMock([])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream,
            ),
            patch.object(
                provider._behavior,
                "normalize_stream",
                side_effect=ValueError("invalid stream wrapper"),
            ),
        ):
            await _collect_stream_error(provider, _make_request())

        assert stream.closed

    """Tests for error paths during stream_messages."""

    @pytest.mark.asyncio
    async def test_pre_start_api_error_raises_provider_error(self):
        """Before holdback commit, provider failures raise for API-level non-200."""
        provider = _make_provider()
        request = _make_request()

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API failed"),
            ),
        ):
            error = await _collect_stream_error(provider, request)

        assert "API failed" in error.message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wire_api", ["messages", "responses"])
    async def test_context_finish_reason_is_terminal_before_output(
        self,
        wire_api: str,
    ) -> None:
        provider = _make_provider()
        stream = AsyncStreamMock(
            [_make_chunk(finish_reason=" Model_Context_Window_Exceeded ")]
        )

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream,
            ) as create,
            pytest.raises(ExecutionFailure) as exc_info,
        ):
            if wire_api == "messages":
                await _collect_stream(provider, _make_request())
            else:
                [
                    frame
                    async for event in provider.stream_responses(
                        OpenAIResponsesRequest(model="test-model", input="hello"),
                        request_id="req_context",
                        response_model="public-model",
                    )
                    for frame in event.frames
                ]

        assert create.await_count == 1
        assert exc_info.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_read_timeout_with_empty_message_raises_fallback(self):
        """ReadTimeout(TimeoutError()) should raise a non-empty timeout message."""
        provider = _make_provider()
        request = _make_request()

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=httpx.ReadTimeout(""),
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            error = await _collect_stream_error(
                provider,
                request,
                request_id="req_timeout123",
            )

        assert "timed out after" in error.message
        assert "Request ID: req_timeout123" in error.message

    @pytest.mark.asyncio
    async def test_error_after_precommit_partial_content_raises(self):
        """Precommit partial text is discarded so the API can return non-200."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(content="Hello ")
        stream_mock = AsyncStreamMock([chunk1], error=RuntimeError("Connection lost"))

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
        ):
            error = await _collect_stream_error(provider, request)

        assert "Connection lost" in error.message

    @pytest.mark.asyncio
    async def test_error_after_native_tool_call_keeps_tool_incomplete_and_raises(self):
        """A failure never marks an unfinished tool complete for the API."""
        provider = _make_provider()
        request = _make_request()
        tool_chunk = _make_tool_calls_chunk(
            name="echo_smoke", arguments="{}", tool_id="call_206", index=0
        )
        stream_mock = AsyncStreamMock(
            [tool_chunk], error=RuntimeError("Connection lost after tool")
        )
        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
        ):
            events, error = await _collect_stream_and_error(provider, request)
        event_text = "".join(events)
        parsed = parse_sse_text(event_text)
        assert "tool_use" not in event_text
        assert not any(event.event == "content_block_stop" for event in parsed)
        assert "Connection lost after tool" in error.message
        assert "Connection lost after tool" not in event_text
        assert "event: error\n" not in event_text
        assert "message_stop" not in event_text
        _assert_error_not_in_text_deltas_after_tool(
            events, "Connection lost after tool"
        )

    @pytest.mark.asyncio
    async def test_empty_response_gets_space(self):
        """Empty response with no text/tools gets a single space text block."""
        provider = _make_provider()
        request = _make_request()

        empty_chunk = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([empty_chunk])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert '"text_delta"' in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_upstream_completion_tokens_null_emits_int_usage(self):
        """NIM/GLM may send usage.completion_tokens=null; final SSE must not use JSON null."""
        provider = _make_provider()
        request = _make_request()

        delta = SimpleNamespace(
            content="hello",
            tool_calls=None,
            reasoning_content=None,
        )
        choice = SimpleNamespace(delta=delta, finish_reason="stop")
        usage = SimpleNamespace(completion_tokens=None, prompt_tokens=None)
        chunk = sdk_chat_chunk(SimpleNamespace(choices=[choice], usage=usage))
        stream_mock = AsyncStreamMock([chunk])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        delta_events = [e for e in parsed if e.event == "message_delta"]
        assert len(delta_events) == 1
        usage_out = delta_events[0].data.get("usage", {})
        assert isinstance(usage_out.get("output_tokens"), int)
        assert usage_out["output_tokens"] is not None
        assert '"output_tokens": null' not in "".join(events)

    @pytest.mark.asyncio
    async def test_reasoning_only_stream_emits_placeholder_text(self):
        """When the model streams only ``reasoning_content`` (no ``content``), add text block.

        NIM / some templates may emit no main ``content``; a minimal text block matches
        the empty-body placeholder and helps clients that expect a text segment.
        """
        provider = _make_provider()
        request = _make_request()
        chunk1 = _make_chunk(reasoning_content="reasoning only from provider")
        chunk2 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2])
        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
        ):
            events = await _collect_stream(provider, request)
        event_text = "".join(events)
        assert "thinking_delta" in event_text
        assert '"text_delta"' in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_stream_with_thinking_content(self):
        """Thinking content via think tags is emitted as thinking blocks."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(content="<think>reasoning</think>answer")
        chunk2 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "thinking" in event_text
        assert "reasoning" in event_text
        assert "answer" in event_text

    @pytest.mark.asyncio
    async def test_stream_with_reasoning_content_field(self):
        """reasoning_content delta field is emitted as thinking block."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(reasoning_content="I think...")
        chunk2 = _make_chunk(content="The answer")
        chunk3 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2, chunk3])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "thinking_delta" in event_text
        assert "I think..." in event_text
        assert "The answer" in event_text

    @pytest.mark.asyncio
    async def test_stream_with_empty_reasoning_content_creates_no_thinking_block(self):
        """Empty reasoning_content is stateful but must not emit visible thinking text."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(reasoning_content="")
        chunk2 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        thinking_starts = [
            event
            for event in parsed
            if event.event == "content_block_start"
            and event.data["content_block"]["type"] == "thinking"
        ]
        thinking_deltas = [
            event
            for event in parsed
            if event.event == "content_block_delta"
            and event.data["delta"]["type"] == "thinking_delta"
        ]
        assert thinking_starts == []
        assert thinking_deltas == []
        assert parsed[-1].event == "message_stop"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wire", ["messages", "responses"])
    @pytest.mark.parametrize(
        ("deltas", "expected"),
        [
            pytest.param(
                [("plan", ""), ("", "The quick"), ("", " brown fox"), ("", "")],
                [("reasoning", "plan"), ("text", "The quick brown fox")],
                id="empty-placeholders-after-reasoning",
            ),
            pytest.param(
                [("", "The quick"), ("", " brown fox"), ("", "")],
                [("text", "The quick brown fox")],
                id="first-empty-reasoning-is-ignored",
            ),
            pytest.param(
                [(None, "The quick"), (None, " brown fox")],
                [("text", "The quick brown fox")],
                id="no-reasoning-is-invented",
            ),
            pytest.param(
                [("", None), ("", "first"), ("more", None), ("", "second")],
                [
                    ("text", "first"),
                    ("reasoning", "more"),
                    ("text", "second"),
                ],
                id="nonempty-reasoning-can-resume",
            ),
        ],
    )
    async def test_empty_reasoning_placeholders_preserve_content_blocks(
        self, wire, deltas, expected
    ):
        provider = profiled_provider(
            "qwencloud",
            make_provider_config(api_key="test", base_url="https://provider.invalid"),
        )
        chunks = [
            _make_chunk(reasoning_content=reasoning, content=content)
            for reasoning, content in deltas
        ] + [_make_chunk(finish_reason="stop")]
        try:
            with patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=lambda **kwargs: AsyncStreamMock(chunks),
            ):
                # Empty placeholders do not create output items on either request.
                for _ in range(2):
                    if wire == "messages":
                        stream = provider.stream_messages(_make_request())
                    else:
                        stream = provider.stream_responses(
                            OpenAIResponsesRequest(model="test-model", input="Hello")
                        )
                    events = parse_sse_text(
                        "".join(
                            [frame async for frame in stream for frame in frame.frames]
                        )
                    )
                    if wire == "messages":
                        assert_anthropic_stream_contract(events)
                        actual = [
                            (
                                "reasoning"
                                if start.data["content_block"]["type"] == "thinking"
                                else start.data["content_block"]["type"],
                                "".join(
                                    event.data["delta"].get(
                                        "thinking", event.data["delta"].get("text", "")
                                    )
                                    for event in events
                                    if event.event == "content_block_delta"
                                    and event.data["index"] == start.data["index"]
                                ),
                            )
                            for start in events
                            if start.event == "content_block_start"
                        ]
                    else:
                        assert events[-1].event == "response.completed"
                        actual = [
                            (
                                "text" if item["type"] == "message" else item["type"],
                                "".join(part["text"] for part in item["content"]),
                            )
                            for item in events[-1].data["response"]["output"]
                        ]
                    assert actual == expected
        finally:
            await provider.cleanup()

    @pytest.mark.asyncio
    async def test_stream_with_reasoning_content_suppressed_when_disabled(self):
        """reasoning deltas are stripped while normal text still streams."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(reasoning_content="I think...")
        chunk2 = _make_chunk(content="<think>secret</think>The answer")
        chunk3 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2, chunk3])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
        ):
            events = await _collect_stream(provider, request, reasoning=REASONING_OFF)

        event_text = "".join(events)
        assert "thinking_delta" not in event_text
        assert "I think..." not in event_text
        assert "secret" not in event_text
        assert "The answer" in event_text

    @pytest.mark.asyncio
    async def test_stream_with_upstream_405_mentions_provider_name(self):
        """HTTP 405s are surfaced as upstream method/endpoint rejections."""
        provider = _make_provider()
        request = _make_request()

        response = httpx.Response(
            status_code=405,
            request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
        )
        error = httpx.HTTPStatusError(
            "Method Not Allowed",
            request=response.request,
            response=response,
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            stream_error = await _collect_stream_error(
                provider,
                request,
                request_id="REQ405",
            )

        assert (
            "Upstream provider NIM rejected the request method or endpoint (HTTP 405)."
            in stream_error.message
        )
        assert "Request ID: REQ405" in stream_error.message

    @pytest.mark.asyncio
    async def test_stream_with_openai_bad_request_surfaces_upstream_body(self):
        """OpenAI SDK bodies should be raised so users can copy exact provider errors."""
        provider = _make_provider()
        request = _make_request()
        response = httpx2.Response(
            status_code=400,
            request=httpx2.Request("POST", "https://example.com/v1/chat/completions"),
        )
        body = {
            "error": {
                "type": "BadRequest",
                "message": "Thinking mode does not support this tool_choice",
            }
        }
        error = openai.BadRequestError("Bad Request", response=response, body=body)

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            stream_error = await _collect_stream_error(
                provider,
                request,
                request_id="REQ_BODY",
            )

        assert "Upstream provider NIM returned HTTP 400." in stream_error.message
        assert "Category: BadRequest" in stream_error.message
        assert "Thinking mode does not support this tool_choice" in stream_error.message
        assert (
            '{"error":{"type":"BadRequest","message":"Thinking mode does not support this tool_choice"}}'
            in stream_error.message
        )
        assert "Request ID: REQ_BODY" in stream_error.message

    @pytest.mark.asyncio
    async def test_error_after_native_tool_call_failure_includes_body(self):
        """Detailed failure data survives without completing an unfinished tool."""
        provider = _make_provider()
        request = _make_request()
        tool_chunk = _make_tool_calls_chunk(
            name="echo_smoke", arguments="{}", tool_id="call_body", index=0
        )
        response = httpx2.Response(
            status_code=400,
            request=httpx2.Request("POST", "https://example.com/v1/chat/completions"),
        )
        body = {"error": {"message": "bad after tool"}}
        error = openai.BadRequestError("Bad Request", response=response, body=body)
        stream_mock = AsyncStreamMock([tool_chunk], error=error)

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events, stream_error = await _collect_stream_and_error(
                provider,
                request,
                request_id="REQ_TOOL_BODY",
            )

        event_text = "".join(events)
        parsed = parse_sse_text(event_text)
        assert "tool_use" not in event_text
        assert not any(event.event == "content_block_stop" for event in parsed)
        assert "event: error\n" not in event_text
        assert "bad after tool" not in event_text
        assert "Request ID: REQ_TOOL_BODY" not in event_text
        assert "message_stop" not in event_text
        assert "bad after tool" in stream_error.message
        assert "Request ID: REQ_TOOL_BODY" in stream_error.message
        _assert_error_not_in_text_deltas_after_tool(events, "bad after tool")

    @pytest.mark.asyncio
    async def test_clean_eof_does_not_promote_json_to_tool_completion(self):
        """Valid JSON without a source finish remains an unfinished call."""
        provider = _make_provider()
        request = _make_request()
        tool_chunk = _make_tool_calls_chunk(
            name="echo_smoke", arguments='{"message":"ok"}', tool_id="call_eof"
        )
        stream_mock = AsyncStreamMock([tool_chunk])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            pytest.raises(ExecutionFailure, match="without finish_reason"),
        ):
            await _collect_stream(provider, request)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("finish_reason", ["tool_calls", "stop"])
    async def test_legacy_tool_markup_remains_visible_text(self, finish_reason):
        """Legacy markup cannot manufacture executable tool calls."""
        provider = _make_provider()
        request = _make_request()
        heuristic_tool = (
            "● <function=Read><parameter=path>test.py</parameter>"
            "<parameter=limit>10</parameter>"
        )
        stream_mock = AsyncStreamMock(
            [
                _make_chunk(content=heuristic_tool),
                _make_chunk(finish_reason=finish_reason),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        assert not any(
            event.event == "content_block_start"
            and event.data.get("content_block", {}).get("type") == "tool_use"
            for event in parsed
        )
        assert not any(
            event.event == "content_block_delta"
            and event.data.get("delta", {}).get("type") == "text_delta"
            and event.data.get("delta", {}).get("text") == " "
            for event in parsed
        )
        assert any(
            event.event == "message_delta"
            and event.data.get("delta", {}).get("stop_reason") == "end_turn"
            for event in parsed
        )

        assert (
            "".join(event.data.get("delta", {}).get("text", "") for event in parsed)
            == heuristic_tool
        )

    @pytest.mark.asyncio
    async def test_function_tag_markup_remains_text_beside_reasoning(self):
        """Function-looking text stays visible after reasoning parsing."""
        provider = _make_provider()
        request = _make_request(
            tools=[
                {
                    "name": "Bash",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                }
            ]
        )
        raw_call = (
            "<think>Use the requested command.</think>"
            "I will invoke Bash now.\n"
            "<tool_call>\n<function=Bash>\n<parameter=command>\n"
            "printf FCC_STEP_TOOL\n</parameter>\n</function>\n</tool_call>"
        )
        stream_mock = AsyncStreamMock(
            [
                _make_chunk(content=raw_call[:43]),
                _make_chunk(content=raw_call[43:]),
                _make_chunk(finish_reason="stop"),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        starts = [
            event.data["content_block"]
            for event in parsed
            if event.event == "content_block_start"
        ]
        tool_starts = [block for block in starts if block["type"] == "tool_use"]
        input_json = "".join(
            event.data.get("delta", {}).get("partial_json", "")
            for event in parsed
            if event.event == "content_block_delta"
            and event.data.get("delta", {}).get("type") == "input_json_delta"
        )
        visible_text = "".join(
            event.data.get("delta", {}).get("text", "")
            for event in parsed
            if event.event == "content_block_delta"
            and event.data.get("delta", {}).get("type") == "text_delta"
        )

        assert tool_starts == []
        assert input_json == ""
        assert visible_text == raw_call.split("</think>", 1)[1]
        assert any(
            event.event == "content_block_delta"
            and event.data.get("delta", {}).get("type") == "thinking_delta"
            for event in parsed
        )
        assert any(
            event.event == "message_delta"
            and event.data.get("delta", {}).get("stop_reason") == "end_turn"
            for event in parsed
        )

    @pytest.mark.asyncio
    async def test_native_tool_call_disables_function_tag_recovery(self):
        """Native structured calls win and release earlier candidate text in order."""
        provider = _make_provider()
        request = _make_request(
            tools=[
                {
                    "name": "Bash",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                }
            ]
        )
        held_text = "<tool_call>\n<function=Bash>"
        stream_mock = AsyncStreamMock(
            [
                _make_chunk(content=held_text),
                _make_tool_calls_chunk(
                    name="Bash",
                    arguments='{"command":"printf native"}',
                    tool_id="call_native",
                ),
                _make_chunk(finish_reason="tool_calls"),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        visible_text = "".join(
            event.data.get("delta", {}).get("text", "")
            for event in parsed
            if event.event == "content_block_delta"
            and event.data.get("delta", {}).get("type") == "text_delta"
        )
        tool_starts = [
            event.data["content_block"]
            for event in parsed
            if event.event == "content_block_start"
            and event.data.get("content_block", {}).get("type") == "tool_use"
        ]

        assert visible_text == held_text
        assert [block["id"] for block in tool_starts] == ["call_native"]

    @pytest.mark.asyncio
    async def test_rejected_function_tag_candidate_bypasses_legacy_heuristics(self):
        """Rejected reserved text is preserved without a second permissive parse."""
        provider = _make_provider()
        request = _make_request(
            tools=[
                {
                    "name": "Bash",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                }
            ]
        )
        raw = (
            "<tool_call><function=Unknown></function></tool_call>"
            "● <function=Bash><parameter=command>printf unsafe</parameter>"
        )
        stream_mock = AsyncStreamMock(
            [_make_chunk(content=raw), _make_chunk(finish_reason="stop")]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        visible_text = "".join(
            event.data.get("delta", {}).get("text", "")
            for event in parsed
            if event.event == "content_block_delta"
            and event.data.get("delta", {}).get("type") == "text_delta"
        )
        tool_starts = [
            event
            for event in parsed
            if event.event == "content_block_start"
            and event.data.get("content_block", {}).get("type") == "tool_use"
        ]

        assert visible_text == raw
        assert tool_starts == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("committed", [False, True])
    async def test_literal_markup_respects_connection_commit_boundary(self, committed):
        """Literal markup follows ordinary retry and continuation boundaries."""
        provider = _make_provider()
        request = _make_request(
            tools=[
                {
                    "name": "Bash",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ]
        )
        abandoned = "<tool_call>\n<function=Bash>"
        if committed:
            abandoned += "x" * 66000
        complete = (
            "<tool_call>\n<function=Bash>\n<parameter=command>\n"
            "printf retry\n</parameter>\n</function>\n</tool_call>"
        )
        first_stream = AsyncStreamMock(
            [_make_chunk(content=abandoned)],
            error=httpx.ReadError("early cutoff"),
        )
        second_stream = AsyncStreamMock(
            [_make_chunk(content=complete), _make_chunk(finish_reason="stop")]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[first_stream, second_stream],
        ) as mock_create:
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        visible_text = "".join(
            event.data.get("delta", {}).get("text", "")
            for event in parsed
            if event.event == "content_block_delta"
            and event.data.get("delta", {}).get("type") == "text_delta"
        )
        tool_starts = [
            event.data["content_block"]
            for event in parsed
            if event.event == "content_block_start"
            and event.data.get("content_block", {}).get("type") == "tool_use"
        ]

        assert mock_create.await_count == 2
        assert visible_text == complete
        assert tool_starts == []

        assert "tools" in mock_create.call_args_list[1].kwargs

    @pytest.mark.asyncio
    async def test_precommit_retry_discards_abandoned_tool_id_candidate(self):
        """An ID observed only by an invisible attempt cannot leak into its replay."""
        provider = _make_provider()
        request = _make_request()
        first_stream = AsyncStreamMock(
            [
                _make_tool_calls_chunk(
                    name=None,
                    arguments="",
                    tool_id="call_abandoned",
                )
            ],
            error=httpx.ReadError("early cutoff"),
        )
        second_stream = AsyncStreamMock(
            [
                _make_tool_calls_chunk(
                    name="Bash",
                    arguments="{}",
                    tool_id=None,
                ),
                _make_chunk(finish_reason="tool_calls"),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[first_stream, second_stream],
        ) as mock_create:
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        [start] = _tool_use_starts(events)
        assert mock_create.await_count == 2
        assert start["id"].startswith("tool_")
        assert start["id"] != "call_abandoned"
        assert sum(event.event == "message_start" for event in parsed) == 1
        assert sum(event.event == "content_block_start" for event in parsed) == 1
        assert sum(event.event == "message_delta" for event in parsed) == 1
        assert sum(event.event == "message_stop" for event in parsed) == 1

    @pytest.mark.asyncio
    async def test_colliding_tool_id_round_trips_as_distinct_matched_pair(self):
        """A repaired public ID stays paired when Claude replays the next turn."""
        provider = _make_provider()
        first_pair = [
            {"role": "user", "content": "Run it once."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "Bash:0",
                        "name": "Bash",
                        "input": {"command": "printf first"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "Bash:0",
                        "content": "first",
                    }
                ],
            },
        ]
        request = _make_request(
            messages=[*first_pair, {"role": "user", "content": "Run it again."}]
        )
        stream = AsyncStreamMock(
            [
                _make_tool_calls_chunk(
                    name="Bash",
                    arguments='{"command":"printf second"}',
                    tool_id="Bash:0",
                ),
                _make_chunk(finish_reason="tool_calls"),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream,
        ):
            events = await _collect_stream(provider, request)

        [start] = _tool_use_starts(events)
        public_id = start["id"]
        assert public_id != "Bash:0"

        replay = _make_request(
            messages=[
                *request.messages,
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": public_id,
                            "name": "Bash",
                            "input": {"command": "printf second"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": public_id,
                            "content": "second",
                        }
                    ],
                },
            ]
        )
        body = provider._chat._build_request_body(
            replay,
            reasoning=DEFAULT_REASONING_POLICY,
        )
        assistant_ids = [
            message["tool_calls"][0]["id"]
            for message in body["messages"]
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        result_ids = [
            message["tool_call_id"]
            for message in body["messages"]
            if message.get("role") == "tool"
        ]
        assert assistant_ids == ["Bash:0", public_id]
        assert result_ids == ["Bash:0", public_id]

    @pytest.mark.asyncio
    async def test_precommit_retry_emits_one_unduplicated_downstream_lifecycle(self):
        """An abandoned attempt contributes no frame to the successful replay."""
        provider = _make_provider()
        request = _make_request()
        first_stream = AsyncStreamMock(
            [_make_chunk(content="hidden")],
            error=httpx.ReadError("early cutoff"),
        )
        second_stream = AsyncStreamMock(
            [
                _make_chunk(content="visible"),
                _make_chunk(finish_reason="stop"),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[first_stream, second_stream],
        ) as mock_create:
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert mock_create.await_count == 2
        assert "hidden" not in event_text
        parsed = parse_sse_text(event_text)
        text_deltas = [
            event.data.get("delta", {}).get("text", "")
            for event in parsed
            if event.event == "content_block_delta"
        ]
        assert text_deltas == ["visible"]
        assert sum(event.event == "message_start" for event in parsed) == 1
        assert sum(event.event == "content_block_start" for event in parsed) == 1
        assert sum(event.event == "content_block_stop" for event in parsed) == 1
        assert sum(event.event == "message_delta" for event in parsed) == 1
        assert sum(event.event == "message_stop" for event in parsed) == 1
        assert parsed[0].event == "message_start"
        assert parsed[-1].event == "message_stop"

    @pytest.mark.asyncio
    async def test_responses_precommit_retry_emits_one_unduplicated_lifecycle(self):
        """Responses output also discards every frame from an abandoned attempt."""
        provider = _make_provider()
        request = OpenAIResponsesRequest(model="test-model", input="hello")
        first_stream = AsyncStreamMock(
            [_make_chunk(content="hidden")],
            error=httpx.ReadError("early cutoff"),
        )
        second_stream = AsyncStreamMock(
            [
                _make_chunk(content="visible"),
                _make_chunk(finish_reason="stop"),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[first_stream, second_stream],
        ) as create:
            events = await collect_response(
                provider.stream_responses(request, response_model="public-model"),
                protocol="responses",
            )

        event_text = "".join(events)
        parsed = parse_sse_text(event_text)
        event_names = [event.event for event in parsed]
        assert create.await_count == 2
        assert "hidden" not in event_text
        assert "visible" in event_text
        assert event_names.count("response.created") == 1
        assert event_names.count("response.completed") == 1
        assert "response.failed" not in event_names

    @pytest.mark.asyncio
    async def test_responses_cancellation_closes_upstream_without_retry(self):
        provider = _make_provider()
        request = OpenAIResponsesRequest(model="test-model", input="hello")
        stream = BlockingClosableAsyncStreamMock()

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream,
        ) as create:
            task = asyncio.create_task(
                anext(
                    provider.stream_responses(
                        request,
                        request_id="req_responses_cancel",
                        response_model="public-model",
                    )
                )
            )
            await asyncio.wait_for(stream.entered.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert create.await_count == 1
        assert stream.close_calls == 1

    @pytest.mark.asyncio
    async def test_precommit_retry_discards_abandoned_parser_and_usage_state(self):
        """A replay owns fresh parsers and terminal usage metadata."""
        provider = _make_provider()
        request = _make_request()
        first_stream = AsyncStreamMock(
            [
                _make_usage_chunk(prompt_tokens=111, completion_tokens=222),
                _make_chunk(content="<thi"),
            ],
            error=httpx.ReadError("early cutoff"),
        )
        second_stream = AsyncStreamMock(
            [
                _make_chunk(content="visible"),
                _make_chunk(finish_reason="stop"),
                _make_usage_chunk(prompt_tokens=7, completion_tokens=3),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[first_stream, second_stream],
        ) as create:
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        visible_text = "".join(
            event.data.get("delta", {}).get("text", "")
            for event in parsed
            if event.event == "content_block_delta"
            and event.data.get("delta", {}).get("type") == "text_delta"
        )
        final_usage = next(
            event.data["usage"] for event in parsed if event.event == "message_delta"
        )

        assert create.await_count == 2
        assert visible_text == "visible"
        assert final_usage == {"input_tokens": 7, "output_tokens": 3}
        assert sum(event.event == "message_start" for event in parsed) == 1
        assert sum(event.event == "message_delta" for event in parsed) == 1
        assert sum(event.event == "message_stop" for event in parsed) == 1

    @pytest.mark.asyncio
    async def test_create_correction_survives_later_precommit_retry(self):
        """A corrected request body outlives the replay-local stream state."""
        provider = _make_provider()
        request = _make_request()
        response = httpx2.Response(
            status_code=400,
            request=httpx2.Request("POST", "https://example.com/v1/chat/completions"),
        )
        usage_rejection = openai.BadRequestError(
            "stream_options is unsupported",
            response=response,
            body={"error": {"message": "stream_options is unsupported"}},
        )
        first_stream = AsyncStreamMock(
            [_make_chunk(content="hidden")],
            error=httpx.ReadError("early cutoff"),
        )
        second_stream = AsyncStreamMock(
            [_make_chunk(content="visible"), _make_chunk(finish_reason="stop")]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[usage_rejection, first_stream, second_stream],
        ) as create:
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        text_deltas = [
            event.data.get("delta", {}).get("text", "")
            for event in parsed
            if event.event == "content_block_delta"
            and event.data.get("delta", {}).get("type") == "text_delta"
        ]

        assert create.await_count == 3
        assert create.await_args_list[0].kwargs["stream_options"] == {
            "include_usage": True
        }
        assert "stream_options" not in create.await_args_list[1].kwargs
        assert "stream_options" not in create.await_args_list[2].kwargs
        assert text_deltas == ["visible"]
        assert sum(event.event == "message_start" for event in parsed) == 1
        assert sum(event.event == "message_stop" for event in parsed) == 1

    @pytest.mark.asyncio
    async def test_precommit_retry_discards_abandoned_tool_name_fragment(self):
        """A retried alias fragment cannot prefix or duplicate the successful call."""
        provider = _make_provider()
        original = "mcp__retry_tool_name__" + "x" * 70
        request = _make_request(
            tools=[{"name": original, "input_schema": {"type": "object"}}]
        )
        alias = OpenAIToolNameCodec.from_request(request).encode(original)
        first_stream = AsyncStreamMock(
            [
                _make_tool_calls_chunk(
                    name=alias[:20],
                    arguments="",
                    tool_id="call_abandoned",
                ),
            ]
        )
        second_stream = AsyncStreamMock(
            [
                _make_tool_calls_chunk(
                    name=alias,
                    arguments="{}",
                    tool_id="call_success",
                ),
                _make_chunk(finish_reason="tool_calls"),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[first_stream, second_stream],
        ) as create:
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        parsed = parse_sse_text(event_text)
        tool_starts = [
            event.data["content_block"]
            for event in parsed
            if event.event == "content_block_start"
            and event.data.get("content_block", {}).get("type") == "tool_use"
        ]
        assert create.await_count == 2
        assert tool_starts == [
            {
                "type": "tool_use",
                "id": "call_success",
                "name": original,
                "input": {},
            }
        ]
        assert alias not in event_text
        assert "call_abandoned" not in event_text
        assert sum(event.event == "message_start" for event in parsed) == 1
        assert sum(event.event == "message_stop" for event in parsed) == 1

    @pytest.mark.asyncio
    async def test_private_replacements_share_one_generation_budget(self):
        """Private drafts are replaced within the same bounded execution."""
        provider = _make_provider()
        request = _make_request()
        primary_attempts = UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS - 1
        primary_streams = [
            AsyncStreamMock([_make_chunk(content="hello")])
            for _ in range(primary_attempts)
        ]
        continuation = AsyncStreamMock(
            [
                _make_chunk(content="hello world"),
                _make_chunk(finish_reason="stop"),
            ]
        )

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=[*primary_streams, continuation],
            ) as create,
            patch("free_claude_code.providers.admission.trace_event") as attempt_trace,
        ):
            events = await _collect_stream(provider, request)

        assert create.await_count == UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
        assert all(
            call.kwargs["messages"] == create.await_args_list[0].kwargs["messages"]
            for call in create.await_args_list[:primary_attempts]
        )
        assert (
            create.await_args_list[primary_attempts].kwargs["messages"]
            == (create.await_args_list[0].kwargs["messages"])
        )
        parsed = parse_sse_text("".join(events))
        text = "".join(
            event.data.get("delta", {}).get("text", "")
            for event in parsed
            if event.event == "content_block_delta"
        )
        assert text == "hello world"
        assert sum(event.event == "message_start" for event in parsed) == 1
        assert sum(event.event == "message_delta" for event in parsed) == 1
        assert sum(event.event == "message_stop" for event in parsed) == 1
        starts = [
            call.kwargs
            for call in attempt_trace.call_args_list
            if call.kwargs.get("event") == "provider.attempt.started"
        ]
        assert [row["operation_kind"] for row in starts] == (
            [ProviderOperationKind.GENERATION.value] * primary_attempts
            + [ProviderOperationKind.GENERATION.value]
        )
        assert len({row["execution_id"] for row in starts}) == 1

    @pytest.mark.asyncio
    async def test_disabled_thinking_recovery_discards_reasoning(self):
        provider = _make_provider()
        request = _make_request()
        initial_stream = AsyncStreamMock([_make_chunk(content="hello")])
        recovery_stream = AsyncStreamMock(
            [
                _make_chunk(reasoning_content="hidden reasoning"),
                _make_chunk(content="hello world"),
                _make_chunk(finish_reason="stop"),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[initial_stream, recovery_stream],
        ):
            events = await _collect_stream(provider, request, reasoning=REASONING_OFF)

        parsed = parse_sse_text("".join(events))
        text = "".join(
            event.data.get("delta", {}).get("text", "")
            for event in parsed
            if event.event == "content_block_delta"
        )
        assert text == "hello world"
        assert "hidden reasoning" not in "".join(events)
        assert not any(
            event.data.get("delta", {}).get("type") == "thinking_delta"
            for event in parsed
        )

    @pytest.mark.asyncio
    async def test_context_exhaustion_during_recovery_remains_terminal(self):
        """A recovery request cannot turn context exhaustion into success."""
        original = ClosableAsyncStreamMock(
            [_make_chunk(content="partial" + ("x" * 70_000))]
        )
        recovery = ClosableAsyncStreamMock(
            [
                _make_chunk(content="continued"),
                _make_chunk(finish_reason="model_context_window_exceeded"),
            ]
        )
        provider = _make_provider()

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[original, recovery],
        ):
            events, failure = await _collect_stream_and_error(
                provider,
                _make_request(),
            )

        assert failure.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
        assert failure.retryable is False
        assert not any(
            event.event == "message_stop" for event in parse_sse_text("".join(events))
        )
        assert original.closed is True
        assert recovery.closed is True

    @pytest.mark.asyncio
    async def test_context_error_opening_continuation_remains_terminal(self):
        """A derived continuation's SDK context error replaces the cutoff."""
        original = ClosableAsyncStreamMock(
            [_make_chunk(content="partial" + ("x" * 70_000))]
        )
        provider = _make_provider()

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[
                original,
                _make_context_length_bad_request(),
                _make_context_length_bad_request(),
            ],
        ) as create:
            events, failure = await _collect_stream_and_error(
                provider,
                _make_request(),
            )

        assert create.await_count == 3
        assert failure.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
        assert failure.retryable is False
        assert not any(
            event.event == "message_stop" for event in parse_sse_text("".join(events))
        )
        assert original.closed is True

    @pytest.mark.asyncio
    async def test_primary_stream_closes_when_iteration_fails(self):
        """OpenAI-chat main streams close after iterator failures."""
        provider = _make_provider()
        request = _make_request()
        stream = ClosableAsyncStreamMock(
            [_make_chunk(content="partial")],
            error=ValueError("provider stream failed"),
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream,
        ):
            error = await _collect_stream_error(provider, request)

        assert stream.closed is True
        assert "provider stream failed" in error.message.lower()

    @pytest.mark.asyncio
    async def test_stream_rate_limit_uses_the_execution_retry_session(self):
        """A create-time 429 consumes one attempt before a successful retry."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(content="Response")
        chunk2 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2])

        response = httpx.Response(
            429,
            request=httpx.Request(
                "POST", "https://test.api.nvidia.com/v1/chat/completions"
            ),
        )
        error = httpx.HTTPStatusError(
            "rate limited",
            request=response.request,
            response=response,
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[error, stream_mock],
        ) as create:
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert create.await_count == 2
        assert "Response" in event_text


class TestStreamChunkEdgeCases:
    """Tests for edge cases in stream chunk handling."""

    @pytest.mark.asyncio
    async def test_stream_chunk_with_empty_choices_skipped(self):
        """Chunk with choices=[] is skipped without crashing."""
        provider = _make_provider()
        request = _make_request()

        empty_choices_chunk = MagicMock()
        empty_choices_chunk.choices = []
        empty_choices_chunk.usage = None

        finish_chunk = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([empty_choices_chunk, finish_chunk])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "message_start" in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_stream_chunk_with_none_delta_handled(self):
        """Chunk with choice.delta=None is handled defensively."""
        provider = _make_provider()
        request = _make_request()

        none_delta_chunk = MagicMock()
        none_delta_chunk.usage = None
        choice = MagicMock()
        choice.delta = None
        choice.finish_reason = None
        none_delta_chunk.choices = [choice]

        finish_chunk = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([none_delta_chunk, finish_chunk])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "message_start" in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_stream_generator_cleanup_on_exception(self):
        """When stream raises mid-iteration, message_stop still emitted."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(content="Partial")
        stream_mock = AsyncStreamMock(
            [chunk1], error=ConnectionResetError("Connection reset")
        )

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
        ):
            error = await _collect_stream_error(provider, request)

        assert "Connection reset" in error.message


@pytest.mark.asyncio
async def test_openai_compat_stream_ends_with_contract_when_tool_name_never_arrives() -> (
    None
):
    """Nameless / incomplete tool-call buffer must not break Anthropic stream contract."""
    provider = _make_provider()
    request = _make_request()
    tc0 = SimpleNamespace(
        index=0,
        id="call_inc",
        function=SimpleNamespace(name=None, arguments="{}"),
    )
    stream_mock = AsyncStreamMock([_make_chunk(tool_calls=[tc0])])
    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ),
    ):
        error = await _collect_stream_error(provider, request)

    assert "Provider stream ended without finish_reason." in error.message
