"""Tests for streaming error handling in providers/nvidia_nim/client.py."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import httpx2
import openai
import pytest

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
)
from free_claude_code.core.anthropic.streaming import (
    make_response_recovery_body,
    make_text_recovery_body,
    tool_schemas_by_name,
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
from free_claude_code.providers.openai_chat.tool_calls import (
    OpenAIToolCallAssembler,
    OpenAIToolCallCollector,
    iter_heuristic_tool_use_events,
)
from free_claude_code.providers.openai_chat.transport import (
    _OpenAIChatStreamRunner,
    _reserved_anthropic_tool_ids,
)
from free_claude_code.providers.stream_recovery import TruncatedProviderStreamError
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    REASONING_OFF,
    immediate_admission,
    make_provider_config,
    profiled_provider,
)
from tests.providers.test_history_transports import _events_for, _harness, _saved_reply
from tests.providers.test_nvidia_nim import (
    _alias_events,
    _alias_provider,
    _alias_request,
)


class AsyncStreamMock:
    """Async iterable mock that yields chunks then optionally raises."""

    def __init__(self, chunks, error=None):
        self._chunks = chunks
        self._error = error

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for chunk in self._chunks:
            yield chunk
        if self._error:
            raise self._error


def _recovery_output(
    text: str = "",
    thinking: str = "",
    tool_calls: tuple[dict, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(text=text, thinking=thinking, tool_calls=tool_calls)


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


class BlockingClosableAsyncStreamMock:
    """Async stream that blocks until its consumer is cancelled."""

    def __init__(self, *, close_error=None):
        self.entered = asyncio.Event()
        self.close_calls = 0
        self._close_error = close_error

    def __aiter__(self):
        return self._aiter()

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
        rate_limit=10,
        rate_window=60,
    )
    return NvidiaNimProvider(
        config,
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )


def _make_tool_assembler(
    provider: NvidiaNimProvider, *, request=None
) -> OpenAIToolCallAssembler:
    concrete_request = request or _make_request()
    return OpenAIToolCallAssembler(
        reserved_tool_ids=_reserved_anthropic_tool_ids(concrete_request),
        record_extra_content=provider._behavior.record_tool_call_extra_content,
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
        tool_schemas=tool_schemas_by_name(concrete_request),
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
    """Create a mock streaming chunk."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    delta.reasoning_content = reasoning_content

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


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
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return chunk


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
    return [e async for e in provider.stream_messages(request, reasoning=reasoning)]


async def _collect_stream_error(provider, request, **kwargs) -> ExecutionFailure:
    with pytest.raises(ExecutionFailure) as exc_info:
        [e async for e in provider.stream_messages(request, **kwargs)]
    return exc_info.value


async def _collect_stream_and_error(
    provider, request, **kwargs
) -> tuple[list[str], ExecutionFailure]:
    events: list[str] = []
    with pytest.raises(ExecutionFailure) as exc_info:
        async for event in provider.stream_messages(request, **kwargs):
            events.extend((event,))
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
        execution = provider._admission.start_execution()

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
            pytest.raises(ValueError, match="invalid stream wrapper"),
        ):
            await provider._chat._create_stream(
                {"model": "test-model", "messages": []},
                execution,
                ProviderOperationKind.GENERATION,
            )

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
                    event
                    async for event in provider.stream_responses(
                        OpenAIResponsesRequest(model="test-model", input="hello"),
                        request_id="req_context",
                        response_model="public-model",
                    )
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
    async def test_error_after_native_tool_call_closes_block_then_raises(self):
        """A provider closes tool state, then leaves terminal serialization to API."""
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
        assert "tool_use" in event_text
        assert parsed[-1].event == "content_block_stop"
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
        chunk = SimpleNamespace(choices=[choice], usage=usage)
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
    async def test_stream_with_empty_reasoning_content_starts_thinking_block_only(self):
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
        assert len(thinking_starts) == 1
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
                [("reasoning", ""), ("text", "The quick brown fox")],
                id="first-empty-reasoning-is-preserved",
            ),
            pytest.param(
                [(None, "The quick"), (None, " brown fox")],
                [("text", "The quick brown fox")],
                id="no-reasoning-is-invented",
            ),
            pytest.param(
                [("", None), ("", "first"), ("more", None), ("", "second")],
                [
                    ("reasoning", ""),
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
                # Each request must preserve its own first explicit reasoning value.
                for _ in range(2):
                    if wire == "messages":
                        stream = provider.stream_messages(_make_request())
                    else:
                        stream = provider.stream_responses(
                            OpenAIResponsesRequest(model="test-model", input="Hello")
                        )
                    events = parse_sse_text("".join([frame async for frame in stream]))
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
        """Detailed failure data survives after the provider closes tool state."""
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
        assert "tool_use" in event_text
        assert parsed[-1].event == "content_block_stop"
        assert "event: error\n" not in event_text
        assert "bad after tool" not in event_text
        assert "Request ID: REQ_TOOL_BODY" not in event_text
        assert "message_stop" not in event_text
        assert "bad after tool" in stream_error.message
        assert "Request ID: REQ_TOOL_BODY" in stream_error.message
        _assert_error_not_in_text_deltas_after_tool(events, "bad after tool")

    @pytest.mark.asyncio
    async def test_clean_eof_after_complete_tool_call_salvages_tool_use(self):
        """A complete tool JSON payload missing finish_reason is committed as tool_use."""
        provider = _make_provider()
        request = _make_request()
        tool_chunk = _make_tool_calls_chunk(
            name="echo_smoke", arguments='{"message":"ok"}', tool_id="call_eof"
        )
        stream_mock = AsyncStreamMock([tool_chunk])

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        assert parsed[-1].event == "message_stop"
        assert any(
            event.event == "message_delta"
            and event.data.get("delta", {}).get("stop_reason") == "tool_use"
            for event in parsed
        )
        assert not any(event.event == "error" for event in parsed)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("finish_reason", ["tool_calls", "stop"])
    async def test_heuristic_only_tool_stream_does_not_emit_fallback_text(
        self, finish_reason
    ):
        """Text-parsed tool calls count as emitted tool output when finalizing."""
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
        assert any(
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
            and event.data.get("delta", {}).get("stop_reason") == "tool_use"
            for event in parsed
        )

    @pytest.mark.asyncio
    async def test_function_tag_tool_stream_becomes_one_anthropic_tool_use(self):
        """Exact control-only function tags use the established tool lifecycle."""
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

        assert [block["name"] for block in tool_starts] == ["Bash"]
        assert json.loads(input_json) == {"command": "printf FCC_STEP_TOOL"}
        assert visible_text == "I will invoke Bash now.\n"
        assert any(
            event.event == "content_block_delta"
            and event.data.get("delta", {}).get("type") == "thinking_delta"
            for event in parsed
        )
        assert any(
            event.event == "message_delta"
            and event.data.get("delta", {}).get("stop_reason") == "tool_use"
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
    async def test_function_tag_candidate_is_reset_before_early_retry(self):
        """An abandoned textual candidate cannot leak or duplicate after retry."""
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
        assert visible_text == ""
        assert [block["name"] for block in tool_starts] == ["Bash"]

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
            events = [
                event
                async for event in provider.stream_responses(
                    request,
                    request_id="req_responses_retry",
                    response_model="public-model",
                )
            ]

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
                _make_chunk(finish_reason="tool_calls"),
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
    async def test_primary_replay_and_continuation_share_five_attempts(self):
        """Four replays plus continuation emit one unduplicated response."""
        provider = _make_provider()
        request = _make_request()
        primary_streams = [
            AsyncStreamMock([_make_chunk(content="hello")]) for _ in range(4)
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
            for call in create.await_args_list[:4]
        )
        assert (
            create.await_args_list[4].kwargs["messages"]
            != (create.await_args_list[0].kwargs["messages"])
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
        assert [row["operation_kind"] for row in starts] == [
            ProviderOperationKind.GENERATION.value,
            ProviderOperationKind.GENERATION.value,
            ProviderOperationKind.GENERATION.value,
            ProviderOperationKind.GENERATION.value,
            ProviderOperationKind.CONTINUATION.value,
        ]
        assert len({row["execution_id"] for row in starts}) == 1

    @pytest.mark.asyncio
    async def test_clean_eof_after_text_continues_with_overlap_trim(self):
        """A truncated text stream is continued and duplicate overlap is trimmed."""
        provider = _make_provider()
        request = _make_request()
        stream_mock = AsyncStreamMock([_make_chunk(content="hello wor")])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                _OpenAIChatStreamRunner,
                "_collect_recovery_output",
                new_callable=AsyncMock,
                return_value=_recovery_output(text="world"),
            ),
        ):
            events = await _collect_stream(provider, request)

        parsed = parse_sse_text("".join(events))
        text_deltas = [
            event.data.get("delta", {}).get("text", "")
            for event in parsed
            if event.event == "content_block_delta"
        ]
        assert text_deltas == ["hello wor", "ld"]
        assert "".join(text_deltas) == "hello world"
        assert sum(event.event == "message_start" for event in parsed) == 1
        assert sum(event.event == "content_block_start" for event in parsed) == 1
        assert sum(event.event == "content_block_stop" for event in parsed) == 1
        assert sum(event.event == "message_delta" for event in parsed) == 1
        assert sum(event.event == "message_stop" for event in parsed) == 1
        assert any(
            event.event == "message_delta"
            and event.data.get("delta", {}).get("stop_reason") == "end_turn"
            for event in parsed
        )
        assert not any(event.event == "error" for event in parsed)

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
    async def test_recovery_collect_text_requires_finish_reason(self):
        """Recovery collectors reject truncated OpenAI-chat continuation streams."""
        streams = [
            ClosableAsyncStreamMock([_make_chunk(content=f"world {index}")])
            for index in range(UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS)
        ]
        provider = _make_provider()
        runner = _make_stream_runner(provider)
        execution = provider._admission.start_execution()

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=streams,
            ) as create,
            pytest.raises(TruncatedProviderStreamError),
        ):
            await runner._collect_recovery_output(
                {"model": "test-model", "messages": []},
                include_reasoning=True,
                execution=execution,
                operation_kind=ProviderOperationKind.CONTINUATION,
            )

        assert create.await_count == UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
        assert all(stream.closed for stream in streams)

    @pytest.mark.asyncio
    async def test_recovery_collect_text_closes_retryable_failed_streams(self):
        """Recovery collectors close failed stream attempts before retrying."""
        streams = [
            ClosableAsyncStreamMock(
                [_make_chunk(content=f"partial {index}")],
                error=TimeoutError("recovery cutoff"),
            )
            for index in range(UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS)
        ]
        provider = _make_provider()
        runner = _make_stream_runner(provider)
        execution = provider._admission.start_execution()

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=streams,
            ) as create,
            pytest.raises(TimeoutError),
        ):
            await runner._collect_recovery_output(
                {"model": "test-model", "messages": []},
                include_reasoning=True,
                execution=execution,
                operation_kind=ProviderOperationKind.CONTINUATION,
            )

        assert create.await_count == UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
        assert all(stream.closed for stream in streams)

    @pytest.mark.asyncio
    async def test_recovery_stream_reopen_retains_accepted_corrected_body(self):
        """Reopening one derived request reuses its accepted correction."""
        provider = _make_provider()
        runner = _make_stream_runner(provider)
        execution = provider._admission.start_execution()
        response = httpx2.Response(
            status_code=400,
            request=httpx2.Request("POST", "https://example.com/v1/chat/completions"),
        )
        usage_rejection = openai.BadRequestError(
            "stream_options is unsupported",
            response=response,
            body={"error": {"message": "stream_options is unsupported"}},
        )
        failed_stream = ClosableAsyncStreamMock(
            [_make_chunk(content="discarded")],
            error=httpx.ReadError("recovery cutoff"),
        )
        successful_stream = ClosableAsyncStreamMock(
            [_make_chunk(content="visible"), _make_chunk(finish_reason="stop")]
        )
        body = {
            "model": "test-model",
            "messages": [],
            "stream_options": {"include_usage": True},
        }

        with (
            patch.object(
                provider._chat,
                "_create_stream",
                new_callable=AsyncMock,
                wraps=provider._chat._create_stream,
            ) as create_stream,
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=[usage_rejection, failed_stream, successful_stream],
            ) as create,
        ):
            recovered = await runner._collect_recovery_output(
                body,
                include_reasoning=True,
                execution=execution,
                operation_kind=ProviderOperationKind.CONTINUATION,
            )

        assert create.await_count == 3
        assert create.await_args_list[0].kwargs["stream_options"] == {
            "include_usage": True
        }
        assert "stream_options" not in create.await_args_list[1].kwargs
        assert "stream_options" not in create.await_args_list[2].kwargs
        assert create_stream.await_count == 2
        assert (
            create_stream.await_args_list[0].kwargs["used_retry_kinds"]
            is create_stream.await_args_list[1].kwargs["used_retry_kinds"]
        )
        assert recovered.text == "visible"
        assert failed_stream.closed
        assert successful_stream.closed

    @pytest.mark.asyncio
    async def test_tool_repair_iterations_reuse_accepted_corrected_body(self):
        """Schema-repair retries reuse corrections accepted for that repair body."""
        provider = _make_provider()
        request = _make_request(
            tools=[
                {
                    "name": "echo_smoke",
                    "description": "Echo one message",
                    "input_schema": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                }
            ]
        )
        runner = _make_stream_runner(provider, request=request)
        assembler = runner._new_stream_assembler(output_reasoning=False)
        tuple(assembler.start_events())
        tuple(
            assembler.feed(
                _make_tool_calls_chunk(
                    name="echo_smoke",
                    arguments='{"message":',
                    tool_id="call_repair",
                )
            )
        )
        body = provider._chat._build_request_body(request)
        body["stream_options"] = {"include_usage": True}
        response = httpx2.Response(
            status_code=400,
            request=httpx2.Request("POST", "https://example.com/v1/chat/completions"),
        )
        usage_rejection = openai.BadRequestError(
            "stream_options is unsupported",
            response=response,
            body={"error": {"message": "stream_options is unsupported"}},
        )
        invalid_repair = ClosableAsyncStreamMock(
            [_make_chunk(content="123}"), _make_chunk(finish_reason="stop")]
        )
        valid_repair = ClosableAsyncStreamMock(
            [_make_chunk(content='"ok"}'), _make_chunk(finish_reason="stop")]
        )
        execution = provider._admission.start_execution()

        with (
            patch.object(
                provider._chat,
                "_create_stream",
                new_callable=AsyncMock,
                wraps=provider._chat._create_stream,
            ) as create_stream,
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=[usage_rejection, invalid_repair, valid_repair],
            ) as create,
        ):
            events = await runner._repair_tool_args(
                body=body,
                output=assembler.output,
                tool_argument_alias_buffers=assembler.tool_argument_alias_buffers,
                execution=execution,
            )

        assert events is not None
        assert create.await_count == 3
        assert create.await_args_list[0].kwargs["stream_options"] == {
            "include_usage": True
        }
        assert "stream_options" not in create.await_args_list[1].kwargs
        assert "stream_options" not in create.await_args_list[2].kwargs
        assert create_stream.await_count == 2
        assert (
            create_stream.await_args_list[0].kwargs["used_retry_kinds"]
            is create_stream.await_args_list[1].kwargs["used_retry_kinds"]
        )
        assert invalid_repair.closed
        assert valid_repair.closed

    @pytest.mark.asyncio
    async def test_recovery_collect_text_accepts_finish_reason(self):
        """Recovery collectors return text only after the upstream terminal marker."""
        stream = ClosableAsyncStreamMock(
            [
                _make_chunk(content="world"),
                _make_chunk(finish_reason="stop"),
            ]
        )
        provider = _make_provider()
        runner = _make_stream_runner(provider)
        execution = provider._admission.start_execution()

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream,
        ):
            result = await runner._collect_recovery_output(
                {"model": "test-model", "messages": []},
                include_reasoning=True,
                execution=execution,
                operation_kind=ProviderOperationKind.CONTINUATION,
            )

        assert result.text == "world"
        assert result.thinking == ""
        assert result.tool_calls == ()
        assert stream.closed is True

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
            side_effect=[original, _make_context_length_bad_request()],
        ) as create:
            events, failure = await _collect_stream_and_error(
                provider,
                _make_request(),
            )

        assert create.await_count == 2
        assert failure.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
        assert failure.retryable is False
        assert not any(
            event.event == "message_stop" for event in parse_sse_text("".join(events))
        )
        assert original.closed is True

    @pytest.mark.asyncio
    async def test_context_error_opening_tool_repair_remains_terminal(self):
        """A derived tool repair's SDK context error replaces the truncation."""
        provider = _make_provider()
        request = _make_request(
            tools=[
                {
                    "name": "echo_smoke",
                    "description": "Echo",
                    "input_schema": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                }
            ]
        )
        original = ClosableAsyncStreamMock(
            [
                _make_tool_calls_chunk(
                    name="echo_smoke",
                    arguments='{"message":"' + ("x" * 70_000),
                    tool_id="call_repair",
                )
            ],
            error=httpx.ReadError("tool stream cutoff"),
        )

        with (
            patch.object(
                provider._chat,
                "_create_stream",
                new_callable=AsyncMock,
                wraps=provider._chat._create_stream,
            ) as create_stream,
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=[original, _make_context_length_bad_request()],
            ) as create,
        ):
            events, failure = await _collect_stream_and_error(provider, request)

        assert create.await_count == 2
        assert [call.args[2] for call in create_stream.await_args_list] == [
            ProviderOperationKind.GENERATION,
            ProviderOperationKind.TOOL_REPAIR,
        ]
        assert failure.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
        assert failure.retryable is False
        assert not any(
            event.event == "message_stop" for event in parse_sse_text("".join(events))
        )
        assert original.closed is True

    @pytest.mark.asyncio
    async def test_recovery_close_failure_preserves_completed_output(self):
        """A failed stream close cannot replace completed recovery output."""
        stream = ClosableAsyncStreamMock(
            [
                _make_chunk(content="world"),
                _make_chunk(finish_reason="stop"),
            ],
            close_error=RuntimeError("cleanup failed"),
        )
        provider = _make_provider()
        runner = _make_stream_runner(provider, request_id="req_recovery_success")
        execution = provider._admission.start_execution(
            request_id="req_recovery_success"
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream,
        ):
            result = await runner._collect_recovery_output(
                {"model": "test-model", "messages": []},
                include_reasoning=True,
                execution=execution,
                operation_kind=ProviderOperationKind.CONTINUATION,
            )

        replacement = await execution.open_attempt(ProviderOperationKind.CONTINUATION)
        await replacement.aclose()
        assert result.text == "world"
        assert stream.close_calls == 1

    @pytest.mark.asyncio
    async def test_recovery_close_failure_does_not_block_retry(self):
        """Cleanup failure cannot replace a retryable recovery failure."""
        failed = ClosableAsyncStreamMock(
            [],
            error=TimeoutError("original retryable failure"),
            close_error=RuntimeError("cleanup failed"),
        )
        recovered = ClosableAsyncStreamMock(
            [
                _make_chunk(content="world"),
                _make_chunk(finish_reason="stop"),
            ]
        )
        provider = _make_provider()
        runner = _make_stream_runner(provider)
        execution = provider._admission.start_execution()

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[failed, recovered],
        ) as create:
            result = await runner._collect_recovery_output(
                {"model": "test-model", "messages": []},
                include_reasoning=True,
                execution=execution,
                operation_kind=ProviderOperationKind.CONTINUATION,
            )

        assert result.text == "world"
        assert create.await_count == 2
        assert failed.close_calls == 1
        assert recovered.close_calls == 1

    @pytest.mark.asyncio
    async def test_recovery_close_failure_preserves_terminal_failure(self):
        """Cleanup failure cannot mask a non-retryable recovery failure."""
        stream = ClosableAsyncStreamMock(
            [],
            error=ValueError("original terminal failure"),
            close_error=RuntimeError("cleanup failed"),
        )
        provider = _make_provider()
        runner = _make_stream_runner(provider)
        execution = provider._admission.start_execution()

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream,
            ),
            pytest.raises(ValueError, match="original terminal failure"),
        ):
            await runner._collect_recovery_output(
                {"model": "test-model", "messages": []},
                include_reasoning=True,
                execution=execution,
                operation_kind=ProviderOperationKind.CONTINUATION,
            )

        assert stream.close_calls == 1

    @pytest.mark.asyncio
    async def test_recovery_close_failure_preserves_cancellation(self):
        """Cleanup failure cannot turn caller cancellation into a provider error."""
        stream = BlockingClosableAsyncStreamMock(
            close_error=RuntimeError("cleanup failed")
        )
        provider = _make_provider()
        runner = _make_stream_runner(provider)
        execution = provider._admission.start_execution()

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream,
        ):
            task = asyncio.create_task(
                runner._collect_recovery_output(
                    {"model": "test-model", "messages": []},
                    include_reasoning=True,
                    execution=execution,
                    operation_kind=ProviderOperationKind.CONTINUATION,
                )
            )
            await asyncio.wait_for(stream.entered.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        replacement = await execution.open_attempt(ProviderOperationKind.CONTINUATION)
        await replacement.aclose()
        assert stream.close_calls == 1

    @pytest.mark.asyncio
    async def test_recovery_collect_text_honors_provider_retry_classification(self):
        """Provider semantics apply before the first recovery chunk as well."""
        provider = _make_provider()
        runner = _make_stream_runner(provider)
        execution = provider._admission.start_execution()
        request = httpx2.Request(
            "POST", "https://test.api.nvidia.com/v1/chat/completions"
        )
        degraded = openai.BadRequestError(
            "Bad Request",
            response=httpx2.Response(400, request=request),
            body={
                "status": 400,
                "detail": (
                    "Function id 'test-function': DEGRADED function cannot be invoked"
                ),
            },
        )
        rejected = ClosableAsyncStreamMock([], error=degraded)
        recovered = ClosableAsyncStreamMock(
            [
                _make_chunk(content="world"),
                _make_chunk(finish_reason="stop"),
            ]
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[rejected, recovered],
        ) as create:
            result = await runner._collect_recovery_output(
                {"model": "test-model", "messages": []},
                include_reasoning=True,
                execution=execution,
                operation_kind=ProviderOperationKind.CONTINUATION,
            )

        assert result.text == "world"
        assert result.thinking == ""
        assert result.tool_calls == ()
        assert create.await_count == 2
        assert rejected.closed
        assert recovered.closed

    def test_text_recovery_body_preserves_thinking_context(self):
        """Continuation prompts include emitted thinking without provider-specific fields."""
        body = {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "Read"}],
            "tool_choice": {"type": "auto"},
        }

        recovery_body = make_text_recovery_body(
            body,
            partial_text="visible answer",
            partial_thinking="hidden reasoning",
        )

        assert "tools" not in recovery_body
        assert "tool_choice" not in recovery_body
        assert "stream" not in recovery_body
        assert recovery_body["messages"][-2] == {
            "role": "assistant",
            "content": "visible answer",
        }
        recovery_prompt = recovery_body["messages"][-1]
        assert recovery_prompt["role"] == "user"
        assert "hidden reasoning" in recovery_prompt["content"]
        assert "reasoning_content" not in recovery_prompt

    def test_tool_protocol_recovery_body_retains_tool_contract(self):
        body = {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "Read"}}],
            "tool_choice": {"type": "function", "function": {"name": "Read"}},
        }

        recovery_body = make_response_recovery_body(body, "visible answer")

        assert recovery_body["tools"] == body["tools"]
        assert recovery_body["tool_choice"] == body["tool_choice"]
        assert recovery_body["messages"][-2] == {
            "role": "assistant",
            "content": "visible answer",
        }

    @pytest.mark.asyncio
    async def test_openai_text_recovery_passes_thinking_context(self):
        """OpenAI-chat recovery call sites seed emitted thinking in the prompt."""
        runner = _make_stream_runner(
            _make_provider(), request=_make_request(), request_id="req_recovery"
        )
        assembler = runner._new_stream_assembler(output_reasoning=True)
        output = assembler.output
        output.ensure_reasoning_block()
        output.emit_reasoning_delta("hidden reasoning")
        output.ensure_text_block()
        output.emit_text_delta("visible answer")

        with patch.object(
            runner,
            "_collect_recovery_output",
            new_callable=AsyncMock,
            return_value=_recovery_output(
                text="visible answer done",
                thinking="hidden reasoning more",
            ),
        ) as mock_collect:
            execution = runner._transport._admission.start_execution()
            events = await runner._recovery_events(
                body={"messages": [{"role": "user", "content": "hello"}]},
                assembler=assembler,
                error=TimeoutError("cutoff"),
                tool_argument_alias_buffers={},
                output_reasoning=True,
                execution=execution,
            )

        assert events is not None
        assert mock_collect.await_args is not None
        recovery_body = mock_collect.await_args.args[0]
        assert "hidden reasoning" in recovery_body["messages"][-1]["content"]
        assert mock_collect.await_args.kwargs["include_reasoning"] is True

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
    async def test_truncated_recovery_stream_closes_block_then_raises(self):
        """Partial recovery bytes never become success or provider-owned wire errors."""
        provider = _make_provider()
        request = _make_request()
        original_text = "hello wor" + ("x" * 70_000)
        original_stream = AsyncStreamMock([_make_chunk(content=original_text)])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=original_stream,
            ) as mock_create,
            patch.object(
                _OpenAIChatStreamRunner,
                "_collect_recovery_output",
                new_callable=AsyncMock,
                side_effect=TruncatedProviderStreamError(
                    "Recovery stream ended without finish_reason."
                ),
            ) as mock_collect,
        ):
            events, error = await _collect_stream_and_error(provider, request)

        event_text = "".join(events)
        assert mock_create.await_count == 1
        assert mock_collect.await_count == 1
        assert original_text in event_text
        assert "world" not in event_text
        assert "Provider stream ended without finish_reason." in error.message
        assert "Provider stream ended without finish_reason." not in event_text
        parsed = parse_sse_text(event_text)
        assert parsed[-1].event == "content_block_stop"
        assert not any(event.event == "error" for event in parsed)
        assert not any(event.event == "message_stop" for event in parsed)
        assert not any(
            event.event == "content_block_delta"
            and event.data.get("delta", {}).get("text") == "ld"
            for event in parse_sse_text(event_text)
        )

    @pytest.mark.asyncio
    async def test_incomplete_tool_call_repair_appends_schema_valid_suffix(self):
        """A truncated tool JSON prefix is repaired append-only before tool_use tail."""
        provider = _make_provider()
        request = _make_request(
            tools=[
                {
                    "name": "echo_smoke",
                    "description": "Echo",
                    "input_schema": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                }
            ]
        )
        tool_chunk = _make_tool_calls_chunk(
            name="echo_smoke", arguments='{"message":', tool_id="call_repair"
        )
        stream_mock = AsyncStreamMock([tool_chunk])

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=stream_mock,
            ),
            patch.object(
                _OpenAIChatStreamRunner,
                "_collect_recovery_output",
                new_callable=AsyncMock,
                return_value=_recovery_output(text='"ok"}'),
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        parsed = parse_sse_text(event_text)
        assert '"partial_json": "\\"ok\\"}"' in event_text
        assert any(
            event.event == "message_delta"
            and event.data.get("delta", {}).get("stop_reason") == "tool_use"
            for event in parsed
        )
        assert not any(event.event == "error" for event in parsed)

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


class TestProcessToolCall:
    """Tests for OpenAI tool-call assembly."""

    def test_heuristic_tool_use_sse_marks_committed_tool_output(self):
        """Heuristic tool blocks are emitted content, even without OpenAI tool state."""
        output = _make_anthropic_output()
        events = list(
            iter_heuristic_tool_use_events(
                output,
                {
                    "id": "toolu_heuristic",
                    "name": "Read",
                    "input": {"path": "test.py"},
                },
            )
        )

        event_text = "".join(events)
        assert "tool_use" in event_text
        assert output.has_emitted_tool_block()
        assert output.committed_output

    def test_tool_call_with_id(self):
        """Tool call with id starts a tool block."""
        provider = _make_provider()
        sse = _make_anthropic_output()
        tc = {
            "index": 0,
            "id": "call_123",
            "function": {"name": "search", "arguments": '{"q": "test"}'},
        }
        events = list(_make_tool_assembler(provider).process_tool_call(tc, sse))
        assert _tool_use_starts(events) == [
            {
                "type": "tool_use",
                "id": "call_123",
                "name": "search",
                "input": {},
            }
        ]

    @pytest.mark.parametrize("missing_id", [None, "", "   "])
    def test_missing_or_blank_tool_call_id_generates_public_id(self, missing_id):
        """An absent identity never escapes as an empty Anthropic tool-use ID."""
        provider = _make_provider()
        sse = _make_anthropic_output()

        events = list(
            _make_tool_assembler(provider).process_tool_call(
                {
                    "index": 0,
                    "id": missing_id,
                    "function": {"name": "Bash", "arguments": "{}"},
                },
                sse,
            )
        )

        [start] = _tool_use_starts(events)
        assert start["id"].startswith("tool_")

    def test_historical_tool_call_id_collision_is_remapped(self):
        """A later turn cannot reuse a tool identity already visible in history."""
        provider = _make_provider()
        request = _make_request(
            messages=[
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
        )
        sse = _make_anthropic_output()

        events = list(
            _make_tool_assembler(provider, request=request).process_tool_call(
                {
                    "index": 0,
                    "id": "Bash:0",
                    "function": {
                        "name": "Bash",
                        "arguments": '{"command":"printf second"}',
                    },
                },
                sse,
            )
        )

        [start] = _tool_use_starts(events)
        assert start["id"].startswith("tool_")
        assert start["id"] != "Bash:0"

    def test_current_response_tool_call_id_collision_is_remapped(self):
        """Two simultaneous calls cannot expose the same public identity."""
        provider = _make_provider()
        sse = _make_anthropic_output()
        assembler = _make_tool_assembler(provider)

        first = list(
            assembler.process_tool_call(
                {
                    "index": 0,
                    "id": "Bash:0",
                    "function": {"name": "Bash", "arguments": "{}"},
                },
                sse,
            )
        )
        second = list(
            assembler.process_tool_call(
                {
                    "index": 1,
                    "id": "Bash:0",
                    "function": {"name": "Bash", "arguments": "{}"},
                },
                sse,
            )
        )

        starts = _tool_use_starts(first + second)
        assert starts[0]["id"] == "Bash:0"
        assert starts[1]["id"].startswith("tool_")
        assert starts[1]["id"] != starts[0]["id"]

    def test_structured_tool_call_restores_portable_wire_alias(self):
        """A wire alias never escapes in the Anthropic tool block."""
        provider = _make_provider()
        original = "mcp__portable_output__" + "x" * 70
        request = _make_request(
            tools=[{"name": original, "input_schema": {"type": "object"}}]
        )
        codec = OpenAIToolNameCodec.from_request(request)
        alias = codec.encode(original)
        sse = _make_anthropic_output()

        events = list(
            _make_tool_assembler(provider).process_tool_call(
                {
                    "index": 0,
                    "id": "call_alias",
                    "function": {"name": alias, "arguments": "{}"},
                },
                sse,
                tool_names=codec,
                tool_name_buffers={},
            )
        )

        event_text = "".join(events)
        assert original in event_text
        assert alias not in event_text
        assert (
            sum(
                event.event == "content_block_start"
                for event in parse_sse_text(event_text)
            )
            == 1
        )

    def test_tool_name_restores_before_nim_argument_alias_lookup(self):
        """Generic name decoding composes with original-name NIM arg metadata."""
        provider = _make_provider()
        original = "mcp__nim_argument_composition__" + "x" * 70
        request = _make_request(
            tools=[
                {
                    "name": original,
                    "input_schema": {
                        "type": "object",
                        "properties": {"type": {"type": "string"}},
                        "required": ["type"],
                    },
                }
            ]
        )
        codec = OpenAIToolNameCodec.from_request(request)
        sse = _make_anthropic_output()

        events = list(
            _make_tool_assembler(provider).process_tool_call(
                {
                    "index": 0,
                    "id": "call_composed",
                    "function": {
                        "name": codec.encode(original),
                        "arguments": '{"_fcc_arg_type":"file"}',
                    },
                },
                sse,
                tool_names=codec,
                tool_name_buffers={},
                tool_argument_aliases={original: {"_fcc_arg_type": "type"}},
                tool_argument_alias_buffers={},
            )
        )

        parsed = parse_sse_text("".join(events))
        start = next(
            event.data["content_block"]
            for event in parsed
            if event.event == "content_block_start"
        )
        argument_delta = next(
            event.data["delta"]["partial_json"]
            for event in parsed
            if event.event == "content_block_delta"
        )
        assert start["name"] == original
        assert json.loads(argument_delta) == {"type": "file"}

    def test_fragmented_wire_alias_starts_one_original_tool_block(self):
        """A split alias is held until the exact request alias is complete."""
        provider = _make_provider()
        original = "mcp__fragmented_output__" + "x" * 70
        request = _make_request(
            tools=[{"name": original, "input_schema": {"type": "object"}}]
        )
        codec = OpenAIToolNameCodec.from_request(request)
        alias = codec.encode(original)
        split = len(alias) // 2
        buffers: dict[int, str] = {}
        sse = _make_anthropic_output()
        assembler = _make_tool_assembler(provider)

        first = list(
            assembler.process_tool_call(
                {
                    "index": 0,
                    "id": "call_split_alias",
                    "function": {"name": alias[:split], "arguments": ""},
                },
                sse,
                tool_names=codec,
                tool_name_buffers=buffers,
            )
        )
        second = list(
            assembler.process_tool_call(
                {
                    "index": 0,
                    "id": None,
                    "function": {"name": alias[split:], "arguments": "{}"},
                },
                sse,
                tool_names=codec,
                tool_name_buffers=buffers,
            )
        )

        assert first == []
        event_text = "".join(second)
        assert original in event_text
        assert alias not in event_text
        assert (
            sum(
                event.event == "content_block_start"
                for event in parse_sse_text(event_text)
            )
            == 1
        )
        assert buffers == {}

    def test_valid_name_that_prefixes_alias_is_resolved_on_flush(self):
        """An ambiguous valid name is delayed, not mistaken for an alias fragment."""
        provider = _make_provider()
        original = "tool"
        long_name = "tool." + "x" * 70
        request = _make_request(
            tools=[
                {"name": original, "input_schema": {"type": "object"}},
                {"name": long_name, "input_schema": {"type": "object"}},
            ]
        )
        codec = OpenAIToolNameCodec.from_request(request)
        assert codec.is_alias_prefix(original)
        buffers: dict[int, str] = {}
        sse = _make_anthropic_output()
        assembler = _make_tool_assembler(provider)

        initial = list(
            assembler.process_tool_call(
                {
                    "index": 0,
                    "id": "call_valid",
                    "function": {"name": original, "arguments": "{}"},
                },
                sse,
                tool_names=codec,
                tool_name_buffers=buffers,
            )
        )
        flushed = list(
            assembler.flush_tool_name_buffers(
                sse,
                tool_names=codec,
                tool_name_buffers=buffers,
                tool_argument_aliases={},
                tool_argument_alias_buffers={},
            )
        )

        assert initial == []
        assert original in "".join(flushed)
        assert buffers == {}

    def test_buffered_collector_decodes_before_schema_validation(self):
        """Recovery validates the original tool identity, not its wire alias."""
        original = "mcp__recovery_output__" + "x" * 70
        request = _make_request(
            tools=[{"name": original, "input_schema": {"type": "object"}}]
        )
        codec = OpenAIToolNameCodec.from_request(request)
        collector = OpenAIToolCallCollector()
        collector.add(
            SimpleNamespace(
                index=0,
                id="call_recovered",
                function=SimpleNamespace(name=codec.encode(original), arguments="{}"),
            )
        )

        calls = collector.completed_calls(
            tool_schemas_by_name(request), tool_names=codec
        )

        assert calls is not None
        assert calls[0]["function"]["name"] == original

    def test_heuristic_tool_call_restores_original_name(self):
        """Complete heuristic calls share the same outbound name contract."""
        original = "mcp__heuristic_output__" + "x" * 70
        request = _make_request(
            tools=[{"name": original, "input_schema": {"type": "object"}}]
        )
        codec = OpenAIToolNameCodec.from_request(request)
        sse = _make_anthropic_output()

        events = list(
            iter_heuristic_tool_use_events(
                sse,
                {
                    "id": "call_heuristic",
                    "name": codec.encode(original),
                    "input": {},
                },
                tool_names=codec,
            )
        )

        event_text = "".join(events)
        assert original in event_text
        assert codec.encode(original) not in event_text

    def test_tool_call_id_arrives_before_name_still_emits_id_and_name(self):
        """Split-stream tool: id (no name) then name then args; id preserved on start."""
        provider = _make_provider()
        sse = _make_anthropic_output()
        t1 = {
            "index": 0,
            "id": "call_split",
            "function": {"name": None, "arguments": ""},
        }
        t2 = {
            "index": 0,
            "id": "call_split",
            "function": {"name": "Grep", "arguments": ""},
        }
        t3 = {
            "index": 0,
            "id": "call_split",
            "function": {"name": None, "arguments": "{}"},
        }
        assembler = _make_tool_assembler(provider)
        b1 = "".join(assembler.process_tool_call(t1, sse))
        b2 = "".join(assembler.process_tool_call(t2, sse))
        b3 = "".join(assembler.process_tool_call(t3, sse))
        combined = b1 + b2 + b3
        assert "call_split" in combined
        assert "Grep" in combined
        assert b1 == ""

    def test_tool_call_arguments_buffered_until_name(self):
        """Argument deltas before tool name are emitted after the block starts."""
        provider = _make_provider()
        sse = _make_anthropic_output()
        t1 = {
            "index": 0,
            "id": "call_buf",
            "function": {"name": None, "arguments": '{"x":'},
        }
        t2 = {
            "index": 0,
            "id": "call_buf",
            "function": {"name": "Read", "arguments": "1}"},
        }
        assembler = _make_tool_assembler(provider)
        b1 = "".join(assembler.process_tool_call(t1, sse))
        b2 = "".join(assembler.process_tool_call(t2, sse))
        assert b1 == ""
        combined = b2
        assert "Read" in combined
        assert "call_buf" in combined
        assert '{"x":' in combined or "partial_json" in combined

    def test_late_upstream_id_cannot_overwrite_started_public_id(self):
        """The identity emitted at block start stays authoritative."""
        provider = _make_provider()
        sse = _make_anthropic_output()
        assembler = _make_tool_assembler(provider)

        initial = list(
            assembler.process_tool_call(
                {
                    "index": 0,
                    "id": None,
                    "function": {"name": "Bash", "arguments": "{}"},
                },
                sse,
            )
        )
        [start] = _tool_use_starts(initial)
        generated_id = start["id"]

        later = list(
            assembler.process_tool_call(
                {
                    "index": 0,
                    "id": "late_upstream_id",
                    "function": {"name": None, "arguments": ""},
                },
                sse,
            )
        )

        assert later == []
        assert sse.tool_states[0].tool_id == generated_id

    def test_task_tool_forces_background_false(self):
        """Task tool with run_in_background=true is forced to false."""
        provider = _make_provider()
        sse = _make_anthropic_output()
        args = json.dumps({"run_in_background": True, "prompt": "test"})
        tc = {
            "index": 0,
            "id": "call_task",
            "function": {"name": "Task", "arguments": args},
        }
        events = list(_make_tool_assembler(provider).process_tool_call(tc, sse))
        event_text = "".join(events)
        # The intercepted args should have run_in_background=false
        assert "false" in event_text.lower()

    def test_task_tool_chunked_args_forces_background_false(self):
        """Chunked Task args are buffered until valid JSON, then forced to false."""
        provider = _make_provider()
        sse = _make_anthropic_output()
        tc1 = {
            "index": 0,
            "id": "call_task_chunked",
            "function": {"name": "Task", "arguments": '{"run_in_background": true,'},
        }
        tc2 = {
            "index": 0,
            "id": "call_task_chunked",
            "function": {"name": None, "arguments": ' "prompt": "test"}'},
        }

        assembler = _make_tool_assembler(provider)
        events1 = list(assembler.process_tool_call(tc1, sse))
        assert len(events1) > 0
        assert "false" not in "".join(events1).lower()

        events2 = list(assembler.process_tool_call(tc2, sse))
        event_text = "".join(events1 + events2)
        assert "false" in event_text.lower()

    def test_task_tool_invalid_json_logs_warning_on_flush(self, caplog):
        """Invalid JSON args for Task tool emits {} on flush and logs a warning."""
        provider = _make_provider()
        sse = _make_anthropic_output()
        tc = {
            "index": 0,
            "id": "call_task2",
            "function": {"name": "Task", "arguments": "not json"},
        }
        assembler = _make_tool_assembler(provider)
        events = list(assembler.process_tool_call(tc, sse))
        assert len(events) > 0

        with caplog.at_level("WARNING"):
            flushed = list(assembler.flush_task_arg_buffers(sse))
        assert len(flushed) > 0
        assert "{}" in "".join(flushed)
        assert any("Task args invalid JSON" in r.message for r in caplog.records)

    def test_negative_tool_index_fallback(self):
        """tc_index < 0 uses len(tool_indices) as fallback."""
        provider = _make_provider()
        sse = _make_anthropic_output()
        tc = {
            "index": -1,
            "id": "call_neg",
            "function": {"name": "test", "arguments": "{}"},
        }
        events = list(_make_tool_assembler(provider).process_tool_call(tc, sse))
        # Should not crash, should still emit events
        assert len(events) > 0

    def test_none_tool_index_defaults_to_zero(self):
        """Gemini may stream tool_call deltas with a null index."""
        provider = _make_provider()
        sse = _make_anthropic_output()
        tc = {
            "index": None,
            "id": "call_none",
            "function": {"name": "test", "arguments": "{}"},
        }
        events = list(_make_tool_assembler(provider).process_tool_call(tc, sse))
        event_text = "".join(events)

        assert "tool_use" in event_text
        assert "call_none" in event_text

    def test_tool_args_emitted_as_delta(self):
        """Arguments are emitted as input_json_delta events."""
        provider = _make_provider()
        sse = _make_anthropic_output()
        tc = {
            "index": 0,
            "id": "call_args",
            "function": {"name": "grep", "arguments": '{"pattern": "test"}'},
        }
        events = list(_make_tool_assembler(provider).process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "input_json_delta" in event_text


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

    def test_stream_malformed_tool_args_chunked(self):
        """Chunked tool args that never form valid JSON are flushed with {}."""
        provider = _make_provider()
        sse = _make_anthropic_output()
        tc1 = {
            "index": 0,
            "id": "call_malformed",
            "function": {"name": "Task", "arguments": '{"broken":'},
        }
        tc2 = {
            "index": 0,
            "id": "call_malformed",
            "function": {"name": None, "arguments": " never valid }"},
        }

        assembler = _make_tool_assembler(provider)
        events1 = list(assembler.process_tool_call(tc1, sse))
        events2 = list(assembler.process_tool_call(tc2, sse))
        flushed = list(assembler.flush_task_arg_buffers(sse))

        event_text = "".join(events1 + events2 + flushed)
        assert "tool_use" in event_text
        assert "{}" in event_text


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


@pytest.mark.asyncio
@pytest.mark.parametrize("collector", [False, True])
async def test_tool_argument_mapping_survives_correction_then_stream_reopen(collector):
    request = _alias_request()

    def responder(bodies):
        if len(bodies) == 1:
            return 400, {"message": "chat_template is unsupported"}
        if len(bodies) == 2:
            template = _events_for("chat")[0]
            return 200, [
                {
                    **template,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "discarded"},
                            "finish_reason": None,
                        }
                    ],
                }
            ]
        return 200, _alias_events()

    async with _harness("chat", responder, chat_provider_factory=_alias_provider) as (
        _,
        bodies,
        provider,
    ):
        if collector:
            runner = _make_stream_runner(provider, request=request)
            execution = provider._admission.start_execution()
            try:
                recovered = await runner._collect_recovery_output(
                    runner._body,
                    include_reasoning=False,
                    execution=execution,
                    operation_kind=ProviderOperationKind.CONTINUATION,
                )
                arguments = json.loads(recovered.tool_calls[0]["function"]["arguments"])
            finally:
                execution.abandon()
        else:
            saved = await _saved_reply(provider.stream_messages(request), "messages")
            call = next(
                block for block in saved[0]["content"] if block["type"] == "tool_use"
            )
            arguments = call["input"]
        assert arguments == {"pattern": "needle", "type": "py"}
        assert len(bodies) == 3
        assert all("chat_template" not in body for body in bodies[1:])
        assert all("_fcc_nim_tool_argument_aliases" not in body for body in bodies)
