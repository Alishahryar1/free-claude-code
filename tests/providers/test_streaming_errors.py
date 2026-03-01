"""Tests for streaming error handling in providers/nvidia_nim/client.py."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.nvidia_nim import NvidiaNimProvider


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


def _make_provider():
    """Create a provider instance for testing."""
    config = ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=10,
        rate_window=60,
    )
    return NvidiaNimProvider(config, nim_settings=NimSettings())


def _make_request(model="test-model", stream=True):
    """Create a mock request with all fields build_request_body needs."""
    req = MagicMock()
    req.model = model
    req.stream = stream
    req.messages = []
    req.system = None
    req.tools = None
    req.tool_choice = None
    req.metadata = None
    req.max_tokens = 4096
    req.temperature = None
    req.top_p = None
    req.top_k = None
    req.stop_sequences = None
    req.extra_body = None
    req.thinking = None
    return req


def _make_chunk(
    content=None, finish_reason=None, tool_calls=None, reasoning_content=None
):
    """Create a mock streaming chunk."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    delta.reasoning_content = reasoning_content if reasoning_content else None

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


async def _collect_stream(provider, request):
    """Collect all SSE events from a stream."""
    return [e async for e in provider.stream_response(request)]


class TestStreamingExceptionHandling:
    """Tests for error paths during stream_response."""

    @pytest.mark.asyncio
    async def test_api_error_emits_sse_error_event(self):
        """When API raises during streaming, SSE error event is emitted."""
        provider = _make_provider()
        request = _make_request()

        mock_stream = AsyncMock()
        mock_stream.__aiter__ = MagicMock(side_effect=RuntimeError("API failed"))

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API failed"),
            ),
            patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        # Should have message_start, error text block, close blocks, message_delta, message_stop
        event_text = "".join(events)
        assert "message_start" in event_text
        assert "API failed" in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_read_timeout_with_empty_message_emits_fallback(self):
        """ReadTimeout(TimeoutError()) should emit a visible, non-empty timeout message."""
        provider = _make_provider()
        request = _make_request()

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=httpx.ReadTimeout(""),
            ),
            patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = [
                e
                async for e in provider.stream_response(
                    request,
                    request_id="req_timeout123",
                )
            ]

        event_text = "".join(events)
        assert "timed out after" in event_text
        assert "request_id=req_timeout123" in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_error_after_partial_content(self):
        """Error after partial content: blocks closed, error emitted."""
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
            patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "Hello" in event_text
        assert "Connection lost" in event_text
        assert "message_stop" in event_text

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
            patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
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
            patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
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
            patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "thinking_delta" in event_text
        assert "I think..." in event_text
        assert "The answer" in event_text

    @pytest.mark.asyncio
    async def test_stream_rate_limited_retries_via_execute_with_retry(self):
        """When rate limited, execute_with_retry handles retries transparently."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(content="Response")
        chunk2 = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([chunk1, chunk2])

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            # Mock execute_with_retry to pass through to the actual function
            async def _passthrough(fn, *args, **kwargs):
                return await fn(*args, **kwargs)

            with patch.object(
                provider._global_rate_limiter,
                "execute_with_retry",
                new_callable=AsyncMock,
                side_effect=_passthrough,
            ):
                events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "Response" in event_text


class TestProcessToolCall:
    """Tests for _process_tool_call method."""

    def test_tool_call_with_id(self):
        """Tool call with id starts a tool block."""
        provider = _make_provider()
        from providers.common import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_123",
            "function": {"name": "search", "arguments": '{"q": "test"}'},
        }
        events = list(provider._process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "tool_use" in event_text
        assert "search" in event_text
        assert "call_123" in event_text

    def test_tool_call_without_id_generates_uuid(self):
        """Tool call without id generates a uuid-based id."""
        provider = _make_provider()
        from providers.common import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": None,
            "function": {"name": "test", "arguments": "{}"},
        }
        events = list(provider._process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "tool_" in event_text

    def test_task_tool_forces_background_false(self):
        """Task tool with run_in_background=true is forced to false."""
        provider = _make_provider()
        from providers.common import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        args = json.dumps({"run_in_background": True, "prompt": "test"})
        tc = {
            "index": 0,
            "id": "call_task",
            "function": {"name": "Task", "arguments": args},
        }
        events = list(provider._process_tool_call(tc, sse))
        event_text = "".join(events)
        # The intercepted args should have run_in_background=false
        assert "false" in event_text.lower()

    def test_task_tool_chunked_args_forces_background_false(self):
        """Chunked Task args are buffered until valid JSON, then forced to false."""
        provider = _make_provider()
        from providers.common import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
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

        events1 = list(provider._process_tool_call(tc1, sse))
        assert len(events1) > 0
        assert "false" not in "".join(events1).lower()

        events2 = list(provider._process_tool_call(tc2, sse))
        event_text = "".join(events1 + events2)
        assert "false" in event_text.lower()

    def test_task_tool_invalid_json_logs_warning_on_flush(self, caplog):
        """Invalid JSON args for Task tool emits {} on flush and logs a warning."""
        provider = _make_provider()
        from providers.common import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_task2",
            "function": {"name": "Task", "arguments": "not json"},
        }
        events = list(provider._process_tool_call(tc, sse))
        assert len(events) > 0

        with caplog.at_level("WARNING"):
            flushed = list(provider._flush_task_arg_buffers(sse))
        assert len(flushed) > 0
        assert "{}" in "".join(flushed)
        assert any("Task args invalid JSON" in r.message for r in caplog.records)

    def test_negative_tool_index_fallback(self):
        """tc_index < 0 uses len(tool_indices) as fallback."""
        provider = _make_provider()
        from providers.common import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": -1,
            "id": "call_neg",
            "function": {"name": "test", "arguments": "{}"},
        }
        events = list(provider._process_tool_call(tc, sse))
        # Should not crash, should still emit events
        assert len(events) > 0

    def test_tool_args_emitted_as_delta(self):
        """Arguments are emitted as input_json_delta events."""
        provider = _make_provider()
        from providers.common import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_args",
            "function": {"name": "grep", "arguments": '{"pattern": "test"}'},
        }
        events = list(provider._process_tool_call(tc, sse))
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
            patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
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
            patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
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
            patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "Partial" in event_text
        assert "Connection reset" in event_text
        assert "message_stop" in event_text

    def test_stream_malformed_tool_args_chunked(self):
        """Chunked tool args that never form valid JSON are flushed with {}."""
        provider = _make_provider()
        from providers.common import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
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

        events1 = list(provider._process_tool_call(tc1, sse))
        events2 = list(provider._process_tool_call(tc2, sse))
        flushed = list(provider._flush_task_arg_buffers(sse))

        event_text = "".join(events1 + events2 + flushed)
        assert "tool_use" in event_text
        assert "{}" in event_text
