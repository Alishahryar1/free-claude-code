"""Tests for streaming error handling in providers/nvidia_nim/client.py."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from providers.nvidia_nim import NvidiaNimProvider
from providers.base import ProviderConfig
from config.nim import NimSettings


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
        nim_settings=NimSettings(),
    )
    return NvidiaNimProvider(config)


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
    events = []
    async for event in provider.stream_response(request):
        events.append(event)
    return events


class TestStreamingExceptionHandling:
    """Tests for error paths during stream_response."""

    @pytest.mark.asyncio
    async def test_api_error_emits_sse_error_event(self):
        """When API raises during streaming, SSE error event is emitted."""
        provider = _make_provider()
        request = _make_request()

        mock_stream = AsyncMock()
        mock_stream.__aiter__ = MagicMock(side_effect=RuntimeError("API failed"))

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API failed"),
        ):
            with patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ):
                events = await _collect_stream(provider, request)

        # Should have message_start, error text block, close blocks, message_delta, message_stop, done
        event_text = "".join(events)
        assert "message_start" in event_text
        assert "API failed" in event_text
        assert "message_stop" in event_text
        assert "[DONE]" in event_text

    @pytest.mark.asyncio
    async def test_error_after_partial_content(self):
        """Error after partial content: blocks closed, error emitted."""
        provider = _make_provider()
        request = _make_request()

        chunk1 = _make_chunk(content="Hello ")
        stream_mock = AsyncStreamMock([chunk1], error=RuntimeError("Connection lost"))

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            with patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
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

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            with patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
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

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            with patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
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

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            with patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ):
                events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "thinking_delta" in event_text
        assert "I think..." in event_text
        assert "The answer" in event_text

    @pytest.mark.asyncio
    async def test_stream_rate_limited_shows_notice(self):
        """When globally rate limited, a notice is shown before stream starts."""
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
            with patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=True,
            ):
                events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "rate limit" in event_text.lower()
        assert "Response" in event_text


class TestProcessToolCall:
    """Tests for _process_tool_call method."""

    def test_tool_call_with_id(self):
        """Tool call with id starts a tool block."""
        provider = _make_provider()
        from providers.nvidia_nim.utils import SSEBuilder

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
        from providers.nvidia_nim.utils import SSEBuilder

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
        from providers.nvidia_nim.utils import SSEBuilder

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

    def test_task_tool_invalid_json_logs_warning(self):
        """Invalid JSON args for Task tool doesn't crash."""
        provider = _make_provider()
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_task2",
            "function": {"name": "Task", "arguments": "not json"},
        }
        # Should not raise
        events = list(provider._process_tool_call(tc, sse))
        assert len(events) > 0

    def test_negative_tool_index_fallback(self):
        """tc_index < 0 uses len(tool_indices) as fallback."""
        provider = _make_provider()
        from providers.nvidia_nim.utils import SSEBuilder

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
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_args",
            "function": {"name": "grep", "arguments": '{"pattern": "test"}'},
        }
        events = list(provider._process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "input_json_delta" in event_text
