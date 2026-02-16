"""Tests for LM Studio provider."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from providers.base import ProviderConfig
from providers.lmstudio import LMStudioProvider
from providers.lmstudio.request import LMSTUDIO_DEFAULT_MAX_TOKENS
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


def _make_request(model="test-model", **kwargs):
    """Create a mock request with all fields build_request_body needs."""
    req = MagicMock()
    req.model = model
    req.messages = [MagicMock(role="user", content="Hello")]
    req.system = None
    req.max_tokens = 100
    req.temperature = None
    req.top_p = None
    req.stop_sequences = None
    req.tools = None
    req.tool_choice = None
    req.thinking = MagicMock(enabled=True)
    for k, v in kwargs.items():
        setattr(req, k, v)
    return req


async def _collect_stream(provider, request):
    """Collect all SSE events from a stream."""
    events = []
    async for event in provider.stream_response(request):
        events.append(event)
    return events


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "lmstudio-community/qwen2.5-7b-instruct"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def lmstudio_config():
    return ProviderConfig(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        rate_limit=10,
        rate_window=60,
        nim_settings=NimSettings(),
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.lmstudio.client.GlobalRateLimiter") as mock:
        instance = mock.get_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def lmstudio_provider(lmstudio_config):
    return LMStudioProvider(lmstudio_config)


def test_init(lmstudio_config):
    """Test provider initialization."""
    with patch("providers.lmstudio.client.AsyncOpenAI") as mock_openai:
        provider = LMStudioProvider(lmstudio_config)
        assert provider._api_key == "lm-studio"
        assert provider._base_url == "http://localhost:1234/v1"
        mock_openai.assert_called_once()


def test_init_with_empty_api_key():
    """Provider uses lm-studio placeholder when api_key is empty."""
    config = ProviderConfig(
        api_key="",
        base_url="http://localhost:1234/v1",
        rate_limit=10,
        rate_window=60,
        nim_settings=NimSettings(),
    )
    with patch("providers.lmstudio.client.AsyncOpenAI"):
        provider = LMStudioProvider(config)
        assert provider._api_key == "lm-studio"


def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""
    config = ProviderConfig(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("providers.lmstudio.client.AsyncOpenAI") as mock_openai:
        LMStudioProvider(config)
        call_kwargs = mock_openai.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


def test_build_request_body_no_extra_body(lmstudio_provider):
    """LM Studio request body does NOT include extra_body/reasoning."""
    req = MockRequest()
    body = lmstudio_provider._build_request_body(req)

    assert body["model"] == "lmstudio-community/qwen2.5-7b-instruct"
    assert body["temperature"] == 0.5
    assert len(body["messages"]) == 2  # System + User
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "System prompt"

    assert "extra_body" not in body


def test_build_request_body_base_url_and_model(lmstudio_provider):
    """Base URL and model are correct in provider config."""
    assert lmstudio_provider._base_url == "http://localhost:1234/v1"
    req = MockRequest(model="lmstudio-community/qwen2.5-7b-instruct")
    body = lmstudio_provider._build_request_body(req)
    assert body["model"] == "lmstudio-community/qwen2.5-7b-instruct"


@pytest.mark.asyncio
async def test_stream_response_text(lmstudio_provider):
    """Test streaming text response."""
    req = MockRequest()

    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [
        MagicMock(
            delta=MagicMock(content="Hello", reasoning_content=None),
            finish_reason=None,
        )
    ]
    mock_chunk1.usage = None

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [
        MagicMock(
            delta=MagicMock(content=" World", reasoning_content=None),
            finish_reason="stop",
        )
    ]
    mock_chunk2.usage = MagicMock(completion_tokens=10)

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    with patch.object(
        lmstudio_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = []
        async for event in lmstudio_provider.stream_response(req):
            events.append(event)

        assert len(events) > 0
        assert "event: message_start" in events[0]

        text_content = ""
        for e in events:
            if "event: content_block_delta" in e and '"text_delta"' in e:
                for line in e.splitlines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "delta" in data and "text" in data["delta"]:
                            text_content += data["delta"]["text"]

        assert "Hello World" in text_content


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(lmstudio_provider):
    """Test streaming with reasoning_content delta (if LM Studio adds support)."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="Thinking..."),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        lmstudio_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = []
        async for event in lmstudio_provider.stream_response(req):
            events.append(event)

        found_thinking = False
        for e in events:
            if "event: content_block_delta" in e and '"thinking_delta"' in e:
                if "Thinking..." in e:
                    found_thinking = True
        assert found_thinking


# --- Stream Error Handling ---


class TestLMStudioStreamingExceptionHandling:
    """Tests for error paths during stream_response."""

    @pytest.mark.asyncio
    async def test_api_error_emits_sse_error_event(self, lmstudio_provider):
        """When API raises during streaming, SSE error event is emitted."""
        request = _make_request()

        with patch.object(
            lmstudio_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API failed"),
        ):
            events = await _collect_stream(lmstudio_provider, request)

        event_text = "".join(events)
        assert "message_start" in event_text
        assert "API failed" in event_text
        assert "message_stop" in event_text
        assert "[DONE]" in event_text

    @pytest.mark.asyncio
    async def test_error_after_partial_content(self, lmstudio_provider):
        """Error after partial content: blocks closed, error emitted."""
        request = _make_request()
        chunk1 = _make_chunk(content="Hello ")
        stream_mock = AsyncStreamMock(
            [chunk1], error=ConnectionResetError("Connection lost")
        )

        with patch.object(
            lmstudio_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events = await _collect_stream(lmstudio_provider, request)

        event_text = "".join(events)
        assert "Hello" in event_text
        assert "Connection lost" in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_empty_response_gets_space(self, lmstudio_provider):
        """Empty response with no text/tools gets a single space text block."""
        request = _make_request()
        empty_chunk = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([empty_chunk])

        with patch.object(
            lmstudio_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events = await _collect_stream(lmstudio_provider, request)

        event_text = "".join(events)
        assert '"text_delta"' in event_text
        assert "message_stop" in event_text


# --- Stream Chunk Edge Cases ---


class TestLMStudioStreamChunkEdgeCases:
    """Tests for edge cases in stream chunk handling."""

    @pytest.mark.asyncio
    async def test_stream_chunk_with_empty_choices_skipped(self, lmstudio_provider):
        """Chunk with choices=[] is skipped without crashing."""
        request = _make_request()
        empty_choices_chunk = MagicMock()
        empty_choices_chunk.choices = []
        empty_choices_chunk.usage = None
        finish_chunk = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([empty_choices_chunk, finish_chunk])

        with patch.object(
            lmstudio_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events = await _collect_stream(lmstudio_provider, request)

        event_text = "".join(events)
        assert "message_start" in event_text
        assert "message_stop" in event_text
        assert "[DONE]" in event_text

    @pytest.mark.asyncio
    async def test_stream_chunk_with_none_delta_handled(self, lmstudio_provider):
        """Chunk with choice.delta=None is handled defensively."""
        request = _make_request()
        none_delta_chunk = MagicMock()
        none_delta_chunk.usage = None
        choice = MagicMock()
        choice.delta = None
        choice.finish_reason = None
        none_delta_chunk.choices = [choice]
        finish_chunk = _make_chunk(finish_reason="stop")
        stream_mock = AsyncStreamMock([none_delta_chunk, finish_chunk])

        with patch.object(
            lmstudio_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=stream_mock,
        ):
            events = await _collect_stream(lmstudio_provider, request)

        event_text = "".join(events)
        assert "message_start" in event_text
        assert "message_stop" in event_text
        assert "[DONE]" in event_text


# --- Native Tool Calls ---


@pytest.mark.asyncio
async def test_stream_response_tool_call(lmstudio_provider):
    """Test streaming tool calls."""
    request = _make_request()
    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "call_1"
    mock_tc.function.name = "search"
    mock_tc.function.arguments = '{"q": "test"}'

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content=None, tool_calls=[mock_tc]),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        lmstudio_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_stream(),
    ):
        events = []
        async for event in lmstudio_provider.stream_response(request):
            events.append(event)

    starts = [
        e for e in events if "event: content_block_start" in e and '"tool_use"' in e
    ]
    assert len(starts) == 1
    assert "search" in starts[0]


# --- Think Tag Parsing ---


@pytest.mark.asyncio
async def test_stream_response_think_tag_parsing(lmstudio_provider):
    """Thinking content via think tags is emitted as thinking blocks."""
    request = _make_request()
    chunk1 = _make_chunk(content="<think>reasoning</think>answer")
    chunk2 = _make_chunk(finish_reason="stop")
    stream_mock = AsyncStreamMock([chunk1, chunk2])

    with patch.object(
        lmstudio_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream_mock,
    ):
        events = await _collect_stream(lmstudio_provider, request)

    event_text = "".join(events)
    assert "thinking" in event_text
    assert "reasoning" in event_text
    assert "answer" in event_text


# --- _process_tool_call and _flush_task_arg_buffers ---


class TestLMStudioProcessToolCall:
    """Tests for _process_tool_call method."""

    def test_tool_call_with_id(self, lmstudio_provider):
        """Tool call with id starts a tool block."""
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_123",
            "function": {"name": "search", "arguments": '{"q": "test"}'},
        }
        events = list(lmstudio_provider._process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "tool_use" in event_text
        assert "search" in event_text
        assert "call_123" in event_text

    def test_tool_call_without_id_generates_uuid(self, lmstudio_provider):
        """Tool call without id generates a uuid-based id."""
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": None,
            "function": {"name": "test", "arguments": "{}"},
        }
        events = list(lmstudio_provider._process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "tool_" in event_text

    def test_task_tool_forces_background_false(self, lmstudio_provider):
        """Task tool with run_in_background=true is forced to false."""
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        args = json.dumps({"run_in_background": True, "prompt": "test"})
        tc = {
            "index": 0,
            "id": "call_task",
            "function": {"name": "Task", "arguments": args},
        }
        events = list(lmstudio_provider._process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "false" in event_text.lower()

    def test_task_tool_chunked_args_forces_background_false(self, lmstudio_provider):
        """Chunked Task args are buffered until valid JSON, then forced to false."""
        from providers.nvidia_nim.utils import SSEBuilder

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

        events1 = list(lmstudio_provider._process_tool_call(tc1, sse))
        assert len(events1) > 0
        assert "false" not in "".join(events1).lower()

        events2 = list(lmstudio_provider._process_tool_call(tc2, sse))
        event_text = "".join(events1 + events2)
        assert "false" in event_text.lower()

    def test_task_tool_invalid_json_logs_warning_on_flush(
        self, lmstudio_provider, caplog
    ):
        """Invalid JSON args for Task tool emits {} on flush and logs a warning."""
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_task2",
            "function": {"name": "Task", "arguments": "not json"},
        }
        events = list(lmstudio_provider._process_tool_call(tc, sse))
        assert len(events) > 0

        with caplog.at_level("WARNING"):
            flushed = list(lmstudio_provider._flush_task_arg_buffers(sse))
        assert len(flushed) > 0
        assert "{}" in "".join(flushed)
        assert any("Task args invalid JSON" in r.message for r in caplog.records)

    def test_negative_tool_index_fallback(self, lmstudio_provider):
        """tc_index < 0 uses len(tool_indices) as fallback."""
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": -1,
            "id": "call_neg",
            "function": {"name": "test", "arguments": "{}"},
        }
        events = list(lmstudio_provider._process_tool_call(tc, sse))
        assert len(events) > 0

    def test_tool_args_emitted_as_delta(self, lmstudio_provider):
        """Arguments are emitted as input_json_delta events."""
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_args",
            "function": {"name": "grep", "arguments": '{"pattern": "test"}'},
        }
        events = list(lmstudio_provider._process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "input_json_delta" in event_text

    def test_stream_malformed_tool_args_chunked(self, lmstudio_provider):
        """Chunked tool args that never form valid JSON are flushed with {}."""
        from providers.nvidia_nim.utils import SSEBuilder

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

        events1 = list(lmstudio_provider._process_tool_call(tc1, sse))
        events2 = list(lmstudio_provider._process_tool_call(tc2, sse))
        flushed = list(lmstudio_provider._flush_task_arg_buffers(sse))

        event_text = "".join(events1 + events2 + flushed)
        assert "tool_use" in event_text
        assert "{}" in event_text


# --- Request Body Edge Cases ---


def test_build_request_body_max_tokens_default(lmstudio_provider):
    """max_tokens=None or 0 uses LMSTUDIO_DEFAULT_MAX_TOKENS."""
    req = MockRequest(max_tokens=None)
    body = lmstudio_provider._build_request_body(req)
    assert body["max_tokens"] == LMSTUDIO_DEFAULT_MAX_TOKENS
    assert body["max_tokens"] == 81920

    req2 = MockRequest(max_tokens=0)
    body2 = lmstudio_provider._build_request_body(req2)
    assert body2["max_tokens"] == LMSTUDIO_DEFAULT_MAX_TOKENS


def test_build_request_body_stop_sequences(lmstudio_provider):
    """stop_sequences non-empty adds stop key to body."""
    req = MockRequest(stop_sequences=["STOP", "END"])
    body = lmstudio_provider._build_request_body(req)
    assert body["stop"] == ["STOP", "END"]


def test_build_request_body_tools_and_tool_choice(lmstudio_provider):
    """tools and tool_choice non-empty add to body."""
    tool = MagicMock()
    tool.name = "test_tool"
    tool.description = "A test"
    tool.input_schema = {"type": "object"}
    req = MockRequest(tools=[tool], tool_choice="auto")
    body = lmstudio_provider._build_request_body(req)
    assert "tools" in body
    assert body["tool_choice"] == "auto"


# --- Base URL Trailing Slash ---


def test_init_base_url_strips_trailing_slash():
    """Config with base_url trailing slash is stored without it."""
    config = ProviderConfig(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1/",
        rate_limit=10,
        rate_window=60,
        nim_settings=NimSettings(),
    )
    with patch("providers.lmstudio.client.AsyncOpenAI"):
        provider = LMStudioProvider(config)
        assert provider._base_url == "http://localhost:1234/v1"
