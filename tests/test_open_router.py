"""Tests for OpenRouter provider."""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from providers.open_router import OpenRouterProvider
from providers.open_router.request import OPENROUTER_DEFAULT_MAX_TOKENS
from providers.base import ProviderConfig
from config.nim import NimSettings


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "stepfun/step-3.5-flash:free"
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
def open_router_config():
    return ProviderConfig(
        api_key="test_openrouter_key",
        base_url="https://openrouter.ai/api/v1",
        rate_limit=10,
        rate_window=60,
        nim_settings=NimSettings(),
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.open_router.client.GlobalRateLimiter") as mock:
        instance = mock.get_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def open_router_provider(open_router_config):
    return OpenRouterProvider(open_router_config)


def test_init(open_router_config):
    """Test provider initialization."""
    with patch("providers.open_router.client.AsyncOpenAI") as mock_openai:
        provider = OpenRouterProvider(open_router_config)
        assert provider._api_key == "test_openrouter_key"
        assert provider._base_url == "https://openrouter.ai/api/v1"
        mock_openai.assert_called_once()


def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""
    config = ProviderConfig(
        api_key="test_openrouter_key",
        base_url="https://openrouter.ai/api/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("providers.open_router.client.AsyncOpenAI") as mock_openai:
        OpenRouterProvider(config)
        call_kwargs = mock_openai.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


def test_build_request_body_has_reasoning_extra(open_router_provider):
    """Request body has extra_body.reasoning.enabled for thinking models."""
    req = MockRequest()
    body = open_router_provider._build_request_body(req)

    assert body["model"] == "stepfun/step-3.5-flash:free"
    assert body["temperature"] == 0.5
    assert len(body["messages"]) == 2  # System + User
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "System prompt"

    assert "extra_body" in body
    assert "reasoning" in body["extra_body"]
    assert body["extra_body"]["reasoning"]["enabled"] is True


def test_build_request_body_base_url_and_model(open_router_provider):
    """Base URL and model are correct in provider config."""
    assert open_router_provider._base_url == "https://openrouter.ai/api/v1"
    req = MockRequest(model="stepfun/step-3.5-flash:free")
    body = open_router_provider._build_request_body(req)
    assert body["model"] == "stepfun/step-3.5-flash:free"


def test_build_request_body_default_max_tokens(open_router_provider):
    """max_tokens=None uses OPENROUTER_DEFAULT_MAX_TOKENS (81920)."""
    req = MockRequest(max_tokens=None)
    body = open_router_provider._build_request_body(req)
    assert body["max_tokens"] == OPENROUTER_DEFAULT_MAX_TOKENS
    assert body["max_tokens"] == 81920


@pytest.mark.asyncio
async def test_stream_response_text(open_router_provider):
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
        open_router_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = []
        async for event in open_router_provider.stream_response(req):
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
async def test_stream_response_reasoning_content(open_router_provider):
    """Test streaming with reasoning_content delta."""
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
        open_router_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = []
        async for event in open_router_provider.stream_response(req):
            events.append(event)

        found_thinking = False
        for e in events:
            if "event: content_block_delta" in e and '"thinking_delta"' in e:
                if "Thinking..." in e:
                    found_thinking = True
        assert found_thinking
