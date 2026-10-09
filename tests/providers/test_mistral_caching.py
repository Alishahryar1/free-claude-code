"""Tests for Mistral prompt-cache affinity (x-affinity header) wiring."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from httpx2 import Request, Response

from free_claude_code.config.provider_catalog import (
    MISTRAL_DEFAULT_BASE,
    XAI_DEFAULT_BASE,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest
from free_claude_code.providers.mistral import MistralProvider
from free_claude_code.providers.mistral.caching import (
    MISTRAL_AFFINITY_HEADER,
    MISTRAL_SESSION_HEADER_NAMES,
    extract_mistral_affinity_key,
    mistral_affinity_headers,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    SDKStreamDouble,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

SESSION_HEADERS = {"x-opencode-session": "sess-123"}


@pytest.fixture
def mistral_config():
    return make_provider_config(
        api_key="test_mistral_key",
        base_url=MISTRAL_DEFAULT_BASE,
    )


@pytest.fixture
def mistral_provider(mistral_config):
    return MistralProvider(mistral_config, admission=immediate_admission())


def _text_chunk(text: str = "Hello back!"):
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(content=text, reasoning_content=None, tool_calls=None),
            finish_reason="stop",
        )
    ]
    chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)
    return chunk


async def _single_chunk_stream():
    yield _text_chunk()


def _make_bad_request_error(message: str) -> openai.BadRequestError:
    request = Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = Response(400, request=request)
    body = {"error": {"message": message}}
    return openai.BadRequestError(message, response=response, body=body)


def test_extract_prefers_the_most_specific_session_header():
    headers = {
        "session-id": "generic",
        "anthropic-session-id": "anthropic",
        "x-opencode-session": "opencode",
    }
    assert extract_mistral_affinity_key(headers) == "opencode"


@pytest.mark.parametrize("name", MISTRAL_SESSION_HEADER_NAMES)
def test_extract_accepts_every_known_session_header(name):
    assert extract_mistral_affinity_key({name: "sess-1"}) == "sess-1"


def test_extract_matches_header_names_case_insensitively():
    assert extract_mistral_affinity_key({"X-OpenCode-Session": "sess-2"}) == "sess-2"


def test_extract_strips_surrounding_whitespace():
    assert extract_mistral_affinity_key({"x-opencode-session": "  sess-3 "}) == "sess-3"


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"x-opencode-session": ""},
        {"x-opencode-session": "   "},
        {"unrelated-header": "value"},
    ],
)
def test_extract_returns_none_without_a_usable_session_header(headers):
    assert extract_mistral_affinity_key(headers) is None


@pytest.mark.parametrize("blank", ["", "   "])
def test_extract_falls_through_blank_preferred_header(blank):
    headers = {
        "x-opencode-session": blank,
        "anthropic-session-id": "anthropic",
    }
    assert extract_mistral_affinity_key(headers) == "anthropic"


@pytest.mark.parametrize("headers", [None, {}])
def test_affinity_headers_are_empty_without_a_session_header(headers):
    assert mistral_affinity_headers(headers) == {}


def test_affinity_headers_forward_only_the_affinity_header():
    headers = {
        "authorization": "Bearer secret",
        "user-agent": "fcc-test/1.0",
        "x-opencode-session": "sess-4",
    }
    assert mistral_affinity_headers(headers) == {MISTRAL_AFFINITY_HEADER: "sess-4"}


@pytest.mark.asyncio
async def test_stream_messages_sends_affinity_header(mistral_provider):
    req = make_messages_request("devstral-small-latest")

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(_single_chunk_stream())

        events = [
            event
            async for event in mistral_provider.stream_messages(
                req, request_headers=SESSION_HEADERS
            )
        ]

    assert any("Hello back!" in event for event in events)
    assert mock_create.await_args.kwargs["extra_headers"] == {
        MISTRAL_AFFINITY_HEADER: "sess-123"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_headers",
    [
        None,
        {},
        {"authorization": "Bearer secret", "user-agent": "fcc-test/1.0"},
    ],
)
async def test_stream_messages_without_session_header_sends_no_extra_headers(
    mistral_provider, request_headers
):
    req = make_messages_request("devstral-small-latest")

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(_single_chunk_stream())

        events = [
            event
            async for event in mistral_provider.stream_messages(
                req, request_headers=request_headers
            )
        ]

    assert any("Hello back!" in event for event in events)
    assert "extra_headers" not in mock_create.await_args.kwargs


@pytest.mark.asyncio
async def test_stream_messages_affinity_header_survives_reasoning_retry(
    mistral_provider,
):
    req = make_messages_request("devstral-small-latest")
    error = _make_bad_request_error("Unsupported field: reasoning_effort")

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, SDKStreamDouble(_single_chunk_stream())]

        events = [
            event
            async for event in mistral_provider.stream_messages(
                req,
                reasoning=reasoning_for(req),
                request_headers=SESSION_HEADERS,
            )
        ]

    assert mock_create.await_count == 2
    for call in mock_create.await_args_list:
        assert call.kwargs["extra_headers"] == {MISTRAL_AFFINITY_HEADER: "sess-123"}
    assert any("Hello back!" in event for event in events)


@pytest.mark.asyncio
async def test_stream_responses_sends_affinity_header(mistral_provider):
    request = OpenAIResponsesRequest(model="devstral-small-latest", input="hi")

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(_single_chunk_stream())

        chunks = [
            chunk
            async for chunk in mistral_provider.stream_responses(
                request, request_headers=SESSION_HEADERS
            )
        ]

    assert chunks
    assert mock_create.await_args.kwargs["extra_headers"] == {
        MISTRAL_AFFINITY_HEADER: "sess-123"
    }


@pytest.mark.asyncio
async def test_stream_responses_without_session_header_sends_no_extra_headers(
    mistral_provider,
):
    request = OpenAIResponsesRequest(model="devstral-small-latest", input="hi")

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(_single_chunk_stream())

        chunks = [chunk async for chunk in mistral_provider.stream_responses(request)]

    assert chunks
    assert "extra_headers" not in mock_create.await_args.kwargs


@pytest.mark.asyncio
async def test_other_openai_chat_providers_do_not_send_affinity_headers():
    provider = profiled_provider(
        "xai",
        make_provider_config(api_key="test-xai-key", base_url=XAI_DEFAULT_BASE),
        admission=immediate_admission(provider_name="xai"),
    )
    req = make_messages_request("grok-4.5")

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(_single_chunk_stream())

        events = [
            event
            async for event in provider.stream_messages(
                req, request_headers=SESSION_HEADERS
            )
        ]

    assert any("Hello back!" in event for event in events)
    assert "extra_headers" not in mock_create.await_args.kwargs


@pytest.mark.asyncio
async def test_stream_messages_maps_cached_usage_to_anthropic_fields(mistral_provider):
    req = make_messages_request("devstral-small-latest")

    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="hello", reasoning_content=None, tool_calls=None
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None, reasoning_content=None, tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                completion_tokens=5,
                prompt_tokens=1000,
                prompt_tokens_details={"cached_tokens": 640},
            ),
        )

    with patch.object(
        mistral_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(fake_stream())

        chunks = [
            chunk
            async for chunk in mistral_provider.stream_messages(
                req, request_headers=SESSION_HEADERS
            )
        ]

    parsed = parse_sse_text("".join(chunks))
    usage = next(
        event.data["usage"] for event in parsed if event.event == "message_delta"
    )
    assert usage == {
        "input_tokens": 360,
        "output_tokens": 5,
        "cache_read_input_tokens": 640,
    }
