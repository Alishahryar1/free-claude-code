"""Google compatibility signature behavior retained by Vertex."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.google_openai import (
    GOOGLE_SKIP_THOUGHT_SIGNATURE_VALIDATOR,
    GoogleOpenAIProvider,
)
from free_claude_code.providers.history_replay import replay_origin
from free_claude_code.providers.vertex.client import _PROFILE
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    SDKStreamDouble,
    immediate_admission,
    make_provider_config,
    reasoning_for,
)


def make_request(**overrides):
    return make_messages_request("models/gemini-3.1-flash-lite", **overrides)


@pytest.fixture
def gemini_provider():
    return GoogleOpenAIProvider(
        make_provider_config(api_key="test", base_url="https://google.invalid/v1"),
        profile=_PROFILE,
        admission=immediate_admission(),
    )


def test_build_request_body_preserves_tool_call_extra_content(gemini_provider):
    req = make_request(
        system=None,
        messages=[
            {"role": "user", "content": "Find files"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "function-call-1",
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                        "extra_content": {
                            "google": {"thought_signature": "sig-from-client"}
                        },
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "function-call-1",
                        "content": "[]",
                    },
                ],
            },
        ],
    )

    body = gemini_provider._chat._build_request_body(req, reasoning=reasoning_for(req))

    tool_call = body["messages"][1]["tool_calls"][0]
    assert tool_call["extra_content"] == {
        "google": {"thought_signature": "sig-from-client"}
    }


def cache_origin(provider):
    return replay_origin(
        "VERTEX", "chat", make_request().model, client=provider._client
    )


def test_build_request_body_uses_cached_tool_call_signature(gemini_provider):
    gemini_provider._behavior.record_tool_call_extra_content(
        "function-call-1",
        {"google": {"thought_signature": "sig-from-cache"}},
        origin=cache_origin(gemini_provider),
    )
    req = make_request(
        system=None,
        messages=[
            {"role": "user", "content": "Find files"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "function-call-1",
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "function-call-1",
                        "content": "[]",
                    },
                ],
            },
        ],
    )

    body = gemini_provider._chat._build_request_body(req, reasoning=reasoning_for(req))

    body = gemini_provider._behavior.prepare_history_body(
        body, cache_origin(gemini_provider)
    )

    tool_call = body["messages"][1]["tool_calls"][0]
    assert tool_call["extra_content"] == {
        "google": {"thought_signature": "sig-from-cache"}
    }


def test_build_request_body_adds_current_turn_fallback_signature(
    gemini_provider,
):
    req = make_request(
        system=None,
        messages=[
            {"role": "user", "content": "Find files"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "function-call-1",
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "function-call-2",
                        "name": "Read",
                        "input": {"file_path": "a.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "function-call-1",
                        "content": "[]",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "function-call-2",
                        "content": "contents",
                    },
                ],
            },
        ],
    )

    body = gemini_provider._chat._build_request_body(req, reasoning=reasoning_for(req))

    body = gemini_provider._behavior.prepare_history_body(
        body, cache_origin(gemini_provider)
    )

    tool_calls = body["messages"][1]["tool_calls"]
    assert tool_calls[0]["extra_content"] == {
        "google": {"thought_signature": GOOGLE_SKIP_THOUGHT_SIGNATURE_VALIDATOR}
    }
    assert "extra_content" not in tool_calls[1]


@pytest.mark.asyncio
async def test_stream_messages_text(gemini_provider):
    req = make_request(thinking={"type": "enabled"})

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello back!",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        gemini_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(mock_stream())

        events = [
            event
            async for event in gemini_provider.stream_messages(
                req, reasoning=reasoning_for(req)
            )
        ]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )
        kwargs = mock_create.call_args.kwargs
        assert "reasoning_effort" not in kwargs
        extra_body = kwargs.get("extra_body")
        assert isinstance(extra_body, dict)
        literal_extra_body = extra_body.get("extra_body")
        assert isinstance(literal_extra_body, dict)
        google = literal_extra_body.get("google")
        assert isinstance(google, dict)
        thinking_config = google.get("thinking_config")
        assert isinstance(thinking_config, dict)
        assert thinking_config.get("include_thoughts") is True


@pytest.mark.asyncio
async def test_stream_messages_preserves_tool_call_extra_content(gemini_provider):
    req = make_request()

    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "function-call-1"
    mock_tc.extra_content = {"google": {"thought_signature": "sig-stream"}}
    mock_tc.function = MagicMock()
    mock_tc.function.name = "Glob"
    mock_tc.function.arguments = '{"pattern":"*.py"}'

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content=None,
                tool_calls=[mock_tc],
            ),
            finish_reason="tool_calls",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        gemini_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(mock_stream())

        events = [event async for event in gemini_provider.stream_messages(req)]

    tool_starts = [
        event
        for event in events
        if '"content_block_start"' in event and '"tool_use"' in event
    ]
    assert any(
        '"extra_content"' in event and "sig-stream" in event for event in tool_starts
    )
    assert gemini_provider._behavior._tool_call_extra_content_by_id[
        (cache_origin(gemini_provider), "function-call-1")
    ] == {"google": {"thought_signature": "sig-stream"}}


@pytest.mark.asyncio
async def test_colliding_stream_tool_id_rekeys_cached_thought_signature(
    gemini_provider,
):
    """Gemini metadata follows the public ID returned to the client."""
    history = [
        {"role": "user", "content": "Find files once."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "function-call-1",
                    "name": "Glob",
                    "input": {"pattern": "*.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "function-call-1",
                    "content": "[]",
                }
            ],
        },
    ]
    request = make_request(
        system=None,
        messages=[*history, {"role": "user", "content": "Find files again."}],
    )
    tool_call = MagicMock()
    tool_call.index = 0
    tool_call.id = "function-call-1"
    tool_call.extra_content = {"google": {"thought_signature": "sig-stream"}}
    tool_call.function = MagicMock()
    tool_call.function.name = "Glob"
    tool_call.function.arguments = '{"pattern":"*.py"}'
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content=None,
                tool_calls=[tool_call],
            ),
            finish_reason="tool_calls",
        )
    ]
    chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield chunk

    with patch.object(
        gemini_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=SDKStreamDouble(mock_stream()),
    ):
        events = [event async for event in gemini_provider.stream_messages(request)]

    starts = [
        event.data["content_block"]
        for event in parse_sse_text("".join(events))
        if event.event == "content_block_start"
        and event.data.get("content_block", {}).get("type") == "tool_use"
    ]
    [start] = starts
    public_id = start["id"]
    assert public_id != "function-call-1"
    assert gemini_provider._behavior._tool_call_extra_content_by_id[
        (cache_origin(gemini_provider), public_id)
    ] == {"google": {"thought_signature": "sig-stream"}}

    replay = make_request(
        system=None,
        messages=[
            *request.messages,
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": public_id,
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": public_id,
                        "content": "[]",
                    }
                ],
            },
        ],
    )
    body = gemini_provider._chat._build_request_body(
        replay,
        reasoning=reasoning_for(replay),
    )
    body = gemini_provider._behavior.prepare_history_body(
        body, cache_origin(gemini_provider)
    )
    replayed_call = body["messages"][-2]["tool_calls"][0]
    assert replayed_call["id"] == public_id
    assert replayed_call["extra_content"] == {
        "google": {"thought_signature": "sig-stream"}
    }


@pytest.mark.asyncio
async def test_stream_messages_reasoning_content(gemini_provider):
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking...",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        gemini_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = SDKStreamDouble(mock_stream())

        events = [event async for event in gemini_provider.stream_messages(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )


@pytest.mark.asyncio
async def test_cleanup(gemini_provider):
    gemini_provider._client = AsyncMock()

    await gemini_provider.cleanup()

    gemini_provider._client.close.assert_called_once()
