"""Tests for LM Studio (OpenAI-compatible chat completions) provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.provider_catalog import LMSTUDIO_DEFAULT_BASE
from free_claude_code.core.anthropic import get_token_count
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.lmstudio import LMStudioProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
)


def make_request(**overrides):
    return make_messages_request("lmstudio-community/qwen2.5-7b-instruct", **overrides)


@pytest.fixture
def lmstudio_config():
    return make_provider_config(
        api_key="lm-studio",
        base_url=LMSTUDIO_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def lmstudio_provider(lmstudio_config):
    return LMStudioProvider(lmstudio_config, admission=immediate_admission())


def test_init(lmstudio_config):
    """Test provider initialization."""
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_openai:
        provider = LMStudioProvider(lmstudio_config, admission=immediate_admission())
        assert provider._api_key == "lm-studio"
        assert provider._base_url == LMSTUDIO_DEFAULT_BASE
        assert provider._provider_name == "LMSTUDIO"
        mock_openai.assert_called_once()


def test_default_base_url_constant():
    assert LMSTUDIO_DEFAULT_BASE == "http://localhost:1234/v1"


def test_build_request_body_basic(lmstudio_provider):
    req = make_request()
    body = lmstudio_provider._chat._build_request_body(req)

    assert body["model"] == "lmstudio-community/qwen2.5-7b-instruct"
    assert body["messages"][0]["role"] == "system"


def test_adaptive_client_reasoning_uses_documented_named_effort(lmstudio_provider):
    req = make_request()

    body = lmstudio_provider._chat._build_request_body(req, reasoning=REASONING_ON)

    assert body["extra_body"]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in body


def test_exact_client_budget_is_not_derived_from_output_tokens(lmstudio_provider):
    req = make_request(max_tokens=8192)

    body = lmstudio_provider._chat._build_request_body(
        req,
        reasoning=ReasoningPolicy.on(
            effort=ReasoningEffort.HIGH,
            budget_tokens=1024,
        ),
    )

    assert body["extra_body"]["reasoning_tokens"] == 1024
    assert "reasoning_tokens" not in body
    assert body["max_tokens"] == 8192


def test_build_request_body_never_replays_prior_thinking(lmstudio_provider):
    """Retain prior thinking as ordinary context without a reasoning role."""
    req = make_request(
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "prior reasoning",
                        "signature": "s",
                    }
                ],
            },
        ]
    )
    body = lmstudio_provider._chat._build_request_body(req)

    roles = [m.get("role") for m in body.get("messages", [])]
    assert "assistant_reasoning_content" not in roles
    assert "[Earlier reasoning]" in str(body)
    assert "prior reasoning" in str(body)


def test_preflight_builds_before_context_budget_and_preserves_policy(
    lmstudio_provider,
):
    request = make_request()
    calls: list[tuple[str, object]] = []

    def build(request_arg, *, reasoning, model_info):
        assert request_arg is request
        assert model_info is None
        calls.append(("build", reasoning))
        return {}

    def check_context(estimate: int):
        calls.append(("context", estimate))

    with (
        patch.object(lmstudio_provider._chat, "_build_request_body", side_effect=build),
        patch.object(
            lmstudio_provider,
            "_preflight_context_budget",
            side_effect=check_context,
        ),
    ):
        lmstudio_provider.preflight_messages(request, reasoning=REASONING_OFF)

    assert calls == [
        ("build", REASONING_OFF),
        ("context", get_token_count(request.messages, request.system, request.tools)),
    ]


def test_preflight_conversion_failure_skips_context_budget(lmstudio_provider):
    request = make_request()
    conversion_error = InvalidRequestError("invalid request conversion")

    with (
        patch.object(
            lmstudio_provider._chat,
            "_build_request_body",
            side_effect=conversion_error,
        ),
        patch.object(lmstudio_provider, "_preflight_context_budget") as context,
        pytest.raises(InvalidRequestError, match="invalid request conversion"),
    ):
        lmstudio_provider.preflight_messages(request, reasoning=REASONING_ON)

    context.assert_not_called()


@pytest.mark.asyncio
async def test_stream_messages_text(lmstudio_provider):
    """Text content deltas are emitted through the shared OpenAI-chat provider."""
    req = make_request()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello back!", reasoning_content=None, tool_calls=None
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        lmstudio_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in lmstudio_provider.stream_messages(req)]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_messages_passes_exact_reasoning_budget_via_extra_body(
    lmstudio_provider,
):
    req = make_request()
    policy = ReasoningPolicy.on(
        effort=ReasoningEffort.HIGH,
        budget_tokens=1024,
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Done", reasoning_content=None, tool_calls=None),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=1, prompt_tokens=1)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        lmstudio_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event
            async for event in lmstudio_provider.stream_messages(
                req,
                reasoning=policy,
            )
        ]

    await_args = mock_create.await_args
    assert await_args is not None
    create_kwargs = await_args.kwargs
    assert create_kwargs["extra_body"] == {"reasoning_tokens": 1024}
    assert "reasoning_tokens" not in create_kwargs
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_cleanup(lmstudio_provider):
    lmstudio_provider._client = AsyncMock()
    await lmstudio_provider.cleanup()


# --- Context-budget preflight (new: guards against LM Studio's silent
# mid-stream truncation when a prompt exceeds the loaded model's context) ---


def test_preflight_context_budget_noop_when_context_length_unknown(lmstudio_provider):
    """No LM Studio /api/v0/models data available -> preflight is a no-op (fail open)."""
    with patch.object(lmstudio_provider, "_loaded_context_length", return_value=None):
        lmstudio_provider._preflight_context_budget(1)  # must not raise


def test_preflight_context_budget_allows_request_under_budget(lmstudio_provider):
    with patch.object(
        lmstudio_provider, "_loaded_context_length", return_value=100_000
    ):
        lmstudio_provider._preflight_context_budget(1)  # must not raise


def test_preflight_context_budget_rejects_request_over_90_percent(lmstudio_provider):
    with (
        patch.object(lmstudio_provider, "_loaded_context_length", return_value=1000),
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        lmstudio_provider._preflight_context_budget(901)

    failure = exc_info.value
    assert failure.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    assert failure.status_code == 400
    assert failure.retryable is False
    assert failure.message == (
        "Estimated provider input (901 tokens) exceeds the safe LM Studio "
        "context budget (900 tokens; 90% of loaded context 1000)."
    )
    assert "prompt is too long" not in failure.message


def test_loaded_context_length_reads_max_across_loaded_models(lmstudio_provider):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "data": [
            {"state": "loaded", "loaded_context_length": 40960},
            {"state": "loaded", "loaded_context_length": 8192},
            {"state": "not-loaded", "loaded_context_length": 999999},
        ]
    }
    with patch(
        "free_claude_code.providers.lmstudio.client.httpx.get", return_value=response
    ) as mock_get:
        value = lmstudio_provider._loaded_context_length()

    assert value == 40960
    mock_get.assert_called_once()
    assert mock_get.call_args[0][0] == "http://localhost:1234/api/v0/models"


def test_loaded_context_length_fails_open_on_error(lmstudio_provider):
    with patch(
        "free_claude_code.providers.lmstudio.client.httpx.get",
        side_effect=httpx.ConnectError("refused"),
    ):
        assert lmstudio_provider._loaded_context_length() is None


def test_loaded_context_length_is_cached_within_ttl(lmstudio_provider):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "data": [{"state": "loaded", "loaded_context_length": 40960}]
    }
    with patch(
        "free_claude_code.providers.lmstudio.client.httpx.get", return_value=response
    ) as mock_get:
        first = lmstudio_provider._loaded_context_length()
        second = lmstudio_provider._loaded_context_length()

    assert first == second == 40960
    mock_get.assert_called_once()
