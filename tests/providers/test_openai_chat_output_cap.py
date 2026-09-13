"""Tests for OpenAI-compatible output-token cap recovery (issue #955).

Covers the pure parse/clamp helpers and the provider behavior that clamps
``max_completion_tokens``/``max_tokens`` to the upstream maximum, retries once,
and learns the cap so later requests clamp proactively.
"""

from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.config.provider_catalog import GROQ_DEFAULT_BASE
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.groq import GroqProvider
from free_claude_code.providers.openai_chat.output_cap import (
    clamp_output_tokens,
    parse_output_token_cap,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
)


class _BadRequest(Exception):
    """Stand-in for openai.BadRequestError (status_code + optional JSON body)."""

    def __init__(self, message: str, body: object | None = None):
        super().__init__(message)
        self.status_code = 400
        self.body = body


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_parse_cap_from_groq_message():
    error = _BadRequest(
        "`max_completion_tokens` must be less than or equal to `16384`, the maximum "
        "value for `max_completion_tokens` is less than the `context_window` for this "
        "model"
    )
    assert parse_output_token_cap(error) == 16384


@pytest.mark.parametrize(
    "message,expected",
    [
        ("max_tokens: maximum value is 8192", 8192),
        ("max_tokens must not exceed 16000", 16000),
        ("`max_completion_tokens` <= 4096 required", 4096),
        ("max_tokens at most 2048 allowed", 2048),
        ("maximum allowed value of 32768 for max_tokens", 32768),
    ],
)
def test_parse_cap_various_phrasings(message, expected):
    assert parse_output_token_cap(_BadRequest(message)) == expected


def test_parse_cap_reads_json_body():
    error = _BadRequest(
        "invalid request",
        body={"error": {"param": "max_completion_tokens", "message": "<= 12000"}},
    )
    assert parse_output_token_cap(error) == 12000


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Range of max_tokens should be [1, 32768]", 32768),
        ("range of max_completion_tokens should be [ 1 , `8192` ]", 8192),
    ],
)
def test_parse_cap_from_inclusive_range(message, expected):
    assert parse_output_token_cap(_BadRequest(message)) == expected


@pytest.mark.parametrize(
    "message",
    [
        "Range of temperature should be [0, 2] for max_tokens requests",
        "Range of max_tokens should be [0, 4096]",
        "Range of max_tokens should be [1, unlimited]",
        "Range of max_tokens is [1, 4096]",
    ],
)
def test_parse_cap_ignores_unrecognized_ranges(message):
    assert parse_output_token_cap(_BadRequest(message)) is None


def test_parse_cap_ignores_non_400():
    error = _BadRequest("max_tokens must be less than or equal to 40960")
    error.status_code = 500
    assert parse_output_token_cap(error) is None


def test_parse_cap_ignores_unrelated_400():
    assert parse_output_token_cap(_BadRequest("temperature must be <= 2")) is None


@pytest.mark.parametrize(
    "message",
    [
        "temperature must be <= 2; max_tokens is invalid",
        "max_completion_tokens is invalid; temperature at most 2",
        "maximum allowed value of 2 for temperature; max_tokens is invalid",
    ],
)
def test_parse_cap_does_not_bind_unrelated_comparator_to_output_field(message):
    assert parse_output_token_cap(_BadRequest(message)) is None


def test_parse_cap_uses_only_the_structured_output_parameter_message():
    error = _BadRequest(
        "max_tokens is invalid",
        body={
            "errors": [
                {"param": "temperature", "message": "must not exceed 2"},
                {"param": "top_p", "message": "<= 1"},
            ]
        },
    )

    assert parse_output_token_cap(error) is None


def test_parse_cap_respects_structured_non_output_parameter():
    body = {
        "param": "temperature",
        "message": "max_tokens must not exceed 2",
    }
    error = _BadRequest(
        f"Error code: 400 - {body}",
        body=body,
    )

    assert parse_output_token_cap(error) is None


def test_parse_cap_does_not_escape_structured_parameter_scope():
    error = _BadRequest(
        "invalid request",
        body={
            "param": "temperature",
            "details": {"message": "max_tokens must not exceed 2"},
        },
    )

    assert parse_output_token_cap(error) is None


def test_parse_cap_ignores_text_outside_structured_error_schema():
    error = _BadRequest(
        "invalid request",
        body={"request": {"message": "max_tokens must not exceed 2"}},
    )

    assert parse_output_token_cap(error) is None


def test_parse_cap_reads_unscoped_structured_message():
    error = _BadRequest(
        "invalid request",
        body={"message": "max_tokens must not exceed 8192"},
    )

    assert parse_output_token_cap(error) == 8192


def test_parse_cap_selects_matching_parameter_from_structured_error_list():
    error = _BadRequest(
        "invalid request",
        body={
            "errors": [
                {
                    "param": "temperature",
                    "message": "max_tokens must not exceed 2",
                },
                {"param": "max_tokens", "message": "<= 8192"},
            ]
        },
    )

    assert parse_output_token_cap(error) == 8192


def test_parse_cap_returns_none_without_number():
    assert (
        parse_output_token_cap(_BadRequest("max_tokens is larger than allowed")) is None
    )


def test_clamp_reduces_max_completion_tokens():
    assert clamp_output_tokens({"max_completion_tokens": 64000}, 40960) == {
        "max_completion_tokens": 40960
    }


def test_clamp_reduces_max_tokens():
    assert clamp_output_tokens({"max_tokens": 100000}, 8192) == {"max_tokens": 8192}


def test_clamp_noop_when_within_cap_returns_none():
    assert clamp_output_tokens({"max_completion_tokens": 1000}, 40960) is None


def test_clamp_does_not_mutate_input():
    body = {"max_tokens": 99999, "model": "m"}
    clamped = clamp_output_tokens(body, 8192)
    assert body["max_tokens"] == 99999
    assert clamped is not None
    assert clamped["max_tokens"] == 8192


def test_clamp_ignores_bool_values():
    assert clamp_output_tokens({"max_tokens": True}, 8192) is None


# --------------------------------------------------------------------------- #
# Provider integration (via Groq's profile, which uses max_completion_tokens)
# --------------------------------------------------------------------------- #


@pytest.fixture
def groq_provider():
    return GroqProvider(
        make_provider_config(
            api_key="test_groq_key",
            base_url=GROQ_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )


@pytest.mark.asyncio
async def test_create_stream_clamps_and_learns_on_cap_rejection(groq_provider):
    body = groq_provider._chat._build_request_body(
        make_messages_request(
            "llama-3.3-70b-versatile",
            max_tokens=64000,
            thinking={"enabled": False},
        )
    )
    assert body["max_completion_tokens"] == 64000
    model = body["model"]

    error = _BadRequest(
        "invalid request",
        body={
            "message": (
                "`max_completion_tokens` must be less than or equal to `16384`, "
                "the maximum value for `max_completion_tokens` is less than the "
                "`context_window` for this model"
            ),
            "type": "invalid_request_error",
            "param": "max_completion_tokens",
        },
    )
    create = AsyncMock(side_effect=[error, object()])

    with patch.object(groq_provider._client.chat.completions, "create", create):
        (
            _stream,
            used_body,
            attempt,
            _sent_body,
        ) = await groq_provider._chat._create_stream(
            body,
            groq_provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )
        await attempt.aclose()

    assert create.call_count == 2
    assert create.call_args_list[1].kwargs["max_completion_tokens"] == 16384
    assert used_body["max_completion_tokens"] == 16384
    assert groq_provider._chat._model_output_caps[model] == 16384


@pytest.mark.asyncio
async def test_learned_cap_clamps_next_request_without_a_400(groq_provider):
    body = groq_provider._chat._build_request_body(
        make_messages_request(
            "llama-3.3-70b-versatile",
            max_tokens=64000,
            thinking={"enabled": False},
        )
    )
    model = body["model"]
    groq_provider._chat._model_output_caps[model] = 40960

    create = AsyncMock(return_value=object())
    with patch.object(groq_provider._client.chat.completions, "create", create):
        (
            _stream,
            used_body,
            attempt,
            _sent_body,
        ) = await groq_provider._chat._create_stream(
            body,
            groq_provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )
        await attempt.aclose()

    assert create.call_count == 1
    assert create.call_args.kwargs["max_completion_tokens"] == 40960
    assert used_body["max_completion_tokens"] == 40960


@pytest.mark.asyncio
async def test_unrelated_400_is_not_clamped_and_propagates(groq_provider):
    body = groq_provider._chat._build_request_body(
        make_messages_request(
            "llama-3.3-70b-versatile",
            max_tokens=100,
            thinking={"enabled": False},
        )
    )
    create = AsyncMock(side_effect=_BadRequest("messages: invalid role 'wizard'"))

    with (
        patch.object(groq_provider._client.chat.completions, "create", create),
        pytest.raises(Exception, match="wizard"),
    ):
        await groq_provider._chat._create_stream(
            body,
            groq_provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )

    assert create.call_count == 1
    assert groq_provider._chat._model_output_caps == {}


@pytest.mark.asyncio
async def test_mixed_field_400_does_not_retry_or_poison_learned_cap(groq_provider):
    body = groq_provider._chat._build_request_body(
        make_messages_request(
            "llama-3.3-70b-versatile",
            max_tokens=64000,
            thinking={"enabled": False},
        )
    )
    error_body = {
        "param": "temperature",
        "message": "max_completion_tokens must not exceed 2",
    }
    create = AsyncMock(
        side_effect=_BadRequest(
            f"Error code: 400 - {error_body}",
            body=error_body,
        )
    )

    with (
        patch.object(groq_provider._client.chat.completions, "create", create),
        pytest.raises(Exception, match="temperature"),
    ):
        await groq_provider._chat._create_stream(
            body,
            groq_provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )

    assert create.call_count == 1
    assert groq_provider._chat._model_output_caps == {}
