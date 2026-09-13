"""Groq request-local recovery from exact TPM-size rejections."""

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import openai
import pytest

from free_claude_code.config.provider_catalog import GROQ_DEFAULT_BASE
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.groq import GroqProvider
from free_claude_code.providers.groq.tpm import correct_tpm_completion_budget
from tests.providers.request_factory import make_messages_request
from tests.providers.support import immediate_admission, make_provider_config

_MODEL = "openai/gpt-oss-120b"
_LIMIT = 8_000
_REQUESTED = 26_206
_ORIGINAL_MAX = 24_576
_CORRECTED_MAX = 6_370


def _detail(
    message: str = (
        "Request too large for model `openai/gpt-oss-120b` on tokens per minute "
        "(TPM): Limit 8000, Requested 26206, please reduce your message size"
    ),
    *,
    error_type: str = "tokens",
    code: str = "rate_limit_exceeded",
) -> dict[str, str]:
    return {"message": message, "type": error_type, "code": code}


def _status_error(
    *,
    status: int = 413,
    detail: object | None = None,
    wrapped: bool = False,
) -> openai.APIStatusError:
    request = httpx2.Request("POST", f"{GROQ_DEFAULT_BASE}/chat/completions")
    response = httpx2.Response(status, request=request)
    body = _detail() if detail is None else detail
    if wrapped:
        body = {"error": body}
    return openai.APIStatusError(
        "Groq request rejected",
        response=response,
        body=body,
    )


def _body(max_completion_tokens: object = _ORIGINAL_MAX) -> dict:
    return {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": max_completion_tokens,
    }


def _provider(*, max_attempts: int = 5) -> GroqProvider:
    return GroqProvider(
        make_provider_config("test-groq-key", GROQ_DEFAULT_BASE),
        admission=immediate_admission(
            provider_name="GROQ",
            max_attempts=max_attempts,
        ),
    )


def _chunk(*, content: str | None = None, finish_reason: str | None = None):
    return MagicMock(
        choices=[
            MagicMock(
                delta=MagicMock(
                    content=content,
                    reasoning_content=None,
                    tool_calls=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


async def _successful_stream():
    yield _chunk(content="working")
    yield _chunk(finish_reason="stop")


@pytest.mark.parametrize("wrapped", [False, True], ids=["sdk-detail", "wire-wrapper"])
def test_exact_tpm_rejection_corrects_only_completion_budget(wrapped: bool) -> None:
    body = _body()
    original = deepcopy(body)

    correction = correct_tpm_completion_budget(
        _status_error(wrapped=wrapped),
        body,
    )

    assert correction is not None
    assert correction.limit == _LIMIT
    assert correction.requested == _REQUESTED
    assert correction.previous_max_completion_tokens == _ORIGINAL_MAX
    assert correction.corrected_max_completion_tokens == _CORRECTED_MAX
    assert correction.body == {**body, "max_completion_tokens": _CORRECTED_MAX}
    assert correction.body is not body
    assert body == original


@pytest.mark.parametrize(
    ("error", "body"),
    [
        (_status_error(status=429), _body()),
        (_status_error(detail=_detail(error_type="rate_limit")), _body()),
        (_status_error(detail=_detail(code="context_length_exceeded")), _body()),
        (_status_error(detail=_detail("Request too large")), _body()),
        (
            _status_error(
                detail=_detail(
                    "tokens per minute (TPM): Limit 8000, Requested 26206garbage"
                )
            ),
            _body(),
        ),
        (
            _status_error(
                detail=_detail(
                    "tokens per minute (TPM): Limit 8000, Requested 26206,000"
                )
            ),
            _body(),
        ),
        (
            _status_error(
                detail=_detail(
                    "tokens per minute (TPM): Limit 8000, Requested 26206, 000"
                )
            ),
            _body(),
        ),
        (
            _status_error(
                detail=_detail(
                    "tokens per minute (TPM): Limit 8000, Requested 26206; "
                    "tokens per minute (TPM): Limit 8000, Requested 26206"
                )
            ),
            _body(),
        ),
        (
            _status_error(
                detail=_detail("tokens per minute (TPM): Limit 8000, Requested 8000")
            ),
            _body(),
        ),
        (
            _status_error(
                detail=_detail("tokens per minute (TPM): Limit 8000, Requested 9000")
            ),
            _body(),
        ),
        (
            _status_error(
                detail=_detail("tokens per minute (TPM): Limit 8000, Requested 24576")
            ),
            _body(),
        ),
        (
            _status_error(
                detail=_detail(
                    "tokens per minute (TPM): Limit 80000000000000000000, "
                    "Requested 90000000000000000000"
                )
            ),
            _body(),
        ),
        (_status_error(), _body(None)),
        (_status_error(), _body(True)),
        (_status_error(), _body(10_000)),
        (
            _status_error(),
            {**_body(), "extra_body": {"max_completion_tokens": _ORIGINAL_MAX}},
        ),
        (
            _status_error(),
            {**_body(), "extra_body": {"max_tokens": _ORIGINAL_MAX}},
        ),
    ],
    ids=[
        "wrong-status",
        "wrong-type",
        "wrong-code",
        "missing-clause",
        "malformed-requested-boundary",
        "grouped-requested-no-space",
        "grouped-requested-with-space",
        "ambiguous-clause",
        "no-overage",
        "requested-under-output-cap",
        "requested-equals-output-cap",
        "oversized-number",
        "non-integer-max",
        "boolean-max",
        "prompt-alone-too-large",
        "extra-max-completion-tokens",
        "extra-max-tokens",
    ],
)
def test_non_authoritative_tpm_rejection_is_not_corrected(
    error: Exception,
    body: dict,
) -> None:
    assert correct_tpm_completion_budget(error, body) is None


@pytest.mark.asyncio
async def test_tpm_correction_emits_one_downstream_lifecycle() -> None:
    provider = _provider()
    request = make_messages_request(_MODEL, max_tokens=_ORIGINAL_MAX)
    create = AsyncMock(side_effect=[_status_error(), _successful_stream()])

    with patch.object(provider._client.chat.completions, "create", create):
        raw = "".join(
            [
                event
                async for event in provider.stream_messages(
                    request,
                    reasoning=ReasoningPolicy.off(),
                )
            ]
        )

    events = parse_sse_text(raw)
    assert create.await_count == 2
    assert create.await_args_list[0].kwargs["max_completion_tokens"] == _ORIGINAL_MAX
    assert create.await_args_list[1].kwargs["max_completion_tokens"] == _CORRECTED_MAX
    assert "working" in raw
    assert "event: error" not in raw
    assert sum(event.event == "message_start" for event in events) == 1
    assert sum(event.event == "message_stop" for event in events) == 1


@pytest.mark.asyncio
async def test_tpm_correction_is_one_shot_per_stream_creation() -> None:
    provider = _provider()
    create = AsyncMock(side_effect=[_status_error(), _status_error()])

    with (
        patch.object(provider._client.chat.completions, "create", create),
        pytest.raises(openai.APIStatusError),
    ):
        await provider._chat._create_stream(
            _body(),
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )

    assert create.await_count == 2


@pytest.mark.asyncio
async def test_tpm_correction_respects_physical_attempt_ceiling() -> None:
    provider = _provider(max_attempts=1)
    create = AsyncMock(side_effect=_status_error())

    with (
        patch.object(provider._client.chat.completions, "create", create),
        pytest.raises(openai.APIStatusError),
    ):
        await provider._chat._create_stream(
            _body(),
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )

    assert create.await_count == 1


def _reasoning_error() -> openai.APIStatusError:
    message = "`reasoning_effort` value `high` must be one of `none` or `default`"
    return _status_error(
        status=400,
        detail={"message": message, "type": "invalid_request_error"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("errors", [("tpm", "reasoning"), ("reasoning", "tpm")])
async def test_tpm_and_reasoning_corrections_compose(
    errors: tuple[str, str],
) -> None:
    provider = _provider()
    request = make_messages_request(_MODEL, max_tokens=_ORIGINAL_MAX)
    body = provider._chat._build_request_body(
        request,
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )
    failures = {
        "tpm": _status_error(),
        "reasoning": _reasoning_error(),
    }
    create = AsyncMock(side_effect=[*(failures[name] for name in errors), object()])
    execution = provider._admission.start_execution()

    with patch.object(provider._client.chat.completions, "create", create):
        (
            _stream,
            accepted_body,
            attempt,
            _sent_body,
        ) = await provider._chat._create_stream(
            body,
            execution,
            ProviderOperationKind.GENERATION,
        )
        await attempt.aclose()

    assert create.await_count == 3
    assert execution.attempts_started == 3
    assert accepted_body["max_completion_tokens"] == _CORRECTED_MAX
    assert accepted_body["reasoning_effort"] == "default"


@pytest.mark.asyncio
async def test_distinct_bodies_get_independent_tpm_corrections() -> None:
    provider = _provider()
    execution = provider._admission.start_execution()
    second_error = _status_error(
        detail=_detail("tokens per minute (TPM): Limit 8000, Requested 22000")
    )
    create = AsyncMock(side_effect=[_status_error(), object(), second_error, object()])

    with patch.object(provider._client.chat.completions, "create", create):
        (
            _stream,
            first_body,
            first_attempt,
            _sent_body,
        ) = await provider._chat._create_stream(
            _body(),
            execution,
            ProviderOperationKind.CONTINUATION,
        )
        await first_attempt.accept()
        await first_attempt.aclose()
        (
            _stream,
            second_body,
            second_attempt,
            _sent_body,
        ) = await provider._chat._create_stream(
            _body(20_000),
            execution,
            ProviderOperationKind.TOOL_REPAIR,
        )
        await second_attempt.aclose()

    assert execution.attempts_started == 4
    assert first_body["max_completion_tokens"] == _CORRECTED_MAX
    assert second_body["max_completion_tokens"] == 6_000
