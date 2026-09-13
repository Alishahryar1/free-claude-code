"""Groq model-agnostic reasoning-vocabulary negotiation tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import openai
import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import GROQ_DEFAULT_BASE
from free_claude_code.core.anthropic.stream_contracts import (
    parse_sse_text,
    text_content,
)
from free_claude_code.core.reasoning import (
    ReasoningCapability,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.groq import GroqProvider
from free_claude_code.providers.groq.client import (
    GroqChatBehavior,
    _parse_reasoning_vocabulary,
    _rewrite_reasoning_effort,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    capture_openai_chat_wire_body,
    immediate_admission,
    make_provider_config,
)

_MODEL = "opaque-model-a"
_ISSUE_MESSAGE = "`reasoning_effort` must be one of `none` or `default`"


class _BadRequest(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = 400,
        body: object | None = None,
        response: object | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = response


def _reasoning_vocabularies(provider: GroqProvider) -> dict[str, frozenset[str]]:
    assert isinstance(provider._behavior, GroqChatBehavior)
    return provider._behavior._model_reasoning_vocabularies


def _provider(*, max_attempts: int = 5) -> GroqProvider:
    return GroqProvider(
        make_provider_config(
            api_key="test_groq_key",
            base_url=GROQ_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(
            provider_name="GROQ",
            max_attempts=max_attempts,
        ),
    )


def _request(model: str = _MODEL):
    return make_messages_request(model)


def _vocabulary_error(
    values: str = "`none` or `default`",
    *,
    current: str = "high",
) -> _BadRequest:
    message = f"`reasoning_effort` value `{current}` must be one of {values}"
    return _BadRequest(
        message,
        body={"message": message, "type": "invalid_request_error"},
    )


def _chunk(*, content: str | None = None, finish_reason: str | None = None):
    delta = MagicMock(
        content=content,
        reasoning_content=None,
        tool_calls=None,
    )
    return MagicMock(
        choices=[MagicMock(delta=delta, finish_reason=finish_reason)],
        usage=None,
    )


async def _successful_stream(text: str = "visible"):
    yield _chunk(content=text)
    yield _chunk(finish_reason="stop")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "`reasoning_effort` must be one of `default`",
        "reasoning_effort value `none` must be one of `default`",
        "accepted options for reasoning_effort include default",
    ],
)
async def test_classifier_correction_does_not_change_later_requests(message):
    provider = _provider()
    sent = []

    async def create(**body):
        sent.append(body)
        if len(sent) == 1:
            raise _BadRequest(message, body={"message": message})
        return _successful_stream("<severity>0</severity>")

    try:
        with patch.object(provider._client.chat.completions, "create", create):
            classifier = "".join(
                [
                    event
                    async for event in provider.stream_messages(
                        make_messages_request(_MODEL, max_tokens=64),
                        reasoning=ReasoningPolicy.prefer_off(),
                        model_info=ProviderModelInfo(
                            _MODEL, reasoning_capability=ReasoningCapability.OPTIONAL
                        ),
                    )
                ]
            )
            ordinary = "".join(
                [
                    event
                    async for event in provider.stream_messages(
                        _request(),
                        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
                    )
                ]
            )
    finally:
        await provider.cleanup()

    assert text_content(parse_sse_text(classifier)) == "<severity>0</severity>"
    assert text_content(parse_sse_text(ordinary)) == "<severity>0</severity>"
    assert len(sent) == 3
    assert sent[2]["reasoning_effort"] == "high"
    assert sent[0]["reasoning_effort"] == "none"
    assert sent[0]["max_completion_tokens"] == 64
    assert "reasoning_effort" not in sent[1]
    assert "max_completion_tokens" not in sent[1]


def test_parse_exact_issue_vocabulary_excludes_rejected_value() -> None:
    error = _BadRequest(
        _ISSUE_MESSAGE,
        body={"message": _ISSUE_MESSAGE, "type": "invalid_request_error"},
    )

    assert _parse_reasoning_vocabulary(error) == frozenset({"none", "default"})


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": _ISSUE_MESSAGE}},
        {"detail": _ISSUE_MESSAGE},
        {
            "error": {
                "param": "reasoning_effort",
                "message": "must be one of 'none', 'default'",
            }
        },
        [
            {
                "parameter": "reasoning_effort",
                "detail": "allowed values are NONE or DEFAULT",
            }
        ],
    ],
)
def test_parse_structured_error_shapes(body: object) -> None:
    assert _parse_reasoning_vocabulary(_BadRequest("invalid", body=body)) == (
        frozenset({"none", "default"})
    )


def test_parse_does_not_inherit_reasoning_param_into_unrelated_descendant() -> None:
    error = _BadRequest(
        "invalid request",
        body={
            "param": "reasoning_effort",
            "details": [{"message": "temperature must be one of low, medium, or high"}],
        },
    )

    assert _parse_reasoning_vocabulary(error) is None


def test_parse_actual_openai_bad_request() -> None:
    request = httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx2.Response(400, request=request)
    error = openai.BadRequestError(
        _ISSUE_MESSAGE,
        response=response,
        body={"message": _ISSUE_MESSAGE},
    )

    assert _parse_reasoning_vocabulary(error) == frozenset({"none", "default"})


def test_parse_response_text_fallback() -> None:
    response = SimpleNamespace(status_code=400, text=_ISSUE_MESSAGE)
    error = _BadRequest(
        "invalid request",
        status_code=None,
        response=response,
    )

    assert _parse_reasoning_vocabulary(error) == frozenset({"none", "default"})


def test_parse_json_response_text_preserves_structured_field_scope() -> None:
    response = SimpleNamespace(
        status_code=400,
        text=(
            '{"param":"reasoning_effort","details":['
            '{"message":"temperature must be one of low, medium, or high"}]}'
        ),
    )
    error = _BadRequest("invalid request", status_code=None, response=response)

    assert _parse_reasoning_vocabulary(error) is None


def test_parse_json_response_text_accepts_direct_reasoning_message() -> None:
    response = SimpleNamespace(
        status_code=400,
        text=(
            '{"error":{"param":"reasoning_effort",'
            '"message":"must be one of none or default"}}'
        ),
    )
    error = _BadRequest("invalid request", status_code=None, response=response)

    assert _parse_reasoning_vocabulary(error) == frozenset({"none", "default"})


def test_parse_exception_string_fallback() -> None:
    assert _parse_reasoning_vocabulary(_BadRequest(_ISSUE_MESSAGE)) == frozenset(
        {"none", "default"}
    )


@pytest.mark.parametrize(
    "message",
    [
        "reasoning_effort should be one of low, medium, or high",
        "reasoning_effort is expected to be one of low / medium / high",
        "allowed values for reasoning_effort are low, medium and high",
        "reasoning_effort valid values: `low`, `medium`, `high`",
        "accepted options for reasoning_effort include low, medium, high",
        "reasoning_effort valid option is low",
    ],
)
def test_parse_supported_allow_list_phrasings(message: str) -> None:
    assert _parse_reasoning_vocabulary(_BadRequest(message)) == frozenset(
        {"low", "medium", "high"} if "medium" in message else {"low"}
    )


def test_parse_unknown_only_values_returns_understood_empty_vocabulary() -> None:
    error = _BadRequest("reasoning_effort must be one of `balanced` or `extreme`")

    assert _parse_reasoning_vocabulary(error) == frozenset()


@pytest.mark.parametrize(
    "message",
    [
        "Unsupported parameter: reasoning_effort",
        "reasoning_effort field is not supported",
        "Unknown field `reasoning_effort`",
        "Unrecognized parameter reasoning_effort",
        "This model does not support reasoning_effort",
        "`reasoning_effort` is not supported with this model",
    ],
)
def test_parse_explicit_unsupported_phrasings(message: str) -> None:
    assert _parse_reasoning_vocabulary(_BadRequest(message)) == frozenset()


def test_parse_prefers_structured_allow_list_over_fallback_text() -> None:
    error = _BadRequest(
        "Unsupported parameter: reasoning_effort",
        body={"message": _ISSUE_MESSAGE},
    )

    assert _parse_reasoning_vocabulary(error) == frozenset({"none", "default"})


def test_parse_finds_reasoning_allow_list_after_unrelated_clause() -> None:
    error = _BadRequest(
        "temperature must be one of 0 or 1; "
        "reasoning_effort must be one of none or default"
    )

    assert _parse_reasoning_vocabulary(error) == frozenset({"none", "default"})


@pytest.mark.parametrize(
    "error",
    [
        _BadRequest(_ISSUE_MESSAGE, status_code=500),
        _BadRequest("temperature must be one of low or high"),
        _BadRequest("reasoning_effort is invalid"),
        _BadRequest("temperature must be one of low or high; reasoning_effort is set"),
        _BadRequest(
            "must be one of low or high",
            body={"param": "temperature", "message": "must be one of low or high"},
        ),
        _BadRequest("Unsupported parameter: temperature; reasoning_effort is set"),
    ],
)
def test_parse_rejects_unrelated_or_ambiguous_errors(error: Exception) -> None:
    assert _parse_reasoning_vocabulary(error) is None


def test_parse_ignores_non_string_response_text() -> None:
    error = _BadRequest(
        "invalid",
        status_code=None,
        response=SimpleNamespace(status_code=400, text=object()),
    )

    assert _parse_reasoning_vocabulary(error) is None


@pytest.mark.parametrize(
    "current,accepted,expected",
    [
        (None, frozenset({"none", "default"}), None),
        ("none", frozenset({"none", "default"}), None),
        ("default", frozenset({"none", "default"}), None),
        ("low", frozenset({"none", "default"}), "default"),
        ("medium", frozenset({"none", "default"}), "default"),
        ("high", frozenset({"none", "default"}), "default"),
        ("none", frozenset({"low", "medium", "high"}), None),
        ("low", frozenset({"low", "medium", "high"}), "low"),
        ("medium", frozenset({"low", "medium", "high"}), "medium"),
        ("high", frozenset({"low", "medium", "high"}), "high"),
        ("default", frozenset({"low", "medium", "high"}), None),
        ("high", frozenset({"low"}), None),
        ("high", frozenset(), None),
    ],
)
def test_rewrite_reasoning_effort_total_mapping(
    current: str | None,
    accepted: frozenset[str],
    expected: str | None,
) -> None:
    body = {"model": _MODEL, "messages": [{"role": "user", "content": "x"}]}
    if current is not None:
        body["reasoning_effort"] = current
    original = {**body, "messages": list(body["messages"])}

    rewritten = _rewrite_reasoning_effort(body, accepted)

    if current is None or current in accepted:
        assert rewritten is None
    else:
        assert rewritten is not None
        assert rewritten.get("reasoning_effort") == expected
        assert rewritten["messages"] is body["messages"]
    assert body == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy,expected",
    [
        (ReasoningPolicy.provider_default(), None),
        (ReasoningPolicy.off(), "none"),
        (ReasoningPolicy.on(), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM), "medium"),
        (ReasoningPolicy.on(effort=ReasoningEffort.XHIGH), "high"),
    ],
)
async def test_initial_reasoning_policy_reaches_sdk_wire(
    policy: ReasoningPolicy,
    expected: str | None,
) -> None:
    body = _provider()._chat._build_request_body(_request(), reasoning=policy)

    wire = await capture_openai_chat_wire_body(body)

    assert wire.get("reasoning_effort") == expected


@pytest.mark.asyncio
async def test_exact_issue_retries_with_default_and_learns_model() -> None:
    provider = _provider()
    policy = ReasoningPolicy.on(effort=ReasoningEffort.HIGH)
    body = provider._chat._build_request_body(_request(), reasoning=policy)
    create = AsyncMock(side_effect=[_vocabulary_error(), object()])

    with patch.object(provider._client.chat.completions, "create", create):
        _stream, used_body, attempt, _sent_body = await provider._chat._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )
        await attempt.aclose()

    assert create.await_count == 2
    assert create.await_args_list[0].kwargs["reasoning_effort"] == "high"
    assert create.await_args_list[1].kwargs["reasoning_effort"] == "default"
    assert used_body["reasoning_effort"] == "default"
    assert _reasoning_vocabularies(provider) == {_MODEL: frozenset({"none", "default"})}

    next_body = provider._chat._build_request_body(_request(), reasoning=policy)
    assert next_body["reasoning_effort"] == "default"
    next_create = AsyncMock(return_value=object())
    with patch.object(provider._client.chat.completions, "create", next_create):
        (
            _stream,
            next_used_body,
            next_attempt,
            _sent_body,
        ) = await provider._chat._create_stream(
            next_body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )
        await next_attempt.aclose()
    assert next_create.await_count == 1
    assert next_used_body["reasoning_effort"] == "default"


def test_cached_none_default_preserves_off() -> None:
    provider = _provider()
    _reasoning_vocabularies(provider)[_MODEL] = frozenset({"none", "default"})

    body = provider._chat._build_request_body(
        _request(), reasoning=ReasoningPolicy.off()
    )

    assert body["reasoning_effort"] == "none"


@pytest.mark.parametrize(
    "effort,expected",
    [
        (ReasoningEffort.LOW, "low"),
        (ReasoningEffort.MEDIUM, "medium"),
        (ReasoningEffort.HIGH, "high"),
    ],
)
def test_cached_named_vocabulary_preserves_enabled_efforts(
    effort: ReasoningEffort,
    expected: str,
) -> None:
    provider = _provider()
    _reasoning_vocabularies(provider)[_MODEL] = frozenset({"low", "medium", "high"})

    body = provider._chat._build_request_body(
        _request(),
        reasoning=ReasoningPolicy.on(effort=effort),
    )

    assert body["reasoning_effort"] == expected


def test_cached_named_vocabulary_omits_unrepresentable_off() -> None:
    provider = _provider()
    _reasoning_vocabularies(provider)[_MODEL] = frozenset({"low", "medium", "high"})

    body = provider._chat._build_request_body(
        _request(), reasoning=ReasoningPolicy.off()
    )

    assert "reasoning_effort" not in body


@pytest.mark.asyncio
async def test_unknown_vocabulary_retries_without_effort_and_negative_caches() -> None:
    provider = _provider()
    body = provider._chat._build_request_body(
        _request(),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )
    create = AsyncMock(
        side_effect=[
            _vocabulary_error("`balanced` or `extreme`"),
            object(),
        ]
    )

    with patch.object(provider._client.chat.completions, "create", create):
        _stream, used_body, attempt, _sent_body = await provider._chat._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )
        await attempt.aclose()

    assert "reasoning_effort" not in create.await_args_list[1].kwargs
    assert "reasoning_effort" not in used_body
    assert _reasoning_vocabularies(provider)[_MODEL] == frozenset()
    assert "reasoning_effort" not in provider._chat._build_request_body(
        _request(),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )


def test_reasoning_cache_isolated_by_opaque_model_id() -> None:
    provider = _provider()
    policy = ReasoningPolicy.on(effort=ReasoningEffort.HIGH)
    _reasoning_vocabularies(provider)[_MODEL] = frozenset({"none", "default"})

    learned = provider._chat._build_request_body(_request(_MODEL), reasoning=policy)
    unlearned = provider._chat._build_request_body(
        _request("opaque-model-b"), reasoning=policy
    )

    assert learned["reasoning_effort"] == "default"
    assert unlearned["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_concurrent_first_requests_can_learn_without_state_corruption() -> None:
    provider = _provider()
    policy = ReasoningPolicy.on(effort=ReasoningEffort.HIGH)
    bodies = [
        provider._chat._build_request_body(_request(), reasoning=policy),
        provider._chat._build_request_body(_request(), reasoning=policy),
    ]
    initial_calls = 0
    release_initial_errors = asyncio.Event()
    sent_efforts: list[str | None] = []

    async def create(**kwargs: object):
        nonlocal initial_calls
        effort = kwargs.get("reasoning_effort")
        sent_efforts.append(effort if isinstance(effort, str) else None)
        if effort == "high":
            initial_calls += 1
            if initial_calls == 2:
                release_initial_errors.set()
            await release_initial_errors.wait()
            raise _vocabulary_error()
        return object()

    async def execute(body: dict):
        _stream, used_body, attempt, _sent_body = await provider._chat._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )
        await attempt.aclose()
        return used_body

    mock_create = AsyncMock(side_effect=create)
    with patch.object(provider._client.chat.completions, "create", mock_create):
        used_bodies = await asyncio.gather(*(execute(body) for body in bodies))

    assert mock_create.await_count == 4
    assert sent_efforts.count("high") == 2
    assert sent_efforts.count("default") == 2
    assert all(body["reasoning_effort"] == "default" for body in used_bodies)
    assert _reasoning_vocabularies(provider) == {_MODEL: frozenset({"none", "default"})}


@pytest.mark.asyncio
async def test_stale_cache_self_heals_without_guessing_original_effort() -> None:
    provider = _provider()
    policy = ReasoningPolicy.on(effort=ReasoningEffort.HIGH)
    _reasoning_vocabularies(provider)[_MODEL] = frozenset({"none", "default"})
    cached_body = provider._chat._build_request_body(_request(), reasoning=policy)
    assert cached_body["reasoning_effort"] == "default"
    create = AsyncMock(
        side_effect=[
            _vocabulary_error("`low`, `medium`, or `high`", current="default"),
            object(),
        ]
    )

    with patch.object(provider._client.chat.completions, "create", create):
        (
            _stream,
            corrected_body,
            attempt,
            _sent_body,
        ) = await provider._chat._create_stream(
            cached_body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )
        await attempt.aclose()

    assert "reasoning_effort" not in corrected_body
    assert _reasoning_vocabularies(provider)[_MODEL] == frozenset(
        {"low", "medium", "high"}
    )
    rebuilt = provider._chat._build_request_body(_request(), reasoning=policy)
    assert rebuilt["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_unrelated_400_propagates_without_cache_poisoning() -> None:
    provider = _provider()
    body = provider._chat._build_request_body(
        _request(),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )
    create = AsyncMock(side_effect=_BadRequest("messages: invalid role 'wizard'"))

    with (
        patch.object(provider._client.chat.completions, "create", create),
        pytest.raises(_BadRequest, match="wizard"),
    ):
        await provider._chat._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )

    assert create.await_count == 1
    assert _reasoning_vocabularies(provider) == {}


@pytest.mark.asyncio
async def test_nested_unrelated_allow_list_cannot_retry_or_poison_cache() -> None:
    provider = _provider()
    body = provider._chat._build_request_body(
        _request(),
        reasoning=ReasoningPolicy.off(),
    )
    error = _BadRequest(
        "invalid request",
        body={
            "param": "reasoning_effort",
            "details": [{"message": "temperature must be one of low, medium, or high"}],
        },
    )
    create = AsyncMock(side_effect=error)

    with (
        patch.object(provider._client.chat.completions, "create", create),
        pytest.raises(_BadRequest),
    ):
        await provider._chat._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )

    assert create.await_count == 1
    assert create.await_args_list[0].kwargs["reasoning_effort"] == "none"
    assert _reasoning_vocabularies(provider) == {}


@pytest.mark.asyncio
async def test_advertised_current_value_does_not_retry_or_cache() -> None:
    provider = _provider()
    body = provider._chat._build_request_body(
        _request(),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )
    create = AsyncMock(side_effect=_vocabulary_error("`low`, `medium`, or `high`"))

    with (
        patch.object(provider._client.chat.completions, "create", create),
        pytest.raises(_BadRequest),
    ):
        await provider._chat._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )

    assert create.await_count == 1
    assert _reasoning_vocabularies(provider) == {}


@pytest.mark.asyncio
async def test_last_attempt_learns_but_does_not_exceed_budget() -> None:
    provider = _provider(max_attempts=1)
    policy = ReasoningPolicy.on(effort=ReasoningEffort.HIGH)
    body = provider._chat._build_request_body(_request(), reasoning=policy)
    create = AsyncMock(side_effect=_vocabulary_error())

    with (
        patch.object(provider._client.chat.completions, "create", create),
        pytest.raises(_BadRequest),
    ):
        await provider._chat._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )

    assert create.await_count == 1
    assert _reasoning_vocabularies(provider)[_MODEL] == frozenset({"none", "default"})
    assert (
        provider._chat._build_request_body(_request(), reasoning=policy)[
            "reasoning_effort"
        ]
        == "default"
    )


@pytest.mark.asyncio
async def test_output_cap_and_reasoning_corrections_share_one_session() -> None:
    provider = _provider()
    body = provider._chat._build_request_body(
        make_messages_request(_MODEL, max_tokens=64_000),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )
    cap_error = _BadRequest("max_completion_tokens must be less than or equal to 40960")
    create = AsyncMock(side_effect=[cap_error, _vocabulary_error(), object()])
    execution = provider._admission.start_execution()

    with patch.object(provider._client.chat.completions, "create", create):
        _stream, used_body, attempt, _sent_body = await provider._chat._create_stream(
            body,
            execution,
            ProviderOperationKind.GENERATION,
        )
        await attempt.aclose()

    assert create.await_count == 3
    assert execution.attempts_started == 3
    assert create.await_args_list[1].kwargs["max_completion_tokens"] == 40_960
    assert create.await_args_list[1].kwargs["reasoning_effort"] == "high"
    assert create.await_args_list[2].kwargs["max_completion_tokens"] == 40_960
    assert create.await_args_list[2].kwargs["reasoning_effort"] == "default"
    assert used_body["max_completion_tokens"] == 40_960
    assert used_body["reasoning_effort"] == "default"


@pytest.mark.asyncio
async def test_corrected_request_cannot_enter_vocabulary_retry_loop() -> None:
    provider = _provider()
    body = provider._chat._build_request_body(
        _request(),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )
    create = AsyncMock(side_effect=[_vocabulary_error(), _vocabulary_error()])

    with (
        patch.object(provider._client.chat.completions, "create", create),
        pytest.raises(_BadRequest),
    ):
        await provider._chat._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )

    assert create.await_count == 2


@pytest.mark.asyncio
async def test_reasoning_correction_emits_one_downstream_lifecycle() -> None:
    provider = _provider()
    request = _request()
    policy = ReasoningPolicy.on(effort=ReasoningEffort.HIGH)
    create = AsyncMock(side_effect=[_vocabulary_error(), _successful_stream()])

    with patch.object(provider._client.chat.completions, "create", create):
        raw = "".join(
            [
                event
                async for event in provider.stream_messages(
                    request,
                    reasoning=policy,
                )
            ]
        )

    events = parse_sse_text(raw)
    assert create.await_count == 2
    assert create.await_args_list[0].kwargs["reasoning_effort"] == "high"
    assert create.await_args_list[1].kwargs["reasoning_effort"] == "default"
    assert "visible" in raw
    assert "event: error" not in raw
    assert sum(event.event == "message_start" for event in events) == 1
    assert sum(event.event == "content_block_start" for event in events) == 1
    assert sum(event.event == "content_block_stop" for event in events) == 1
    assert sum(event.event == "message_delta" for event in events) == 1
    assert sum(event.event == "message_stop" for event in events) == 1
