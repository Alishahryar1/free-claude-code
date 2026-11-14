"""A request corrector turns one rejection into one corrected body, in a fixed order.

Order: history record repair, reasoning control, transport steps, and the
history-mode fallback after an HTTP 400.
"""

import json
from typing import Any

import httpx
import pytest

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.history_negotiation import HistoryModeCache
from free_claude_code.providers.reasoning_compatibility import ReasoningCorrection
from free_claude_code.providers.request_correction import (
    CorrectionStep,
    Rejection,
    RequestCorrector,
)

ORIGIN = ReplayOrigin(
    "provider", "chat", "https://provider.test/v1", "private-credential", "model"
)


def _http_400(error: JsonObject) -> httpx.HTTPStatusError:
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://provider.test"),
        json={"error": error},
    )
    return httpx.HTTPStatusError(
        "rejected", request=response.request, response=response
    )


def _step(kind: str | None, calls: list[str], result: JsonObject | None):
    def propose(rejection: Rejection) -> JsonObject | None:
        calls.append(kind or "kindless")
        return result

    return CorrectionStep(kind, propose)


def _readable_source() -> dict[str, Any]:
    return {
        "model": "model",
        "messages": [
            {"role": "assistant", "content": "Answer", "reasoning_content": "Old"},
            {"role": "user", "content": "Next"},
            {"role": "assistant", "content": "Answer", "reasoning_content": "New"},
        ],
    }


def _rejection(error: Exception, body: JsonObject | None = None) -> Rejection:
    body = body if body is not None else {"model": "model", "messages": []}
    return Rejection(error, body, body)


def test_steps_run_in_order_and_the_first_correction_wins():
    calls: list[str] = []
    corrector = RequestCorrector(
        "chat",
        steps=(
            _step("first", calls, None),
            _step("second", calls, {"corrected": "second"}),
            _step("third", calls, {"corrected": "third"}),
        ),
    )

    assert corrector.correct(_rejection(RuntimeError("no"))) == {"corrected": "second"}
    assert calls == ["first", "second"]


def test_a_step_with_a_kind_corrects_at_most_once():
    calls: list[str] = []
    corrector = RequestCorrector("chat", steps=(_step("once", calls, {"fixed": 1}),))

    assert corrector.correct(_rejection(RuntimeError("no"))) == {"fixed": 1}
    assert corrector.correct(_rejection(RuntimeError("no"))) is None
    assert calls == ["once"]


def test_a_step_without_a_kind_may_correct_again():
    calls: list[str] = []
    corrector = RequestCorrector("chat", steps=(_step(None, calls, {"fixed": 1}),))

    assert corrector.correct(_rejection(RuntimeError("no"))) == {"fixed": 1}
    assert corrector.correct(_rejection(RuntimeError("no"))) == {"fixed": 1}
    assert calls == ["kindless", "kindless"]


def test_reasoning_correction_runs_before_transport_steps():
    calls: list[str] = []
    corrector = RequestCorrector(
        "chat",
        reasoning=ReasoningCorrection((("reasoning_effort",),), "max_tokens", None),
        steps=(_step("provider", calls, {"corrected": "provider"}),),
    )
    body: JsonObject = {"model": "model", "reasoning_effort": "none", "max_tokens": 9}
    error = _http_400(
        {
            "message": "Unsupported parameter: reasoning_effort",
            "param": "reasoning_effort",
        }
    )

    corrected = corrector.correct(Rejection(error, body, body))

    assert corrected is not None
    assert "reasoning_effort" not in corrected
    assert calls == []


def test_history_record_repair_runs_before_any_step():
    calls: list[str] = []
    first = {"type": "reasoning.text", "text": "First", "signature": "first-sig"}
    carrier = encode_replay(ReplayRecord(ORIGIN, {"reasoning_details": [first]}))
    source: JsonObject = {
        "model": "model",
        "messages": [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": carrier}],
            }
        ],
    }
    corrector = RequestCorrector("chat", steps=(_step("step", calls, {"x": 1}),))
    sent = corrector.prepare_history(source, ORIGIN, HistoryModeCache())
    error = _http_400(
        {
            "code": "invalid_signature",
            "message": "invalid signature",
            "param": "messages[0].reasoning_details[0]",
        }
    )

    corrected = corrector.correct(Rejection(error, source, sent))

    assert corrected is not None
    assert "first-sig" not in json.dumps(
        corrector.prepare_history(corrected, ORIGIN, HistoryModeCache())
    )
    assert calls == []


def test_history_mode_moves_on_after_an_http_400_no_step_corrects():
    calls: list[str] = []
    corrector = RequestCorrector("chat", steps=(_step("step", calls, None),))
    source = _readable_source()
    sent = corrector.prepare_history(source, ORIGIN, HistoryModeCache())

    corrected = corrector.correct(
        Rejection(_http_400({"message": "bad request"}), source, sent)
    )

    assert corrected is not None
    rendered = json.dumps(
        corrector.prepare_history(corrected, ORIGIN, HistoryModeCache())
    )
    assert "Old" not in rendered and "New" in rendered
    assert calls == ["step"]


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("bad request"),
        ExecutionFailure(FailureKind.INVALID_REQUEST, 400, "bad request", False),
    ],
    ids=["no-status", "canonical-400-without-response"],
)
def test_history_mode_stays_without_an_http_400_response(error):
    corrector = RequestCorrector("chat")
    source = _readable_source()
    sent = corrector.prepare_history(source, ORIGIN, HistoryModeCache())

    assert corrector.correct(Rejection(error, source, sent)) is None
