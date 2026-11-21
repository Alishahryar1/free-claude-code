"""A request corrector decides how to correct one rejection, in a fixed order.

Order: history record repair, reasoning control, transport steps, and the
history-mode fallback after an HTTP 400. Deciding has no effects: lessons are
returned, never applied.
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
from free_claude_code.providers.history_negotiation import (
    HistoryModeCache,
    RequestHistory,
)
from free_claude_code.providers.reasoning_compatibility import ReasoningCorrection
from free_claude_code.providers.request_correction import (
    Correction,
    CorrectionStep,
    NextHistoryMode,
    Proposal,
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


def _step(
    name: str, calls: list[str], result: JsonObject | None, *, once: bool = True
) -> CorrectionStep:
    def propose(rejection: Rejection) -> Proposal | None:
        calls.append(name)
        return Proposal(result) if result is not None else None

    return CorrectionStep(name, propose, once=once)


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


def _negotiated(
    error: Exception, source: JsonObject
) -> tuple[RequestHistory, Rejection]:
    history = RequestHistory()
    sent = history.prepare(source, ORIGIN, HistoryModeCache())
    return history, Rejection(
        error,
        source,
        sent,
        history_source=history.source,
        history_paths=history.paths,
    )


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

    assert corrector.correct(_rejection(RuntimeError("no"))) == Correction(
        {"corrected": "second"}, "second"
    )
    assert calls == ["first", "second"]


def test_a_once_step_corrects_at_most_once():
    calls: list[str] = []
    corrector = RequestCorrector("chat", steps=(_step("once", calls, {"fixed": 1}),))

    assert corrector.correct(_rejection(RuntimeError("no"))) is not None
    assert corrector.correct(_rejection(RuntimeError("no"))) is None
    assert calls == ["once"]


def test_a_repeatable_step_may_correct_again():
    calls: list[str] = []
    corrector = RequestCorrector(
        "chat", steps=(_step("again", calls, {"fixed": 1}, once=False),)
    )

    assert corrector.correct(_rejection(RuntimeError("no"))) is not None
    assert corrector.correct(_rejection(RuntimeError("no"))) is not None
    assert calls == ["again", "again"]


def test_a_lesson_is_returned_and_never_applied():
    applied: list[str] = []

    class Taught:
        def apply(self) -> None:
            applied.append("taught")

    lesson = Taught()
    corrector = RequestCorrector(
        "chat",
        steps=(
            CorrectionStep(
                "learn", lambda _rejection: Proposal({"fixed": 1}, "why", lesson)
            ),
        ),
    )

    assert corrector.correct(_rejection(RuntimeError("no"))) == Correction(
        {"fixed": 1}, "learn", "why", lesson
    )
    assert applied == []


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

    assert isinstance(corrected, Correction) and corrected.step == "reasoning"
    assert "reasoning_effort" not in corrected.body
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
    error = _http_400(
        {
            "code": "invalid_signature",
            "message": "invalid signature",
            "param": "messages[0].reasoning_details[0]",
        }
    )
    history, rejection = _negotiated(error, source)

    corrected = corrector.correct(rejection)

    assert isinstance(corrected, Correction) and corrected.step == "history_record"
    assert "first-sig" not in json.dumps(
        history.prepare(corrected.body, ORIGIN, HistoryModeCache())
    )
    assert calls == []


def test_history_mode_moves_on_after_an_http_400_no_step_corrects():
    calls: list[str] = []
    corrector = RequestCorrector("chat", steps=(_step("step", calls, None),))
    source = _readable_source()
    history, rejection = _negotiated(_http_400({"message": "bad request"}), source)

    assert corrector.correct(rejection) == NextHistoryMode()
    assert calls == ["step"]
    corrected = history.next_mode()
    assert corrected is not None
    rendered = json.dumps(history.prepare(corrected, ORIGIN, HistoryModeCache()))
    assert "Old" not in rendered and "New" in rendered


def test_history_mode_stays_without_negotiated_history():
    corrector = RequestCorrector("chat")
    error = _http_400({"message": "bad request"})

    assert corrector.correct(_rejection(error)) is None


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("bad request"),
        ExecutionFailure(FailureKind.INVALID_REQUEST, 400, "bad request", False),
    ],
    ids=["no-status", "canonical-400-without-response"],
)
def test_history_mode_stays_without_an_http_400_response(error: Exception):
    corrector = RequestCorrector("chat")
    _, rejection = _negotiated(error, _readable_source())

    assert corrector.correct(rejection) is None
