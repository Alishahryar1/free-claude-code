"""Decide how to correct one request's upstream rejections, in a fixed order.

The order is: repair a rejected history record, drop a rejected reasoning
control, run the transport's own steps, and finally move history replay to
the next mode after an HTTP 400. Deciding has no effects: what a rejection
teaches is returned as a lesson for the attempt loop to apply.
"""

from collections.abc import Sequence

from free_claude_code.core.history_replay import HistoryProtocol
from free_claude_code.providers.history_replay import history_retry_body
from free_claude_code.providers.reasoning_compatibility import ReasoningCorrection

from .rejection import (
    Correction,
    CorrectionDecision,
    CorrectionStep,
    NextHistoryMode,
    Proposal,
    Rejection,
)


class RequestCorrector:
    """Remember which corrections one request body has already used."""

    def __init__(
        self,
        protocol: HistoryProtocol,
        *,
        reasoning: ReasoningCorrection | None = None,
        steps: Sequence[CorrectionStep] = (),
    ) -> None:
        self._protocol = protocol
        self._steps = (
            (_reasoning_step(reasoning),) if reasoning is not None else ()
        ) + tuple(steps)
        self._used: set[str] = set()

    def correct(self, rejection: Rejection) -> CorrectionDecision | None:
        """Return how to correct this rejection, or ``None`` when nothing can."""
        repaired = history_retry_body(
            rejection.error,
            rejection.sent_body,
            self._protocol,
            source=rejection.history_source,
            source_paths=rejection.history_paths,
        )
        if repaired is not None:
            return Correction(
                repaired, "history_record", "repaired a rejected history record"
            )
        for step in self._steps:
            if step.once and step.name in self._used:
                continue
            proposal = step.propose(rejection)
            if proposal is not None:
                if step.once:
                    self._used.add(step.name)
                return Correction(
                    proposal.body, step.name, proposal.note, proposal.lesson
                )
        if (
            rejection.history_source is not None
            and _http_response_status(rejection.error) == 400
        ):
            return NextHistoryMode()
        return None


def _reasoning_step(reasoning: ReasoningCorrection) -> CorrectionStep:
    def propose(rejection: Rejection) -> Proposal | None:
        body = reasoning.retry_body(
            rejection.error, rejection.body, sent_body=rejection.sent_body
        )
        if body is None:
            return None
        return Proposal(body, "dropped a rejected reasoning control")

    return CorrectionStep("reasoning", propose)


def _http_response_status(error: Exception) -> int | None:
    # Only an HTTP response's own status. An in-stream failure that is
    # canonically a 400 does not imply the history mode was rejected.
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) else None
