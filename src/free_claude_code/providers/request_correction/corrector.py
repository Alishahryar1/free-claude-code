"""Turn one request's upstream rejections into corrected bodies, in a fixed order.

The order is: repair a rejected history record, drop a rejected reasoning
control, run the transport's own steps, and finally move history replay to
the next mode after an HTTP 400.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from free_claude_code.core.history_replay import HistoryProtocol, ReplayOrigin
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.history_negotiation import (
    HistoryModeCache,
    HistoryNegotiation,
)
from free_claude_code.providers.history_replay import history_retry_body
from free_claude_code.providers.reasoning_compatibility import ReasoningCorrection

from .rejection import CorrectionStep, Rejection


class RequestCorrector:
    """Own the corrections of one request body and what they already used."""

    def __init__(
        self,
        protocol: HistoryProtocol,
        *,
        reasoning: ReasoningCorrection | None = None,
        steps: Sequence[CorrectionStep] = (),
    ) -> None:
        self._protocol = protocol
        self._history: HistoryNegotiation | None = None
        self._steps = (
            (_reasoning_step(reasoning),) if reasoning is not None else ()
        ) + tuple(steps)
        self._used: set[str] = set()

    @property
    def history(self) -> HistoryNegotiation | None:
        """This request's history negotiation, once history has been prepared."""
        return self._history

    def prepare_history(
        self,
        body: Mapping[str, Any],
        origin: ReplayOrigin,
        cache: HistoryModeCache,
        *,
        reasoning_field: str = "reasoning_content",
        structured_details: bool = True,
        finalize: Callable[[JsonObject], JsonObject] | None = None,
        continuation: bool = False,
        native_chat: bool = False,
    ) -> JsonObject:
        """Render history in this request's negotiated replay mode."""
        if self._history is None:
            self._history = HistoryNegotiation(cache, origin)
        return self._history.prepare(
            body,
            origin,
            reasoning_field=reasoning_field,
            structured_details=structured_details,
            finalize=finalize,
            continuation=continuation,
            native_chat=native_chat,
        )

    def accepted_history(self) -> None:
        """Record that the upstream accepted this request's history mode."""
        if self._history is not None:
            self._history.accepted()

    def correct(self, rejection: Rejection) -> JsonObject | None:
        """Return a corrected body for this rejection, or ``None``."""
        history = self._history
        corrected = history_retry_body(
            rejection.error,
            rejection.sent_body,
            self._protocol,
            source=history.source if history is not None else None,
            source_paths=history.paths if history is not None else None,
        )
        if corrected is not None:
            return corrected
        for step in self._steps:
            if step.kind is not None and step.kind in self._used:
                continue
            corrected = step.propose(rejection)
            if corrected is not None:
                if step.kind is not None:
                    self._used.add(step.kind)
                return corrected
        if history is not None and _http_response_status(rejection.error) == 400:
            return history.rejected()
        return None


def _reasoning_step(reasoning: ReasoningCorrection) -> CorrectionStep:
    return CorrectionStep(
        "reasoning",
        lambda rejection: reasoning.retry_body(
            rejection.error, rejection.body, sent_body=rejection.sent_body
        ),
    )


def _http_response_status(error: Exception) -> int | None:
    # Only an HTTP response's own status. An in-stream failure that is
    # canonically a 400 does not imply the history mode was rejected.
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) else None
