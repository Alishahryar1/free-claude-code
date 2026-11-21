"""The evidence one upstream rejection leaves, and what correcting it yields."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from free_claude_code.core.history_replay import HistoryPath
from free_claude_code.core.json_types import JsonObject


@dataclass(frozen=True, slots=True)
class Rejection:
    """One upstream rejection and the request it rejected."""

    error: Exception
    body: JsonObject
    """The request the transport meant to send."""
    sent_body: Mapping[str, Any]
    """What was sent, after history rendering and provider preparation."""
    history_source: JsonObject | None = None
    """The normalized history the request renders, when it negotiates one."""
    history_paths: Mapping[HistoryPath, HistoryPath] | None = None
    """Where each rendered history record came from in ``history_source``."""


class Lesson(Protocol):
    """What a rejection teaches a memory that outlives the request."""

    def apply(self) -> None:
        """Teach the memory. The attempt loop calls this, never a corrector."""
        ...


@dataclass(frozen=True, slots=True)
class Proposal:
    """A corrected request, and what the rejection taught."""

    body: JsonObject
    note: str | None = None
    """What changed, for the operator."""
    lesson: Lesson | None = None


@dataclass(frozen=True, slots=True)
class CorrectionStep:
    """One transport-owned correction for a rejected request.

    A step that runs ``once`` corrects at most once per request; any other
    step may run again, and must return ``None`` once it has nothing to fix.
    """

    name: str
    propose: Callable[[Rejection], Proposal | None]
    once: bool = True


@dataclass(frozen=True, slots=True)
class Correction:
    """Send ``body`` instead of the rejected request."""

    body: JsonObject
    step: str
    note: str | None = None
    lesson: Lesson | None = None


@dataclass(frozen=True, slots=True)
class NextHistoryMode:
    """Render the request's history in its next replay mode."""


type CorrectionDecision = Correction | NextHistoryMode
