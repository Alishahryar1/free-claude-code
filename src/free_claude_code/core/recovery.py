"""Request-local recovery values, independent of provider admission and HTTP."""

from dataclasses import dataclass
from typing import Literal

from .failures import ExecutionFailure
from .history_replay import ReplayOrigin
from .json_types import JsonObject

type WireApi = Literal["messages", "responses"]


class CandidateIncompatible(Exception):
    """A candidate cannot represent the required request or recovery state."""


@dataclass(slots=True, eq=False)
class AttemptFailure(Exception):
    failure: ExecutionFailure
    corrected: bool = False
    deferred: bool = False
    retry_allowed: bool = False
    request_rejected: bool = False
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.failure.message)


@dataclass(frozen=True, slots=True)
class RecoveryCheckpoint:
    wire_api: WireApi
    content: tuple[JsonObject, ...] = ()
    text: str = ""
    revision: int = 0
    published_tools: bool = False
    required_origins: tuple[ReplayOrigin, ...] = ()
    blocked_reason: str | None = None
    recovering: bool = False

    def require_origin(self, origin: ReplayOrigin) -> None:
        if self.blocked_reason is not None:
            raise CandidateIncompatible(self.blocked_reason)
        if any(not origin.accepts(source) for source in self.required_origins):
            raise CandidateIncompatible(
                "Committed native state belongs to another origin."
            )
