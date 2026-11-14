"""The evidence one upstream rejection leaves for correction."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from free_claude_code.core.json_types import JsonObject


@dataclass(frozen=True, slots=True)
class Rejection:
    """One upstream rejection and the request it rejected."""

    error: Exception
    body: JsonObject
    """The request the transport meant to send."""
    sent_body: Mapping[str, Any]
    """What was sent, after history rendering and provider preparation."""


@dataclass(frozen=True, slots=True)
class CorrectionStep:
    """One transport-owned correction for a rejected request.

    A step with a ``kind`` runs at most once per request; a step without one
    may run again, and must return ``None`` once it has nothing left to fix.
    """

    kind: str | None
    propose: Callable[[Rejection], JsonObject | None]
