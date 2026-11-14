"""The single answer to one failed stream attempt."""

from dataclasses import dataclass

from free_claude_code.core.json_types import JsonObject


@dataclass(frozen=True, slots=True)
class Retry:
    """Open another attempt of the same operation with this body."""

    body: JsonObject


@dataclass(frozen=True, slots=True)
class Continue:
    """Extend the delivered response with this continuation body."""

    body: JsonObject


@dataclass(frozen=True, slots=True)
class Handoff:
    """Finish the response on the content already released."""


@dataclass(frozen=True, slots=True)
class Fail:
    """Raise this error, chained to the failure the transport caught."""

    error: Exception


type StreamFailureResolution = Retry | Continue | Handoff | Fail
