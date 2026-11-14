"""What a transport supplies for one attempt of a provider request."""

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.continuation import SourceRecoveryState
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.request_correction import RequestCorrector


@dataclass(frozen=True, slots=True)
class AttemptContext:
    """The request state one attempt starts from."""

    body: JsonObject
    operation_kind: ProviderOperationKind
    corrections: RequestCorrector
    source: SourceRecoveryState

    @property
    def continuing(self) -> bool:
        return self.operation_kind is ProviderOperationKind.CONTINUATION


class TransportAttempt:
    """One attempt's protocol-specific steps.

    The loop calls ``prepare`` before admission opens the attempt, then
    ``stream`` with the open scope. ``body`` and ``sent_body`` are what the
    attempt intended and actually sent; failures are resolved against them.
    """

    def __init__(self, context: AttemptContext) -> None:
        self.context = context
        self.body = context.body
        self.sent_body = context.body

    @property
    def normal_stop_seen(self) -> bool:
        return False

    async def prepare(self) -> None:
        """Resolve credentials and build the request; no provider call is made."""

    def stream(self, scope: ProviderAttemptScope) -> AsyncIterator[StreamOutput]:
        """Send the request and present its response, accepting the attempt."""
        raise NotImplementedError

    def resolvable(self, error: Exception) -> bool:
        """Whether ``error`` is a provider failure the loop should resolve."""
        return True

    def resolution_error(self, error: Exception) -> Exception:
        """The error the loop classifies in place of the raised one."""
        return error

    def abandon(self, error: Exception) -> Iterable[StreamOutput]:
        """Frames still owed to the client when a safe attempt fails."""
        return ()

    def failure_payload(
        self, error: Exception, failure: ExecutionFailure
    ) -> JsonObject | None:
        """The protocol's own failure event for a terminal ``failure``."""
        return None

    def handoff(self) -> Iterable[StreamOutput]:
        """Frames that end the response so the client can run its tools."""
        return ()

    def completed_fields(self) -> dict[str, Any]:
        """Extra fields for the completion trace."""
        return {}
