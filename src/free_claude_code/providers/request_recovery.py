"""Shared request correction and authentication recovery decisions."""

from collections.abc import Callable, Mapping
from typing import Any

from free_claude_code.core.history_replay import HistoryProtocol
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.admission import (
    ProviderAttempt,
    ProviderCorrectionAction,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.history_replay import history_retry_body
from free_claude_code.providers.reasoning_compatibility import ReasoningCorrection
from free_claude_code.providers.stream_recovery import RecoveryController


class RequestRecovery:
    """Authorize shared corrections within one execution and public stream."""

    def __init__(
        self,
        execution: ProviderExecution,
        *,
        endpoint: RequestEndpoint | None = None,
        stream: RecoveryController | None = None,
    ) -> None:
        self._execution = execution
        self._endpoint = endpoint
        self._stream = stream
        self._refreshed = False

    @property
    def execution(self) -> ProviderExecution:
        return self._execution

    @property
    def _committed(self) -> bool:
        return self._stream is not None and self._stream.committed

    async def _authorize(self, error: Exception, attempt: ProviderAttempt) -> bool:
        return (
            self.execution.can_attempt
            if attempt.accepted
            else await attempt.correct(error) is ProviderCorrectionAction.RETRY
        )

    async def retry_authentication(
        self, error: Exception, auth_status: int | None, attempt: ProviderAttempt
    ) -> bool:
        if (
            self._endpoint is None
            or auth_status not in {401, 403}
            or self._refreshed
            or self._committed
        ):
            return False
        if not await self._authorize(error, attempt):
            return False
        self._refreshed = True
        self._endpoint.request_refresh()
        return True

    async def retry_request(
        self,
        error: Exception,
        auth_status: int | None,
        attempt: ProviderAttempt,
        body: JsonObject,
        *,
        operation_kind: ProviderOperationKind,
        propose_correction: Callable[[], JsonObject | None],
    ) -> JsonObject | None:
        if await self.retry_authentication(error, auth_status, attempt):
            return body
        if operation_kind is ProviderOperationKind.GENERATION and self._committed:
            return None
        # A separately buffered continuation/repair body can still be corrected
        # at creation, while authentication follows the original public stream.
        corrected = propose_correction()
        if corrected is not None and await self._authorize(error, attempt):
            return corrected
        return None


class RequestCorrections:
    """Retain common and transport correction history for one request body."""

    def __init__(
        self,
        protocol: HistoryProtocol,
        reasoning: ReasoningCorrection | None = None,
    ) -> None:
        self._protocol = protocol
        self._reasoning = reasoning
        self._used_retry_kinds: set[str] = set()

    def next_body(
        self,
        history_error: Exception,
        body: JsonObject,
        *,
        sent_body: Mapping[str, Any],
        reasoning_error: Exception,
        reasoning_sent_body: Mapping[str, Any] | None = None,
        after_common: Callable[[set[str]], JsonObject | None] | None = None,
    ) -> JsonObject | None:
        corrected = history_retry_body(history_error, sent_body, self._protocol)
        if corrected is not None:
            return corrected
        if self._reasoning is not None and "reasoning" not in self._used_retry_kinds:
            corrected = self._reasoning.retry_body(
                reasoning_error, body, sent_body=reasoning_sent_body
            )
            if corrected is not None:
                self._used_retry_kinds.add("reasoning")
                return corrected
        return (
            after_common(self._used_retry_kinds) if after_common is not None else None
        )
