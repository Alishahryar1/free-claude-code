"""Authorize credential refresh and corrected retries for one execution."""

from collections.abc import Callable

from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.admission import (
    ProviderAttempt,
    ProviderCorrectionAction,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.endpoint import RequestEndpoint


class RequestRecovery:
    """Authorize shared corrections within one execution and public stream."""

    def __init__(
        self,
        execution: ProviderExecution,
        *,
        endpoint: RequestEndpoint | None = None,
    ) -> None:
        self._execution = execution
        self._endpoint = endpoint
        self._refreshed = False

    @property
    def execution(self) -> ProviderExecution:
        return self._execution

    @property
    def _committed(self) -> bool:
        delivery = self.execution.delivery
        return delivery is not None and delivery.content_released

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
            or not self._endpoint.can_refresh
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
        normal_stop_seen: bool = False,
    ) -> JsonObject | None:
        delivery = self.execution.delivery
        if (
            normal_stop_seen
            and delivery is not None
            and (
                not delivery.content_released
                or operation_kind is ProviderOperationKind.CONTINUATION
            )
        ):
            return None
        if await self.retry_authentication(error, auth_status, attempt):
            return body
        if (
            delivery.attempt_content_released
            if delivery is not None
            else operation_kind is ProviderOperationKind.GENERATION and self._committed
        ):
            return None
        # Correct only an attempt whose public content can still be replaced.
        corrected = propose_correction()
        if corrected is not None and await self._authorize(error, attempt):
            return corrected
        return None
