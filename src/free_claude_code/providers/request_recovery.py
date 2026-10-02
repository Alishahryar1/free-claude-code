"""Shared request correction and authentication recovery decisions."""

from collections.abc import Callable, Mapping
from copy import copy, deepcopy
from typing import Any

from free_claude_code.core.history_replay import HistoryProtocol
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.admission import (
    ProviderAttempt,
    ProviderCorrectionAction,
    ProviderExecution,
)
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.history_replay import (
    history_retry_body,
    reapply_history_ids,
)
from free_claude_code.providers.reasoning_compatibility import ReasoningCorrection


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

    async def _authorize(self, error: Exception, attempt: ProviderAttempt) -> bool:
        return (
            self.execution.can_attempt
            if attempt.accepted
            else await attempt.correct(error) is ProviderCorrectionAction.RETRY
        )

    async def retry_authentication(
        self, error: Exception, auth_status: int | None, attempt: ProviderAttempt
    ) -> bool:
        if self._endpoint is None or auth_status not in {401, 403} or self._refreshed:
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
        propose_correction: Callable[[], JsonObject | None],
    ) -> JsonObject | None:
        if await self.retry_authentication(error, auth_status, attempt):
            return body
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
        self.reasoning = reasoning
        self._used_retry_kinds: set[str] = set()
        self._normalized_ids: dict[str, str] = {}
        self._controls: JsonObject = {}
        self._omitted_controls: set[str] = set()
        self.history_correction: Callable[[JsonObject], JsonObject | None] | None = None

    def copy(self) -> RequestCorrections:
        """Copy request decisions while retaining provider-owned policies."""
        result = copy(self)
        result._used_retry_kinds = self._used_retry_kinds.copy()
        result._normalized_ids = self._normalized_ids.copy()
        result._controls = deepcopy(self._controls)
        result._omitted_controls = self._omitted_controls.copy()
        return result

    def retain_controls(self, before: JsonObject, after: JsonObject) -> None:
        """Remember accepted control choices without retaining an old history body."""
        for key in before.keys() | after.keys():
            if key in {"model", "input", "messages", "extra_headers"}:
                continue
            if key not in after:
                self._omitted_controls.add(key)
                self._controls.pop(key, None)
            elif key not in before or before[key] != after[key]:
                self._controls[key] = deepcopy(after[key])
                self._omitted_controls.discard(key)

    def reapply(self, body: JsonObject) -> JsonObject:
        """Carry accepted optional controls across continuation preparation."""
        result = deepcopy(body)
        reapply_history_ids(result, self._protocol, self._normalized_ids)
        if self.reasoning is not None and "reasoning" in self._used_retry_kinds:
            result = self.reasoning.without_off_control(result)
        if "stream_usage" in self._used_retry_kinds:
            options = result.get("stream_options")
            if isinstance(options, dict):
                options.pop("include_usage", None)
                if not options:
                    result.pop("stream_options", None)
        for key in self._omitted_controls:
            result.pop(key, None)
        for key, value in self._controls.items():
            current = result.get(key)
            if (
                key in {"max_tokens", "max_completion_tokens", "max_output_tokens"}
                and isinstance(current, int)
                and isinstance(value, int)
            ):
                result[key] = min(current, value)
            else:
                result[key] = deepcopy(value)
        return result

    def reapply_history(self, body: JsonObject) -> JsonObject:
        """Apply accepted history choices after native replay has been restored."""
        reapply_history_ids(body, self._protocol, self._normalized_ids)
        return (
            (self.history_correction(body) or body) if self.history_correction else body
        )

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
        corrected = history_retry_body(
            history_error,
            sent_body,
            self._protocol,
            normalized_ids=self._normalized_ids,
        )
        if corrected is not None:
            return corrected
        if self.reasoning is not None and "reasoning" not in self._used_retry_kinds:
            corrected = self.reasoning.retry_body(
                reasoning_error, body, sent_body=reasoning_sent_body
            )
            if corrected is not None:
                self._used_retry_kinds.add("reasoning")
                return corrected
        return (
            after_common(self._used_retry_kinds) if after_common is not None else None
        )
