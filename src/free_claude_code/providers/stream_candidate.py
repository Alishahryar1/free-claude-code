"""Physical streaming attempts under one provider admission execution."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import suppress
from functools import partial

import httpx
import httpx2
import openai

from free_claude_code.core.async_iterators import CleanupBudget
from free_claude_code.core.failures import ExecutionFailure, UnsupportedRequestFeature
from free_claude_code.core.history_replay import HistoryProtocol, ReplayOrigin
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.recovery import (
    AttemptDispatch,
    AttemptFailure,
    CandidateIncompatible,
    RecoveryCheckpoint,
)
from free_claude_code.core.request_preservation import (
    require_output_limit,
    require_preserved_body,
)
from free_claude_code.core.stream_observations import DecodedStreamEvent
from free_claude_code.core.trace import trace_event

from .admission import ProviderAdmissionController, ProviderOperationKind
from .endpoint import RequestEndpoint
from .failure_policy import (
    ProviderFailureOverride,
    ProviderRecoveryDeferred,
    ProviderRecoveryExhausted,
    RetryableProviderProtocolError,
    classify_provider_failure,
    provider_authentication_status,
)
from .history_replay import require_original_origin, requires_native_origin
from .http import ProviderAttemptScope, maybe_await_aclose
from .request_recovery import RequestCorrections, RequestRecovery

_PROVIDER_ERRORS = (
    ExecutionFailure,
    openai.APIError,
    httpx.HTTPError,
    httpx2.HTTPError,
    RetryableProviderProtocolError,
    ProviderRecoveryExhausted,
    OSError,
    TimeoutError,
)


class StreamCandidate(ABC):
    """Expose one physical call at a time; the application decides what follows."""

    def __init__(
        self,
        *,
        admission: ProviderAdmissionController,
        provider_name: str,
        protocol: HistoryProtocol,
        read_timeout_s: float | None,
        request_id: str | None,
        endpoint: RequestEndpoint | None = None,
        failure_override: ProviderFailureOverride | None = None,
    ) -> None:
        self.execution = admission.start_execution(request_id=request_id)
        self.provider_name = provider_name
        self.protocol = protocol
        self.read_timeout_s = read_timeout_s
        self.request_id = request_id
        self.endpoint = endpoint
        self.failure_override = failure_override
        self.request_recovery = RequestRecovery(self.execution, endpoint=endpoint)
        self.corrections = RequestCorrections(protocol)
        self.body: JsonObject = {}
        self.sent_body: JsonObject = {}
        self.origin: ReplayOrigin | None = None
        self.replay_safe = True
        self.preserve_features = False
        self.original_body: JsonObject = {}
        self.input_protocol: HistoryProtocol = protocol
        self._reported_usage: dict[str, int] = {}

    def record_usage(self, payload: JsonObject) -> None:
        container = payload.get("response", payload.get("message", payload))
        if not isinstance(container, dict) or not isinstance(
            usage := container.get("usage"), dict
        ):
            return
        for source, target in (
            ("input_tokens", "input_tokens"),
            ("prompt_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("completion_tokens", "output_tokens"),
            ("cache_read_input_tokens", "cache_read_tokens"),
            ("cache_creation_input_tokens", "cache_creation_tokens"),
        ):
            value = usage.get(source)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self._reported_usage["reported_" + target] = value
        for source, child, target in (
            ("input_tokens_details", "cached_tokens", "cache_read_tokens"),
            ("prompt_tokens_details", "cached_tokens", "cache_read_tokens"),
            ("output_tokens_details", "reasoning_tokens", "reasoning_tokens"),
            ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens"),
        ):
            details = usage.get(source)
            value = details.get(child) if isinstance(details, dict) else None
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self._reported_usage["reported_" + target] = value

    def _request_trace_fields(self) -> JsonObject:
        return {}

    def _dispatch(self, scope: ProviderAttemptScope) -> None:
        scope.dispatched = True
        trace_event(
            self._request_trace_fields,
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=self.provider_name,
            request_id=self.request_id,
            execution_id=self.execution.execution_id,
            transport=self.protocol,
            operation_kind=scope.attempt.operation_kind.value,
            downstream_model=self.body.get("model"),
        )

    @property
    def can_attempt(self) -> bool:
        return self.execution.can_attempt

    def _prepare(self, checkpoint: RecoveryCheckpoint) -> None:
        self.preserve_features = checkpoint.recovering
        try:
            self._build_body(checkpoint)
            self.body = self.corrections.reapply(self.body)
            if self.preserve_features:
                require_output_limit(self.original_body, self.body)
        except UnsupportedRequestFeature as error:
            raise CandidateIncompatible(str(error)) from error

    @abstractmethod
    def _build_body(self, checkpoint: RecoveryCheckpoint) -> None: ...

    async def _validate_request(self) -> None:
        """Run provider validation after conversion and before admission."""
        return None

    @abstractmethod
    async def _prepare_endpoint(self) -> ReplayOrigin: ...

    @abstractmethod
    def _read(
        self, scope: ProviderAttemptScope
    ) -> AsyncIterator[DecodedStreamEvent]: ...

    def _correction(self, error: Exception) -> JsonObject | None:
        return self.corrections.next_body(
            error,
            self.body,
            sent_body=self.sent_body,
            reasoning_error=error,
        )

    def _effective_error(self, error: Exception) -> Exception:
        return error

    def _retain_correction(self, error: Exception, body: JsonObject) -> None:
        self.corrections.retain_controls(self.body, body)

    def _preserved_correction(self, error: Exception) -> JsonObject | None:
        corrected = self._correction(error)
        if corrected is not None and self.preserve_features:
            try:
                require_preserved_body(
                    self.corrections.reapply(self.body), corrected, "Request correction"
                )
                require_output_limit(self.original_body, corrected)
            except UnsupportedRequestFeature:
                return None
        return corrected

    def open_attempt(
        self, checkpoint: RecoveryCheckpoint, *, wait_for_recovery: bool
    ) -> CandidateAttempt:
        return CandidateAttempt(self, checkpoint, wait_for_recovery=wait_for_recovery)

    def _failure(self, error: Exception) -> ExecutionFailure:
        failure = classify_provider_failure(
            error,
            provider_name=self.provider_name,
            read_timeout_s=self.read_timeout_s,
            request_id=self.request_id,
            provider_failure_override=self.failure_override,
        )
        if failure is not error:
            failure.__cause__ = error
        return failure

    def finish(self, failure: ExecutionFailure | None) -> None:
        if failure is None:
            self.execution.succeed()
        else:
            self.execution.fail(failure)

    async def aclose(self) -> None:
        await self.execution.aclose()

    async def suspend(self) -> None:
        await self.execution.suspend()


class CandidateAttempt:
    """Own the read, admission and HTTP lifetime until the caller resolves failure."""

    def __init__(
        self,
        candidate: StreamCandidate,
        checkpoint: RecoveryCheckpoint,
        *,
        wait_for_recovery: bool,
    ) -> None:
        self._candidate = candidate
        self._checkpoint = checkpoint
        self._wait_for_recovery = wait_for_recovery
        self._operation = (
            ProviderOperationKind.CONTINUATION
            if checkpoint.content
            else ProviderOperationKind.GENERATION
        )
        self._scope: ProviderAttemptScope | None = None
        self._source: AsyncIterator[DecodedStreamEvent] | None = None
        self._read_task: asyncio.Task[DecodedStreamEvent] | None = None
        self._raw_error: Exception | None = None
        self._error: Exception | None = None
        self._completed = False
        self._closed = False
        self._cleanup = CleanupBudget()

    @property
    def dispatch(self) -> AttemptDispatch | None:
        candidate = self._candidate
        if self._scope is None or not self._scope.dispatched:
            return None
        assert candidate.origin is not None
        return AttemptDispatch(
            candidate.replay_safe,
            (candidate.origin,)
            if (
                requires_native_origin(candidate.sent_body, candidate.protocol)
                or requires_native_origin(
                    candidate.original_body, candidate.input_protocol
                )
            )
            else (),
        )

    async def read(self) -> DecodedStreamEvent:
        # The scope closes and drains this one read after timeout or cancellation.
        # Shielding here prevents repeated caller cancellation from owning its unwind.
        self._read_task = asyncio.create_task(self._advance())
        return await asyncio.shield(self._read_task)

    async def _advance(self) -> DecodedStreamEvent:
        candidate = self._candidate
        with self._cleanup.bind():
            try:
                if self._source is None:
                    candidate._reported_usage = {}
                    candidate._prepare(self._checkpoint)
                    await candidate._validate_request()
                    admitted = await candidate.execution.open_attempt(
                        self._operation, wait_for_recovery=self._wait_for_recovery
                    )
                    self._scope = ProviderAttemptScope(
                        admitted,
                        provider_name=candidate.provider_name,
                        request_id=candidate.request_id,
                    )
                    try:
                        candidate.origin = await candidate._prepare_endpoint()
                    except _PROVIDER_ERRORS as error:
                        # Credential initialization is not a failed generation.
                        raise AttemptFailure(candidate._failure(error)) from error
                    self._checkpoint.require_origin(candidate.origin)
                    if candidate.preserve_features:
                        require_original_origin(
                            candidate.original_body,
                            candidate.input_protocol,
                            candidate.origin,
                        )
                    candidate.sent_body = candidate.corrections.reapply_history(
                        candidate.sent_body
                    )
                    self._source = candidate._read(self._scope)
                event = await anext(self._source)
                candidate.record_usage(event.source.payload)
                self._completed = event.completed
                return event
            except ProviderRecoveryDeferred as error:
                raise AttemptFailure(
                    candidate._failure(error.last_error), deferred=True
                ) from error
            except UnsupportedRequestFeature as error:
                raise CandidateIncompatible(str(error)) from error
            except AttemptFailure, CandidateIncompatible, StopAsyncIteration:
                raise
            except Exception as raw_error:
                error = candidate._effective_error(raw_error)
                if not isinstance(error, _PROVIDER_ERRORS):
                    raise
                self._raw_error, self._error = raw_error, error
                failure = candidate._failure(error)
                rejected = self._scope is not None and (
                    provider_authentication_status(error) is not None
                    or (
                        not self._scope.attempt.accepted
                        and failure.status_code in {400, 401, 403, 404, 413, 422, 429}
                        and isinstance(
                            error,
                            (
                                httpx.HTTPStatusError,
                                httpx2.HTTPStatusError,
                                openai.APIStatusError,
                            ),
                        )
                    )
                )
                raise AttemptFailure(failure, request_rejected=rejected) from raw_error

    async def resolve_failure(
        self, failure: AttemptFailure, *, can_correct: bool
    ) -> AttemptFailure:
        candidate = self._candidate
        error = self._error
        if error is None:
            return failure
        if self._scope is not None:
            attempt = self._scope.attempt
            if can_correct:
                previous = candidate.corrections
                candidate.corrections = previous.copy()
                corrected = False
                try:
                    body = await candidate.request_recovery.retry_request(
                        error,
                        provider_authentication_status(error),
                        attempt,
                        candidate.body,
                        propose_correction=partial(
                            candidate._preserved_correction, self._raw_error or error
                        ),
                    )
                    if body is not None:
                        candidate._retain_correction(self._raw_error or error, body)
                        candidate.body = body
                        corrected = True
                finally:
                    if not corrected:
                        candidate.corrections = previous
                failure.corrected = corrected
            if not failure.corrected and not attempt.accepted:
                await attempt.fail(
                    error, provider_failure_override=candidate.failure_override
                )
        if isinstance(error, ProviderRecoveryExhausted):
            candidate.execution.fail(error)
        failure.retry_allowed = (
            self._scope is not None
            and candidate.can_attempt
            and failure.failure.retryable
        )
        trace_event(
            stage="provider",
            event="provider.response.error",
            source="provider",
            provider=candidate.provider_name,
            request_id=candidate.request_id,
            execution_id=candidate.execution.execution_id,
            transport=candidate.protocol,
            failure_kind=failure.failure.kind.value,
            status_code=failure.failure.status_code,
            corrected=failure.corrected,
            exc_type=type(self._raw_error).__name__,
            provider_retryable=failure.failure.retryable,
        )
        return failure

    async def aclose(self, *, active_error: BaseException | None) -> None:
        if self._closed:
            return
        self._closed = True
        candidate = self._candidate
        if self.dispatch is not None:
            trace_event(
                stage="provider",
                event="provider.attempt.usage",
                source="provider",
                provider=candidate.provider_name,
                request_id=candidate.request_id,
                execution_id=candidate.execution.execution_id,
                transport=candidate.protocol,
                operation_kind=self._operation.value,
                downstream_model=candidate.body.get("model"),
                completed=self._completed,
                usage_status="reported" if candidate._reported_usage else "missing",
                **candidate._reported_usage,
            )

        async def close() -> None:
            try:
                if self._read_task is not None:
                    if not self._read_task.done():
                        self._read_task.cancel()
                    with suppress(Exception, asyncio.CancelledError):
                        await self._read_task
                if self._source is not None:
                    await maybe_await_aclose(self._source)
            finally:
                if self._scope is not None:
                    await self._scope.aclose(active_error=active_error)

        await self._cleanup.run(close())


def content_progress(protocol: HistoryProtocol, kind: str, payload: JsonObject) -> bool:
    """Count new content, including buffered arguments, without counting pings."""
    if protocol == "chat":
        choices = payload.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            return False
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            return False
        if any(
            delta.get(key)
            for key in (
                "content",
                "reasoning",
                "reasoning_content",
                "reasoning_details",
            )
        ):
            return True
        calls = delta.get("tool_calls")
        return isinstance(calls, list) and any(
            isinstance(call, dict)
            and isinstance(function := call.get("function"), dict)
            and bool(function.get("arguments"))
            for call in calls
        )
    if protocol == "messages":
        delta = payload.get("delta")
        if kind == "content_block_delta" and isinstance(delta, dict):
            return any(
                bool(delta.get(key))
                for key in ("text", "thinking", "signature", "partial_json", "citation")
            )
        block = payload.get("content_block")
        return (
            kind == "content_block_start"
            and isinstance(block, dict)
            and any(
                bool(block.get(key)) for key in ("text", "thinking", "data", "input")
            )
        )
    if kind == "response.output_item.added":
        item = payload.get("item")
        return (
            isinstance(item, dict)
            and item.get("type") in {"function_call", "custom_tool_call"}
            and bool(item.get("arguments", item.get("input")))
        )
    return kind.endswith(".delta") and bool(payload.get("delta"))


def request_may_run_server_tools(body: JsonObject, protocol: HistoryProtocol) -> bool:
    """Only known client tool declarations can be replayed after uncertain dispatch."""
    if protocol != "chat" and (body.get("mcp_servers") or body.get("container")):
        return True
    tools = body.get("tools")
    if not isinstance(tools, list) or protocol == "chat":
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        kind = tool.get("type")
        if protocol == "responses":
            if kind == "namespace":
                if request_may_run_server_tools({"tools": tool.get("tools")}, protocol):
                    return True
                continue
            if kind in {"function", "custom"} or (
                kind == "tool_search" and tool.get("execution") == "client"
            ):
                continue
        elif kind in (None, "custom") or (
            isinstance(kind, str)
            and kind.startswith(("bash_", "text_editor_", "computer_"))
        ):
            continue
        return True
    return False
