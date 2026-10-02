"""Physical streaming attempts under one provider admission execution."""

import asyncio
import sys
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from functools import partial

import httpx
import httpx2
import openai

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.history_replay import HistoryProtocol, ReplayOrigin
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.recovery import AttemptFailure, RecoveryCheckpoint
from free_claude_code.core.stream_events import DecodedStreamEvent, StreamEvent
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
from .history_replay import requires_native_origin
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
        self._revision: int | None = None
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

    @property
    def can_attempt(self) -> bool:
        return self.execution.can_attempt

    async def prepare(self, checkpoint: RecoveryCheckpoint) -> None:
        if self._revision != checkpoint.revision:
            self._build_body(checkpoint)
            self._revision = checkpoint.revision
        if checkpoint.required_origins:
            try:
                self.origin = await self._prepare_endpoint()
            except _PROVIDER_ERRORS as error:
                raise self._failure(error) from error
            checkpoint.require_origin(self.origin)

    @abstractmethod
    def _build_body(self, checkpoint: RecoveryCheckpoint) -> None: ...

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

    async def stream_attempt(
        self,
        checkpoint: RecoveryCheckpoint,
        *,
        wait_for_recovery: bool,
        can_correct: Callable[[], bool],
    ) -> AsyncIterator[DecodedStreamEvent]:
        operation = (
            ProviderOperationKind.CONTINUATION
            if checkpoint.content
            else ProviderOperationKind.GENERATION
        )
        scope: ProviderAttemptScope | None = None
        stream: AsyncIterator[DecodedStreamEvent] | None = None
        dispatched = completed = False
        self._reported_usage = {}
        try:
            await self.prepare(checkpoint)
            attempt = await self.execution.open_attempt(
                operation, wait_for_recovery=wait_for_recovery
            )
            scope = ProviderAttemptScope(
                attempt, provider_name=self.provider_name, request_id=self.request_id
            )
            # Credentials can change while waiting for admission.
            try:
                self.origin = await self._prepare_endpoint()
            except _PROVIDER_ERRORS as error:
                # Credential resolution is initialization, not a failed generation.
                raise AttemptFailure(self._failure(error)) from error
            checkpoint.require_origin(self.origin)
            dispatched = True
            # A remote operation can have run even if the first upstream byte is lost.
            yield DecodedStreamEvent(
                self.origin,
                StreamEvent("request.dispatched", {}),
                (),
                replay_safe=self.replay_safe,
                required_origins=(self.origin,)
                if requires_native_origin(self.sent_body, self.protocol)
                else (),
            )
            trace_event(
                self._request_trace_fields,
                stage="provider",
                event="provider.request.sent",
                source="provider",
                provider=self.provider_name,
                request_id=self.request_id,
                execution_id=self.execution.execution_id,
                transport=self.protocol,
                operation_kind=operation.value,
                downstream_model=self.body.get("model"),
            )
            stream = self._read(scope)
            async for event in stream:
                self.record_usage(event.source.payload)
                completed = event.completed
                yield event
        except asyncio.CancelledError, GeneratorExit:
            raise
        except ProviderRecoveryDeferred as error:
            raise AttemptFailure(
                self._failure(error.last_error), deferred=True
            ) from error
        except Exception as raw_error:
            error = self._effective_error(raw_error)
            if not isinstance(error, _PROVIDER_ERRORS):
                raise
            corrected = False
            if scope is not None:
                if can_correct():
                    body = await self.request_recovery.retry_request(
                        error,
                        provider_authentication_status(error),
                        scope.attempt,
                        self.body,
                        propose_correction=partial(self._correction, raw_error),
                    )
                    if body is not None:
                        self.body = body
                        corrected = True
                if not corrected and not scope.attempt.accepted:
                    await scope.attempt.fail(
                        error, provider_failure_override=self.failure_override
                    )
            if isinstance(error, ProviderRecoveryExhausted):
                self.execution.fail(error)
            failure = self._failure(error)
            trace_event(
                stage="provider",
                event="provider.response.error",
                source="provider",
                provider=self.provider_name,
                request_id=self.request_id,
                execution_id=self.execution.execution_id,
                transport=self.protocol,
                failure_kind=failure.kind.value,
                status_code=failure.status_code,
                corrected=corrected,
                exc_type=type(raw_error).__name__,
                provider_retryable=failure.retryable,
            )
            rejected = (
                scope is not None
                and not scope.attempt.accepted
                and (
                    provider_authentication_status(error) is not None
                    or (
                        failure.status_code in {400, 401, 403, 404, 413, 422, 429}
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
            )
            raise AttemptFailure(
                failure,
                corrected=corrected,
                retry_allowed=scope is not None
                and self.can_attempt
                and failure.retryable,
                request_rejected=rejected,
            ) from raw_error
        finally:
            if dispatched:
                trace_event(
                    stage="provider",
                    event="provider.attempt.usage",
                    source="provider",
                    provider=self.provider_name,
                    request_id=self.request_id,
                    execution_id=self.execution.execution_id,
                    transport=self.protocol,
                    operation_kind=operation.value,
                    downstream_model=self.body.get("model"),
                    completed=completed,
                    usage_status="reported" if self._reported_usage else "missing",
                    **self._reported_usage,
                )
            try:
                if stream is not None:
                    await maybe_await_aclose(stream)
            finally:
                if scope is not None:
                    active_error = sys.exception()
                    if completed and isinstance(active_error, GeneratorExit):
                        active_error = None
                    await scope.aclose(active_error=active_error)

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
