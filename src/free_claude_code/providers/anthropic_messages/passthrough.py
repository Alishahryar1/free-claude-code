"""Opaque native Messages attempts under the shared recovery coordinator."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx

from free_claude_code.application.ports import ProviderCandidate
from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    validate_messages_json,
)
from free_claude_code.core.anthropic.native_stream import NativeMessagesStreamState
from free_claude_code.core.anthropic.passthrough import (
    NativeMessagesRequest,
    restore_native_history,
)
from free_claude_code.core.diagnostics import (
    extract_upstream_error_detail,
    format_execution_failure_message,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.history_replay import ReplayOrigin
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.recovery import RecoveryCheckpoint
from free_claude_code.core.recovery_request import continue_request
from free_claude_code.core.stream_events import RequestOutcome, StreamEvent
from free_claude_code.core.stream_observations import (
    DecodedStreamEvent,
    MessagesObservation,
)
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.failure_policy import RetryableProviderProtocolError
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.stream_candidate import (
    StreamCandidate,
    content_progress,
    request_may_run_server_tools,
)

from .wire import check_messages_failure, messages_events, messages_status_error


@asynccontextmanager
async def open_native_messages(
    client: httpx.AsyncClient,
    admission: ProviderAdmissionController,
    *,
    base_url: str,
    headers: Mapping[str, str],
    body: JsonObject,
    public_model: str,
    provider_name: str,
    read_timeout_s: float,
    request_id: str | None,
    origin: ReplayOrigin,
) -> AsyncIterator[ProviderCandidate]:
    candidate = _NativeMessagesCandidate(
        client,
        admission,
        base_url=base_url,
        headers=headers,
        body=body,
        public_model=public_model,
        provider_name=provider_name,
        read_timeout_s=read_timeout_s,
        request_id=request_id,
        origin=origin,
    )
    try:
        yield candidate
    finally:
        await candidate.aclose()


class _NativeMessagesCandidate(StreamCandidate):
    def __init__(
        self,
        client: httpx.AsyncClient,
        admission: ProviderAdmissionController,
        *,
        base_url: str,
        headers: Mapping[str, str],
        body: JsonObject,
        public_model: str,
        provider_name: str,
        read_timeout_s: float,
        request_id: str | None,
        origin: ReplayOrigin,
    ) -> None:
        super().__init__(
            admission=admission,
            provider_name=provider_name,
            protocol="messages",
            read_timeout_s=read_timeout_s,
            request_id=request_id,
        )
        self._client = client
        self._base_url = base_url
        self._headers = dict(headers)
        self._request = NativeMessagesRequest(body)
        self.original_body = self._request.body
        self._public_model = public_model
        self._native_origin = origin

    def _build_body(self, checkpoint: RecoveryCheckpoint) -> None:
        self.body = continue_request(self._request, checkpoint).body
        self.replay_safe = not request_may_run_server_tools(self.body, "messages")

    async def _prepare_endpoint(self) -> ReplayOrigin:
        self.sent_body = restore_native_history(self.body, self._native_origin)
        return self._native_origin

    def _correction(self, error: Exception) -> JsonObject | None:
        # Opaque native requests do not authorize normalized history/body rewrites.
        return None

    async def _read(
        self, scope: ProviderAttemptScope
    ) -> AsyncIterator[DecodedStreamEvent]:
        assert self.origin is not None
        request = self._client.build_request(
            "POST",
            self._base_url.rstrip("/") + "/messages",
            json=self.sent_body,
            headers=self._headers,
        )
        self._dispatch(scope)
        response = scope.retain(await self._client.send(request, stream=True))
        if not response.is_success:
            raise await messages_status_error(response)
        if not self._request.stream:
            await response.aread()
            try:
                message = response.json()
                validate_messages_json(message)
            except (ValueError, NativeMessagesError) as error:
                raise RetryableProviderProtocolError(
                    "Messages upstream returned invalid JSON."
                ) from error
            if not isinstance(message, dict) or message.get("type") != "message":
                raise RetryableProviderProtocolError(
                    "Messages upstream did not return a Message object."
                )
            content = message.get("content", [])
            invalid = next(
                (
                    index
                    for index, block in enumerate(content)
                    if isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and not isinstance(block.get("input"), dict)
                ),
                None,
            )
            if invalid is not None:
                if message.get("stop_reason") != "max_tokens":
                    raise RetryableProviderProtocolError(
                        "Native tool input must be a complete JSON object."
                    )
                message = {**message, "content": content[:invalid]}
            await scope.attempt.accept()
            yield DecodedStreamEvent(
                self.origin,
                StreamEvent("message", message),
                progress=True,
                outcome=RequestOutcome.INCOMPLETE
                if message.get("stop_reason") in {"max_tokens", "pause_turn"}
                else RequestOutcome.SUCCESS,
                stop_reason=message.get("stop_reason"),
            )
            return
        if "text/event-stream" not in response.headers.get("content-type", "").lower():
            raise RetryableProviderProtocolError(
                "Messages upstream did not return an SSE stream."
            )
        source = NativeMessagesStreamState(permissive=True)
        async for kind, payload in messages_events(response):
            self.record_usage(payload)
            check_messages_failure(kind, payload, native=True)
            if kind == "ping" and not source.started:
                continue
            if kind == "message_start" and not isinstance(payload.get("message"), dict):
                raise NativeMessagesError("Invalid native message_start.")
            completed = source.accept(kind, payload)
            if (
                source.completed
                and source.invalid_input
                and source.stop_reason != "max_tokens"
            ):
                raise RetryableProviderProtocolError(
                    "Provider completed a response with unfinished or invalid tool input."
                )
            if not scope.attempt.accepted:
                await scope.attempt.accept()
            yield DecodedStreamEvent(
                self.origin,
                StreamEvent(kind, payload),
                observation=MessagesObservation(completed),
                progress=content_progress("messages", kind, payload),
                outcome=(
                    RequestOutcome.INCOMPLETE
                    if source.stop_reason in {"max_tokens", "pause_turn"}
                    else RequestOutcome.SUCCESS
                )
                if source.completed
                else None,
                stop_reason=source.stop_reason,
                native_reasoning_pending=source.native_reasoning_pending,
            )
            if source.completed:
                return
        raise RetryableProviderProtocolError(
            "Messages stream ended without message_stop."
        )

    def _effective_error(self, error: Exception) -> Exception:
        return (
            RetryableProviderProtocolError(str(error))
            if isinstance(error, NativeMessagesError)
            else error
        )

    def _failure(self, error: Exception) -> ExecutionFailure:
        if isinstance(error, ExecutionFailure):
            return replace(
                error,
                message=format_execution_failure_message(
                    error,
                    extract_upstream_error_detail(error),
                    upstream_name=self.provider_name,
                    request_id=self.request_id,
                ),
            )
        return super()._failure(error)
