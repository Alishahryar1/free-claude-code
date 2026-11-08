"""Same-provider continuation using evidence from the public delivery boundary."""

from dataclasses import dataclass, replace
from typing import Any

from free_claude_code.core.continuation_request import (
    ContinuationRequest,
    UpstreamProtocol,
)
from free_claude_code.core.delivered_response import (
    client_call,
    is_structured_text_format,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.history_replay import protected_reasoning
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_recovery import (
    RecoveryAction,
    SourceRecoverySnapshot,
    StreamFailureContext,
    assess_recovery,
)
from free_claude_code.core.trace import trace_event

from .admission import ProviderExecution


class SourceRecoveryState:
    """Native source evidence cannot be inferred from buffered public events."""

    def __init__(
        self,
        execution: ProviderExecution,
        *,
        previous: SourceRecoveryState | None = None,
    ) -> None:
        self._execution = execution
        self._prior_unsafe_reason = (
            previous.unsafe_reason or previous._prior_unsafe_reason
            if previous is not None
            else None
        )
        self.unsafe_reason: str | None = None
        self._pending_reasoning: set[int | str] = set()
        self._signed: set[int | str] = set()
        self._handoff_blocked = False

    @property
    def can_handoff(self) -> bool:
        return not self._handoff_blocked and not self._pending_reasoning

    def snapshot(
        self,
        *,
        body: dict[str, Any],
        protocol: UpstreamProtocol,
        normal_stop_seen: bool,
        for_model_fallback: bool = False,
    ) -> SourceRecoverySnapshot:
        return SourceRecoverySnapshot(
            normal_stop_seen=normal_stop_seen,
            structured_output=_structured_output(body, protocol),
            unsafe_reason=self.unsafe_reason
            or (self._prior_unsafe_reason if for_model_fallback else None),
            can_handoff=self.can_handoff,
        )

    def observe(
        self,
        protocol: UpstreamProtocol,
        payload: dict[str, Any],
        *,
        continuing: bool = False,
    ) -> None:
        kind = str(payload.get("type", ""))
        usage = payload.get("usage")
        if protocol == "messages" and kind == "message_start":
            usage = payload.get("message", {}).get("usage")
        elif protocol == "responses":
            usage = payload.get("response", {}).get("usage")
        if isinstance(usage, dict) and usage:
            trace_event(
                stage="provider",
                event="provider.attempt.usage",
                source="provider",
                request_id=self._execution.request_id,
                execution_id=self._execution.execution_id,
                attempt=self._execution.attempts_started,
                usage=usage,
            )
        if protocol == "gemini":
            # The final native replay carrier is published only after the source
            # completes. A call alone cannot prove its late signature is complete.
            self._handoff_blocked = True
            for candidate in payload.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if part.get("thoughtSignature") and self.unsafe_reason is None:
                        self.unsafe_reason = "signed_thinking"
                    if any(
                        key
                        not in {"text", "thought", "thoughtSignature", "functionCall"}
                        for key in part
                    ):
                        self.unsafe_reason = "native_content"
        elif protocol == "chat":
            if len(payload.get("choices", [])) > 1 or any(
                choice.get("index", 0) != 0 or choice.get("logprobs")
                for choice in payload.get("choices", [])
            ):
                self.unsafe_reason = "native_content"
                self._handoff_blocked = True
            for choice in payload.get("choices", []):
                delta = choice.get("delta") or {}
                if protected_reasoning({"extra_content": delta.get("extra_content")}):
                    self.unsafe_reason = "native_content"
                details = delta.get("reasoning_details")
                if delta.get("audio") or (
                    details
                    and (
                        not isinstance(details, list)
                        or any(
                            not isinstance(detail, dict)
                            or detail.get("type")
                            not in {"reasoning.text", "reasoning.summary"}
                            or protected_reasoning(detail)
                            for detail in details
                        )
                    )
                ):
                    self.unsafe_reason = "native_content"
                    self._handoff_blocked = True
        elif protocol == "messages":
            block = payload.get("content_block")
            index = payload.get("index", -1)
            if isinstance(block, dict) and block.get("type") in {
                "thinking",
                "redacted_thinking",
            }:
                self.unsafe_reason = "signed_thinking"
                self._pending_reasoning.add(index)
                if block.get("signature") or block.get("data"):
                    self._signed.add(index)
            if isinstance(block, dict) and block.get("type") not in {
                "text",
                "tool_use",
                "thinking",
                "redacted_thinking",
            }:
                self.unsafe_reason = "native_content"
                self._handoff_blocked = True
            delta = payload.get("delta") or {}
            if (isinstance(block, dict) and block.get("citations")) or delta.get(
                "type"
            ) == "citations_delta":
                self.unsafe_reason = "native_content"
                self._handoff_blocked = True
            if delta.get("type") in {"thinking_delta", "signature_delta"}:
                self.unsafe_reason = "signed_thinking"
                if delta.get("signature"):
                    self._signed.add(index)
            if kind == "content_block_stop" and index in self._signed:
                self._pending_reasoning.discard(index)
        else:
            item = payload.get("item")
            if isinstance(item, dict):
                self._response_item(item, completed=kind == "response.output_item.done")
            response = payload.get("response")
            if isinstance(response, dict):
                for item in response.get("output", []):
                    if isinstance(item, dict):
                        self._response_item(
                            item, completed=item.get("status") == "completed"
                        )
            part = payload.get("part")
            if (
                kind in {"response.content_part.added", "response.content_part.done"}
                and isinstance(part, dict)
                and part.get("annotations")
            ):
                self.unsafe_reason = "native_content"
                self._handoff_blocked = True
            if kind.startswith(
                (
                    "response.web_search",
                    "response.code_interpreter",
                    "response.mcp_",
                    "response.image_generation",
                    "response.audio",
                )
            ):
                self.unsafe_reason = "hosted_or_native_content"
                self._handoff_blocked = True
            if kind == "response.output_text.annotation.added":
                self.unsafe_reason = "native_content"
                self._handoff_blocked = True
        if (
            continuing
            and self.unsafe_reason
            and not (protocol == "gemini" and self.unsafe_reason == "signed_thinking")
        ):
            raise ExecutionFailure(
                kind=FailureKind.UPSTREAM,
                status_code=502,
                message="The continuation returned native state that cannot be combined with the delivered response.",
                retryable=False,
            )

    def _response_item(self, item: dict[str, Any], *, completed: bool) -> None:
        if item.get("type") not in {"message", "reasoning"} and not client_call(item):
            self.unsafe_reason = "hosted_or_native_content"
            self._handoff_blocked = True
        if item.get("encrypted_content"):
            self.unsafe_reason = "opaque_reasoning"
            if not completed:
                self._pending_reasoning.add(item.get("id", ""))
        if completed:
            self._pending_reasoning.discard(item.get("id", ""))
        if any(
            part.get("annotations")
            or part.get("type")
            not in {"output_text", "refusal", "reasoning_text", "summary_text"}
            for part in [*(item.get("content") or []), *(item.get("summary") or [])]
        ):
            self.unsafe_reason = "native_content"
            self._handoff_blocked = True


@dataclass(frozen=True, slots=True)
class PublicRecovery:
    body: dict[str, Any] | None = None


def public_stream_failure(
    failure: ExecutionFailure,
    *,
    source: SourceRecoveryState,
    body: dict[str, Any],
    protocol: UpstreamProtocol,
    normal_stop_seen: bool,
    responses_failure_payload: JsonObject | None = None,
) -> ExecutionFailure:
    """Retain source evidence until routing has decided the public outcome."""
    return replace(
        failure,
        stream_context=StreamFailureContext(
            source.snapshot(
                body=body,
                protocol=protocol,
                normal_stop_seen=normal_stop_seen,
                for_model_fallback=True,
            ),
            responses_failure_payload,
        ),
    )


def _structured_output(body: dict[str, Any], protocol: UpstreamProtocol) -> bool:
    if protocol == "gemini":
        config = body.get("generationConfig") or {}
        text = (config.get("responseFormat") or {}).get("text") or {}
        return (
            config.get("responseMimeType") == "application/json"
            or text.get("mimeType") == "APPLICATION_JSON"
            or bool(
                config.get("responseSchema")
                or config.get("responseJsonSchema")
                or text.get("schema")
            )
        )
    if protocol == "chat":
        extra = body.get("extra_body") or {}
        value = extra.get("response_format", body.get("response_format"))
    else:
        options = body.get("text" if protocol == "responses" else "output_config") or {}
        value = options.get("format")
    return is_structured_text_format(value)


def public_recovery(
    execution: ProviderExecution,
    *,
    body: dict[str, Any],
    request: ContinuationRequest,
    retryable: bool,
    normal_stop_seen: bool,
    source: SourceRecoveryState,
) -> PublicRecovery | None:
    delivery = execution.delivery
    if delivery is None or not delivery.content_released:
        return None
    if not retryable:
        return None
    assessment = assess_recovery(
        content_released=True,
        prefix=delivery.prefix,
        source=source.snapshot(
            body=body, protocol=request.protocol, normal_stop_seen=normal_stop_seen
        ),
        retryable=retryable,
    )
    if assessment.action is RecoveryAction.STOP:
        trace_event(
            stage="provider",
            event="provider.recovery.ineligible",
            source="provider",
            request_id=execution.request_id,
            reason=assessment.reason,
        )
        return None
    if assessment.action is RecoveryAction.HANDOFF:
        delivery.begin_continuation(handoff=True)
        trace_event(
            stage="provider",
            event="provider.recovery.tool_handoff",
            source="provider",
            request_id=execution.request_id,
            usage_estimated=True,
        )
        return PublicRecovery()
    if not execution.can_attempt:
        trace_event(
            stage="provider",
            event="provider.recovery.exhausted",
            source="provider",
            request_id=execution.request_id,
            attempts_started=execution.attempts_started,
            max_attempts=execution.max_attempts,
        )
        return None
    seed = assessment.seed
    if seed is None:
        return None
    replacement = request.build(body, seed)
    delivery.begin_continuation()
    trace_event(
        stage="provider",
        event="provider.recovery.continuation",
        source="provider",
        request_id=execution.request_id,
        attempts_started=execution.attempts_started,
        max_attempts=execution.max_attempts,
        usage_estimated=True,
    )
    return PublicRecovery(body=replacement)
