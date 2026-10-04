"""Same-provider continuation using evidence from the public delivery boundary."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from free_claude_code.core.delivered_response import client_call
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.history_replay import reasoning_context
from free_claude_code.core.trace import trace_event

from .admission import ProviderExecution

type UpstreamProtocol = Literal["chat", "messages", "responses"]


class SourceRecoveryState:
    """Native source evidence cannot be inferred from buffered public events."""

    def __init__(self, execution: ProviderExecution) -> None:
        self._execution = execution
        self.unsafe_reason: str | None = None
        self._pending_reasoning: set[int | str] = set()
        self._signed: set[int | str] = set()
        self._handoff_blocked = False

    @property
    def can_handoff(self) -> bool:
        return not self._handoff_blocked and not self._pending_reasoning

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
        if protocol == "chat":
            for choice in payload.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("reasoning_details") or delta.get("audio"):
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
        if continuing and self.unsafe_reason:
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


def continuation_body(
    body: dict[str, Any], protocol: UpstreamProtocol, text: str, thinking: str
) -> dict[str, Any]:
    result = deepcopy(body)
    instruction = "The previous provider stream was interrupted. Continue the assistant response exactly where it stopped. Do not repeat text already written."
    if result.get("tools"):
        instruction += " No tool calls from this interrupted assistant turn were delivered or executed. Any unfinished calls were discarded. Emit the required tool calls using the provided tools."
    if thinking:
        instruction = f"{reasoning_context(thinking)}\n\n{instruction}"
    if protocol == "responses":
        original = result.get("input", [])
        if isinstance(original, str):
            original = [{"role": "user", "content": original}]
        result["input"] = [
            *original,
            *(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ]
                if text
                else []
            ),
            {"role": "user", "content": [{"type": "input_text", "text": instruction}]},
        ]
    else:
        result["messages"] = [
            *result.get("messages", []),
            *([{"role": "assistant", "content": text}] if text else []),
            {"role": "user", "content": instruction},
        ]
    return result


def public_recovery(
    execution: ProviderExecution,
    *,
    body: dict[str, Any],
    protocol: UpstreamProtocol,
    retryable: bool,
    normal_stop_seen: bool,
    source: SourceRecoveryState,
) -> PublicRecovery | None:
    delivery = execution.delivery
    if delivery is None or not delivery.content_released:
        return None
    if not retryable or normal_stop_seen:
        return None
    prefix = delivery.prefix
    if prefix is None:
        return None
    handoff = prefix.has_calls and source.can_handoff and prefix.can_handoff
    if not handoff and (source.unsafe_reason or not prefix.eligible):
        trace_event(
            stage="provider",
            event="provider.recovery.ineligible",
            source="provider",
            request_id=execution.request_id,
            reason=source.unsafe_reason or prefix.unsafe_reason,
        )
        return None
    if handoff:
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
    if not (prefix.text or prefix.thinking):
        return None
    replacement = continuation_body(body, protocol, prefix.text, prefix.thinking)
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
