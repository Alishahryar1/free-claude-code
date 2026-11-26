"""Project native Google Parts directly to the requested client protocol."""

import json
from collections.abc import Mapping
from copy import deepcopy
from uuid import uuid4

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.gemini.models import Part
from free_claude_code.core.gemini.response import conversion_failure, gemini_frame
from free_claude_code.core.history_replay import (
    ContinuationReplay,
    ReplayOrigin,
    ReplayRecord,
    encode_gemini_replay,
    encode_replay,
)
from free_claude_code.core.json_types import (
    JsonObject,
    JsonValue,
    json_array,
    json_object,
    json_objects,
    json_text,
)
from free_claude_code.core.recovery_limits import RECOVERY_RETENTION_BYTES
from free_claude_code.providers.openai_chat import (
    ChatStreamOutput,
    ChatStreamUsage,
    CompletedToolCall,
)


def _usage_count(value: JsonValue) -> int:
    if not isinstance(value, (int, float, str)):
        raise TypeError("Expected a number or numeric string for a Gemini token count")
    return int(value)


class GeminiStream:
    def __init__(
        self,
        *,
        model: str,
        origin: ReplayOrigin,
        output: ChatStreamOutput | None,
        tool_names: frozenset[str],
        tool_defaults: Mapping[str, JsonValue],
        include_thoughts: bool,
        continuation_context: tuple[JsonObject, ...] | None = None,
    ) -> None:
        self._model = model
        self._origin = origin
        self._output = output
        self._tools = tool_names
        self._tool_defaults = tool_defaults
        self._include_thoughts = include_thoughts
        self._continuation_context = deepcopy(continuation_context)
        self._terminal: JsonObject | None = None
        self._parts: list[JsonObject] = []
        self._retained_bytes = (
            len(json.dumps(continuation_context, ensure_ascii=False).encode("utf-8"))
            if continuation_context is not None
            else 0
        )
        self._started = False
        self._reason: str | int | None = None
        self._usage: JsonObject = {}
        self._tool_index = 0
        self.completed = False
        if output is not None:
            output.replay_origin = origin

    @property
    def normal_stop_seen(self) -> bool:
        return self._reason is not None

    def feed(self, payload: JsonObject) -> list[str]:
        if self.completed:
            raise conversion_failure("Gemini emitted content after completion")
        if payload.get("error"):
            error = payload["error"]
            if not isinstance(error, dict):
                raise conversion_failure("Gemini returned a malformed error")
            status = error.get("code", 502)
            if (
                not isinstance(status, int)
                or isinstance(status, bool)
                or not 400 <= status <= 599
            ):
                status = 502
            raise ExecutionFailure(
                kind={
                    400: FailureKind.INVALID_REQUEST,
                    401: FailureKind.AUTHENTICATION,
                    403: FailureKind.PERMISSION,
                    429: FailureKind.RATE_LIMIT,
                }.get(status, FailureKind.UPSTREAM),
                status_code=status,
                message=str(error.get("message", "Gemini generation failed")),
                retryable=status in {408, 429} or status >= 500,
            )
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) > 1:
            raise conversion_failure("Gemini returned an unexpected candidate count")
        if self._reason is not None and candidates:
            raise conversion_failure("Gemini emitted candidates after finishReason")
        events: list[str] = []
        if self._output is not None and not self._started:
            events.extend(self._output.start_events())
        self._started = True
        if isinstance(payload.get("usageMetadata"), dict):
            self._usage.update(payload["usageMetadata"])
        for candidate in json_objects(candidates):
            if candidate.get("index", 0) != 0:
                raise conversion_failure(
                    "Gemini returned an unexpected candidate index"
                )
            if reason := candidate.get("finishReason"):
                if isinstance(reason, float) and reason.is_integer():
                    reason = int(reason)
                if not isinstance(reason, str) and type(reason) is not int:
                    raise conversion_failure("Gemini returned an invalid finish reason")
                self._reason = reason
            if (
                self._output is not None or self._continuation_context is not None
            ) and any(
                candidate.get(key) for key in ("groundingMetadata", "citationMetadata")
            ):
                raise conversion_failure(
                    "This client protocol cannot represent Gemini grounding or citations"
                )
            for raw in json_objects(
                json_object(candidate.get("content", {})).get("parts", [])
            ):
                if self._output is None and self._continuation_context is None:
                    if "functionCall" in raw:
                        try:
                            Part.model_validate(raw, extra="forbid")
                        except ValueError as error:
                            raise conversion_failure(
                                "Gemini returned an unsupported partial function call"
                            ) from error
                    continue
                try:
                    part = Part.model_validate(raw, extra="forbid")
                except ValueError as error:
                    raise conversion_failure(
                        "Gemini returned unsupported native content"
                    ) from error
                saved = deepcopy(raw)
                self._retained_bytes += len(
                    json.dumps(saved, ensure_ascii=False).encode("utf-8")
                )
                if self._retained_bytes > RECOVERY_RETENTION_BYTES:
                    raise conversion_failure(
                        "Gemini native state exceeded the replay retention limit"
                    )
                if (
                    self._parts
                    and set(saved) <= {"text", "thought"}
                    and set(self._parts[-1]) <= {"text", "thought"}
                    and isinstance(saved.get("text"), str)
                    and isinstance(self._parts[-1].get("text"), str)
                    and bool(saved.get("thought"))
                    == bool(self._parts[-1].get("thought"))
                ):
                    self._parts[-1]["text"] = json_text(
                        self._parts[-1]["text"]
                    ) + json_text(saved["text"])
                else:
                    self._parts.append(saved)
                if self._output is None:
                    continue
                if part.text is not None:
                    if not part.text:
                        continue
                    if part.thought and self._include_thoughts:
                        events.extend(self._output.ensure_reasoning_block())
                        events.append(
                            self._output.emit_reasoning_delta(part.text, summary=True)
                        )
                    elif not part.thought:
                        events.extend(self._output.ensure_text_block())
                        events.append(self._output.emit_text_delta(part.text))
                elif part.function_call is not None:
                    call = part.function_call
                    if call.name not in self._tools:
                        raise conversion_failure(
                            f"Gemini called undeclared function {call.name}"
                        )
                    arguments = json.dumps(
                        call.args,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    events.extend(self._output.close_content_blocks())
                    events.extend(
                        self._output.emit_complete_tool(
                            self._tool_index,
                            CompletedToolCall(
                                call.id or f"call_{uuid4().hex}", call.name, arguments
                            ),
                        )
                    )
                    self._tool_index += 1
                elif part.inline_data is not None or part.function_response is not None:
                    raise conversion_failure("Gemini returned unsupported model output")
        if not candidates and json_object(payload.get("promptFeedback", {})).get(
            "blockReason"
        ):
            raise conversion_failure("Gemini blocked the input prompt")
        if self._output is None:
            result = deepcopy(payload)
            result["modelVersion"] = self._model
            if self._continuation_context is not None:
                for candidate in json_objects(result.get("candidates", [])):
                    for part in json_objects(
                        json_object(candidate.get("content", {})).get("parts", [])
                    ):
                        part.pop("thoughtSignature", None)
                    content = candidate.get("content")
                    if isinstance(content, dict):
                        content.setdefault("role", "model")
                        content["parts"] = [
                            part
                            for part in json_objects(content.get("parts", []))
                            if any(
                                key in part
                                for key in (
                                    "text",
                                    "functionCall",
                                    "inlineData",
                                    "functionResponse",
                                )
                            )
                        ]
                        if not content["parts"]:
                            candidate.pop("content")
                    if candidate.get("finishReason"):
                        self._terminal = deepcopy(result)
                        if not any(
                            part.get("thoughtSignature") for part in self._parts
                        ):
                            return []
                        terminal_candidate = json_object(
                            json_array(self._terminal["candidates"])[0]
                        )
                        terminal_candidate.pop("content", None)
                        candidate.pop("finishReason")
                if not any(
                    json_object(candidate.get("content", {})).get("parts")
                    for candidate in json_objects(result.get("candidates", []))
                ):
                    return []
            return [gemini_frame(result)]
        return [event for event in events if event]

    def finish(self) -> list[str]:
        if self._reason is None:
            raise conversion_failure("Gemini stream ended without finishReason")
        self.completed = True
        if self._output is None and self._continuation_context is None:
            return []
        if self._reason not in {
            "STOP",
            "MAX_TOKENS",
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
        }:
            raise conversion_failure(f"Gemini generation failed with {self._reason}")
        events: list[str] = []
        if any(part.get("thoughtSignature") for part in self._parts):
            carrier = (encode_gemini_replay if self._output is None else encode_replay)(
                ReplayRecord(
                    self._origin,
                    {"role": "model", "parts": self._parts},
                    tool_defaults={
                        name: self._tool_defaults[name]
                        for part in self._parts
                        if isinstance(
                            name := json_object(part.get("functionCall", {})).get(
                                "name"
                            ),
                            str,
                        )
                        and name in self._tool_defaults
                        and self._tool_defaults[name]
                    }
                    or None,
                    continuation=ContinuationReplay(self._continuation_context)
                    if self._continuation_context is not None
                    else None,
                )
            )
            if self._output is None:
                assert self._terminal is not None
                terminal_candidate = json_object(
                    json_array(self._terminal["candidates"])[0]
                )
                terminal_candidate["content"] = {
                    "role": "model",
                    "parts": [{"thoughtSignature": carrier}],
                }
            else:
                events.extend(self._output.emit_opaque_reasoning(carrier))
        if self._output is None:
            assert self._terminal is not None
            self._terminal["usageMetadata"] = self._usage
            return [*events, gemini_frame(self._terminal)]
        events.extend(self._output.close_content_blocks())
        usage = self._usage
        input_tokens = _usage_count(
            usage.get("promptTokenCount", self._output.input_tokens)
        )
        reasoning_tokens = _usage_count(usage.get("thoughtsTokenCount", 0))
        output_tokens = (
            _usage_count(usage.get("candidatesTokenCount", 0)) + reasoning_tokens
        )
        reason = (
            "max_tokens"
            if self._reason == "MAX_TOKENS"
            else "refusal"
            if self._reason != "STOP"
            else "tool_use"
            if self._tool_index
            else "end_turn"
        )
        events.extend(
            self._output.finish_success(
                stop_reason=reason,
                usage=ChatStreamUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=_usage_count(usage.get("cachedContentTokenCount", 0)),
                    reasoning_tokens=reasoning_tokens,
                    reported={
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "total_tokens": _usage_count(
                            usage.get("totalTokenCount", input_tokens + output_tokens)
                        ),
                    },
                    finish_reason="length" if self._reason == "MAX_TOKENS" else "stop",
                ),
            )
        )
        return events
