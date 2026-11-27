"""Shared Chat Completions transport and per-request stream execution."""

import asyncio
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
)
from dataclasses import dataclass
from functools import partial
from typing import Any, cast, override

import httpx2
from loguru import logger
from openai import AsyncOpenAI, Omit
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic import (
    ContentBlockToolUse,
    ContentType,
    ThinkTagParser,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.streaming import (
    map_stop_reason,
)
from free_claude_code.core.chat_completions.request import ChatRequest
from free_claude_code.core.chat_completions.response import NativeChatRelay
from free_claude_code.core.chat_completions.tool_calls import (
    ChatSourceOutput,
    ChatToolCallBuffer,
    CompletedChatBatch,
    CompletedChatCall,
    finalize_chat_call,
)
from free_claude_code.core.diagnostics import (
    exception_cause_types,
    redacted_exception_traceback,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.gemini import GeminiConversionError, GeminiRequest
from free_claude_code.core.gemini.replay import CallReplay, apply_call_replay
from free_claude_code.core.gemini.request import build_gemini_chat_request
from free_claude_code.core.history_replay import ReplayOrigin
from free_claude_code.core.json_types import (
    JsonObject,
    JsonTree,
    JsonValue,
    json_object,
    json_objects,
    json_text,
    normalize_empty_array_field,
)
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesChatRequest,
    ResponsesConversionError,
    build_responses_chat_request,
)
from free_claude_code.core.openai_tool_names import (
    OpenAIToolNameCodec,
    encode_openai_chat_tool_names,
)
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningControl,
    ReasoningPolicy,
)
from free_claude_code.core.stream_output import StreamOutput, completed_output
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.core.tool_call_scope import ToolCallScope, has_tool_selection
from free_claude_code.core.trace import provider_chat_body_snapshot, trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.attempt_loop import (
    AttemptContext,
    AttemptPlan,
    FailureReporting,
    TransportAttempt,
    run_attempts,
)
from free_claude_code.providers.canonical_failures import (
    context_window_exceeded_provider_failure,
    is_context_window_finish_reason,
)
from free_claude_code.providers.continuation import (
    ContinuationRequest,
)
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.endpoint_types import EndpointContext
from free_claude_code.providers.history_negotiation import (
    HistoryModeCache,
    RequestHistory,
    resolve_history_overrides,
)
from free_claude_code.providers.history_replay import (
    replay_origin,
    validate_history,
)
from free_claude_code.providers.http import (
    ProviderAttemptScope,
    maybe_await_aclose,
)
from free_claude_code.providers.openai_client import OpenAIRequestClient
from free_claude_code.providers.openai_stream import OpenAIStreamAdapter
from free_claude_code.providers.reasoning_compatibility import (
    ReasoningCorrection,
    prepare_messages_reasoning,
)
from free_claude_code.providers.request_correction import (
    CorrectionStep,
    Proposal,
    Rejection,
    RequestCorrector,
)
from free_claude_code.providers.stream_recovery import (
    TruncatedProviderStreamError,
)

from .behavior import OpenAIChatBehavior
from .gemini_output import GeminiChatStreamOutput
from .media_support import TextOnlyModels
from .native_output import NativeChatStreamOutput
from .output_cap import OutputCapMemory
from .profiles import OpenAIChatProfile
from .reasoning_details import StructuredReasoningStream
from .request_policy import (
    apply_openai_chat_body_policy,
)
from .stream_output import (
    ORDINARY_REASONING_SOURCE,
    AnthropicChatStreamOutput,
    ChatStreamOutput,
    ChatStreamUsage,
    CompletedToolCall,
    ResponsesChatStreamOutput,
)
from .usage import (
    clone_without_stream_usage,
    is_stream_usage_rejection,
    nested_usage_int,
    request_stream_usage,
    usage_int,
)

OpenAIAsyncCredentialProvider = Callable[[], Awaitable[str]]
PreparedBodyValidator = Callable[[Mapping[str, Any]], Awaitable[None]]
_ExtraReasoningEvents = Callable[[JsonObject, ChatStreamOutput], Iterator[str]]
_ChatOutputFactory = Callable[[], ChatStreamOutput]


def _iter_visible_text_events(
    output: ChatStreamOutput,
    text: str,
) -> Iterator[str]:
    yield from output.ensure_text_block()
    yield output.emit_text_delta(text)


@dataclass(frozen=True, slots=True)
class _OpenAIChatCompletion:
    finish_reason: str | None
    output_tokens: int
    input_tokens: int
    provider_input_tokens: int | None


def _reserved_anthropic_tool_ids(request: MessagesRequest) -> frozenset[str]:
    """Return prior tool-use IDs that generated output must not reuse."""
    return frozenset(
        block.id
        for message in request.messages
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, ContentBlockToolUse) and block.id.strip()
    )


type _SDKRequestBody = dict[str, JsonTree[Omit]]


def _sdk_create_body(create_body: Mapping[str, JsonTree[Omit]]) -> _SDKRequestBody:
    """Move fields the SDK does not declare into ``extra_body``."""
    sdk_fields = CompletionCreateParamsStreaming.__annotations__
    passthrough = {"extra_body", "extra_headers"}
    extras = {
        key: value
        for key, value in create_body.items()
        if key not in sdk_fields and key not in passthrough
    }
    sent: _SDKRequestBody = {
        key: value
        for key, value in create_body.items()
        if key in sdk_fields or key in passthrough
    }
    if extras:
        sent["extra_body"] = {**extras, **json_object(sent.get("extra_body", {}))}
    return sent


def _correction_steps(
    behavior: OpenAIChatBehavior,
    output_caps: OutputCapMemory,
    text_only: TextOnlyModels,
) -> tuple[CorrectionStep, ...]:
    """Chat's own corrections, in the order they are tried."""

    def without_stream_usage(rejection: Rejection) -> Proposal | None:
        if not is_stream_usage_rejection(rejection.error):
            return None
        corrected = clone_without_stream_usage(rejection.body)
        if corrected is None:
            return None
        return Proposal(corrected, "dropped stream_options.include_usage")

    def without_media(rejection: Rejection) -> Proposal | None:
        if not behavior.media_rejected(rejection.error):
            return None
        return text_only.correct(rejection)

    return (
        CorrectionStep("output_cap", output_caps.correct, once=False),
        CorrectionStep("stream_usage", without_stream_usage),
        CorrectionStep("media", without_media),
        CorrectionStep(
            "provider_specific",
            lambda rejection: behavior.retry_request_body(
                rejection.error, rejection.body
            ),
        ),
        CorrectionStep(
            "provider_late",
            lambda rejection: behavior.retry_after_standard_corrections(
                rejection.error, rejection.body
            ),
        ),
    )


class _ChatStreamPresenter:
    """Own one discardable OpenAI-chat replay epoch."""

    def __init__(
        self,
        *,
        output: ChatStreamOutput,
        profile: OpenAIChatProfile,
        provider_name: str,
        output_reasoning: bool,
        tool_names: OpenAIToolNameCodec,
        reserved_tool_ids: frozenset[str],
        record_extra_content: Callable[[str, dict[str, Any]], None],
        extra_reasoning_events: _ExtraReasoningEvents,
    ) -> None:
        self._output = output
        self._profile = profile
        self._provider_name = provider_name
        self._output_reasoning = output_reasoning
        self._tool_names = tool_names
        self._tool_scope: ToolCallScope | None = None
        self._tool_buffer = ChatToolCallBuffer()
        self._reserved_tool_ids = set(reserved_tool_ids)
        self._record_extra_content = record_extra_content
        self._extra_reasoning_events = extra_reasoning_events
        self._think_parser = ThinkTagParser()
        self._structured_reasoning = StructuredReasoningStream(
            structured_details=profile.structured_reasoning_details
        )
        self._output.reasoning_replay = self._structured_reasoning
        self._finish_reason: str | None = None
        self._refused = False
        self._raw_stop_seen = False
        self._usage_info: object = None
        self._tool_argument_aliases: dict[str, dict[str, str]] = {}
        self._started = False
        self._tools_bound = False
        self._upstream_finished = False
        self._completion: _OpenAIChatCompletion | None = None
        self._completed = False

    @property
    def output(self) -> ChatStreamOutput:
        return self._output

    @property
    def usage_info(self) -> object:
        return self._usage_info

    @property
    def completion(self) -> _OpenAIChatCompletion:
        if self._completion is None:
            raise RuntimeError("stream completion has not been prepared")
        return self._completion

    @property
    def generated_output(self) -> bool:
        return self._output.committed_output

    @property
    def normal_stop_seen(self) -> bool:
        return self._raw_stop_seen or self._finish_reason is not None

    def observe_upstream_chunk(self, chunk: ChatCompletionChunk) -> None:
        if chunk.choices and chunk.choices[0].finish_reason is not None:
            self._raw_stop_seen = True

    def start_events(self) -> Iterator[str]:
        if self._started:
            return
        self._started = True
        yield from self._output.start_events()

    def bind_tools(
        self, aliases: dict[str, dict[str, str]], scope: ToolCallScope | None
    ) -> None:
        if self._tools_bound:
            raise RuntimeError("tool scope and argument aliases already bound")
        self._tools_bound = True
        self._tool_argument_aliases = aliases
        self._tool_scope = scope

    def feed(self, chunk: ChatCompletionChunk) -> Iterator[StreamOutput]:
        if not self._started or self._upstream_finished:
            raise RuntimeError("stream presenter is not accepting chunks")

        if chunk.choices and chunk.choices[0].finish_reason is not None:
            self._finish_reason = chunk.choices[0].finish_reason
        payload: JsonObject = chunk.model_dump(mode="json", exclude_unset=True)
        yield from self._present(self._tool_buffer.feed(payload))

    def abandon(self) -> Iterator[StreamOutput]:
        yield from self._present(self._tool_buffer.abandon())

    def _present(
        self, items: Iterable[ChatSourceOutput | CompletedChatBatch]
    ) -> Iterator[StreamOutput]:
        for item in items:
            if isinstance(item, CompletedChatCall):
                call = finalize_chat_call(
                    item,
                    codec=self._tool_names,
                    aliases=self._tool_argument_aliases,
                    reserved_ids=self._reserved_tool_ids,
                )
                function = call.body.get("function")
                if function is None:
                    raise ExecutionFailure(
                        FailureKind.UPSTREAM,
                        502,
                        "Custom Chat calls cannot be represented in this protocol",
                        False,
                    )
                function = json_object(function)
                tool_id = call.body["id"]
                name = function["name"]
                arguments = function["arguments"]
                if (
                    not isinstance(tool_id, str)
                    or not isinstance(name, str)
                    or not isinstance(arguments, str)
                ):
                    raise ExecutionFailure(
                        FailureKind.UPSTREAM,
                        502,
                        "Invalid completed Chat tool call",
                        False,
                    )
                if self._tool_scope is not None:
                    canonical = self._output.canonical_function_name(name)
                    self._tool_scope.require(
                        "function", self._tool_names.encode(canonical)
                    )
                raw_extra = call.body.get("extra_content")
                extra = raw_extra if isinstance(raw_extra, dict) else None
                if extra:
                    self._record_extra_content(tool_id, extra)
                frames = [
                    *self._output.close_content_blocks(),
                    *self._output.emit_complete_tool(
                        call.index,
                        CompletedToolCall(
                            tool_id,
                            name,
                            arguments,
                            extra,
                        ),
                    ),
                ]
                yield completed_output(frames)
            else:
                assert isinstance(item, dict)
                frames = tuple(self._feed_frame(item))
                if frames:
                    yield StreamOutput(frames)

    def _feed_frame(self, chunk: JsonObject) -> Iterator[str]:

        chunk_usage = chunk.get("usage")
        if chunk_usage is not None:
            self._usage_info = chunk_usage

        choices = json_objects(chunk.get("choices") or [])
        choice = next(choices, None)
        if choice is None:
            return

        raw_delta = choice.get("delta")
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            self._finish_reason = finish_reason
        if raw_delta is None:
            return
        delta = json_object(raw_delta)

        if choice.get("finish_reason"):
            logger.debug(
                "{} finish_reason: {}",
                self._provider_name,
                self._finish_reason,
            )

        field, reasoning = self._profile.native_reasoning_delta(delta)
        yield from self._structured_reasoning.events(
            delta,
            self._output,
            native_reasoning=reasoning if self._output_reasoning else None,
            native_field=field,
            output_reasoning=self._output_reasoning,
        )

        yield from self._extra_reasoning_events(delta, self._output)

        refusal = delta.get("refusal")
        if isinstance(self._output, AnthropicChatStreamOutput) and refusal:
            self._refused = True
            yield from _iter_visible_text_events(self._output, json_text(refusal))
        content = delta.get("content")
        if content:
            for part in self._think_parser.feed(json_text(content)):
                if part.type == ContentType.THINKING:
                    if not self._output_reasoning:
                        continue
                    yield from self._output.ensure_reasoning_block(
                        source=ORDINARY_REASONING_SOURCE
                    )
                    yield self._output.emit_reasoning_delta(
                        part.content, source=ORDINARY_REASONING_SOURCE
                    )
                else:
                    yield from _iter_visible_text_events(self._output, part.content)

    def finish_upstream(self) -> Iterator[str]:
        if self._upstream_finished:
            return
        if self._finish_reason is None:
            raise TruncatedProviderStreamError(
                "Provider stream ended without finish_reason."
            )
        if is_context_window_finish_reason(self._finish_reason):
            raise context_window_exceeded_provider_failure()
        remaining = self._think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                if self._output_reasoning:
                    yield from self._output.ensure_reasoning_block(
                        source=ORDINARY_REASONING_SOURCE
                    )
                    yield self._output.emit_reasoning_delta(
                        remaining.content, source=ORDINARY_REASONING_SOURCE
                    )
            else:
                yield from _iter_visible_text_events(self._output, remaining.content)

        yield from self._output.flush_reasoning_replay()
        self._upstream_finished = True

    def prepare_completion(self, *, pad_empty: bool) -> Iterator[str]:
        if not self._upstream_finished or self._completion is not None:
            raise RuntimeError("stream completion cannot be prepared")

        has_emitted_tool = self._output.has_emitted_tool_block()
        has_content_blocks = self._output.has_content_block()
        if pad_empty and (
            not has_content_blocks
            or (
                not has_emitted_tool
                and not self._output.accumulated_text.strip()
                and self._output.accumulated_reasoning.strip()
            )
        ):
            yield from self._output.ensure_text_block()
            yield self._output.emit_text_delta(" ")

        yield from self._output.close_all_blocks()

        completion = usage_int(self._usage_info, "completion_tokens")
        output_tokens = (
            completion
            if isinstance(completion, int)
            else self._output.estimate_output_tokens()
        )
        provider_input = usage_int(self._usage_info, "prompt_tokens")
        input_tokens = (
            provider_input if provider_input is not None else self._output.input_tokens
        )
        self._completion = _OpenAIChatCompletion(
            finish_reason=self._finish_reason,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            provider_input_tokens=provider_input,
        )

    def terminal_events(self, *, usage: ChatStreamUsage) -> Iterator[str]:
        if self._completed:
            return
        completion = self.completion
        stop_reason = map_stop_reason(completion.finish_reason)
        if self._refused and stop_reason != "max_tokens":
            stop_reason = "refusal"
        yield from self._output.finish_success(
            stop_reason=stop_reason,
            usage=usage,
        )
        self._completed = True


def _native_chunk_payload(chunk: ChatCompletionChunk) -> JsonObject:
    """Normalize empty provider arrays at the native Chat SDK boundary."""
    payload: JsonObject = chunk.model_dump(mode="json", exclude_unset=True)
    normalize_empty_array_field(payload, "choices")
    if isinstance(choices := payload.get("choices"), list):
        for choice in choices:
            if not isinstance(choice, dict) or not isinstance(
                delta := choice.get("delta"), dict
            ):
                continue
            for field in ("tool_calls", "reasoning_details"):
                normalize_empty_array_field(delta, field)
    return payload


class _NativeChatStreamPresenter(_ChatStreamPresenter):
    """Use the existing attempt owner without translating native Chat chunks."""

    def __init__(self, *, choices: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert isinstance(self._output, NativeChatStreamOutput)
        self._output.reasoning_replay = None
        self._relay = NativeChatRelay(self._output.chat.model, choices=choices)
        self._tool_buffer = ChatToolCallBuffer(native=True)
        self._generated = False

    @property
    @override
    def generated_output(self) -> bool:
        return self._generated

    @property
    @override
    def normal_stop_seen(self) -> bool:
        return self._relay.normal_stop_seen

    @override
    def observe_upstream_chunk(self, chunk: ChatCompletionChunk) -> None:
        pass

    @override
    def start_events(self) -> Iterator[str]:
        self._started = True
        return iter(())

    @override
    def feed(self, chunk: ChatCompletionChunk) -> Iterator[StreamOutput]:
        payload = _native_chunk_payload(chunk)
        if chunk.usage is not None:
            self._usage_info = chunk.usage
        for choice in json_objects(payload.get("choices", [])):
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                self._finish_reason = finish_reason
            delta = json_object(choice.get("delta") or {})
            self._generated |= any(
                value for key, value in delta.items() if key != "role"
            )
        yield from self._present(self._tool_buffer.feed(payload))

    @override
    def _present(
        self, items: Iterable[ChatSourceOutput | CompletedChatBatch]
    ) -> Iterator[StreamOutput]:
        for item in items:
            if isinstance(item, CompletedChatBatch):
                yield completed_output(self._relay.feed(frame) for frame in item.frames)
            else:
                assert isinstance(item, dict)
                yield StreamOutput((self._relay.feed(item),))

    @override
    def finish_upstream(self) -> Iterator[str]:
        if not self._relay.normal_stop_seen:
            raise TruncatedProviderStreamError(
                "Native Chat stream ended before all choices completed"
            )
        self._upstream_finished = True
        return iter(())

    @override
    def prepare_completion(self, *, pad_empty: bool) -> Iterator[str]:
        provider_input = usage_int(self._usage_info, "prompt_tokens")
        self._completion = _OpenAIChatCompletion(
            self._finish_reason,
            usage_int(self._usage_info, "completion_tokens") or 0,
            provider_input if provider_input is not None else self._output.input_tokens,
            provider_input,
        )
        return iter(())

    @override
    def terminal_events(self, *, usage: ChatStreamUsage) -> Iterator[str]:
        yield from self._relay.finish()


class OpenAIChatTransport:
    """Execute Chat requests while borrowing provider-owned HTTP resources."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        admission: ProviderAdmissionController,
        history_cache: HistoryModeCache | None = None,
        behavior: OpenAIChatBehavior,
        read_timeout_s: float,
        log_raw_sse_events: bool,
        log_api_error_tracebacks: bool,
        endpoint_transport: httpx2.AsyncBaseTransport | None = None,
        api_key_provider: OpenAIAsyncCredentialProvider | None = None,
        validate_prepared_body: PreparedBodyValidator | None = None,
    ) -> None:
        self._client = client
        self._admission = admission
        self._history_cache = (
            history_cache if history_cache is not None else HistoryModeCache()
        )
        self._behavior = behavior
        self._profile = behavior.profile
        self._provider_name = self._profile.provider_name
        self._read_timeout_s = read_timeout_s
        self._log_raw_sse_events = log_raw_sse_events
        self._log_api_error_tracebacks = log_api_error_tracebacks
        self._endpoint_transport = endpoint_transport
        self._api_key_provider = api_key_provider
        self._validate_prepared_body = validate_prepared_body
        self._output_caps = OutputCapMemory()
        self._text_only = TextOnlyModels()
        self._correction_steps = _correction_steps(
            behavior, self._output_caps, self._text_only
        )

    def _log_stream_transport_error(
        self,
        tag: str,
        req_tag: str,
        error: Exception,
        *,
        request_id: str | None = None,
    ) -> None:
        """Log streaming transport failures (metadata-only unless verbose is enabled)."""
        response: object = getattr(error, "response", None)
        http_status: object = (
            getattr(response, "status_code", None) if response is not None else None
        )
        cause_types = exception_cause_types(error)
        trace_event(
            stage="provider",
            event="provider.response.transport_error",
            source="provider",
            provider=tag,
            request_id=request_id,
            exc_type=type(error).__name__,
            http_status=http_status,
            cause_types=cause_types,
        )

        if self._log_api_error_tracebacks:
            logger.error(
                "{}_ERROR:{} exc_type={}\n{}",
                tag,
                req_tag,
                type(error).__name__,
                redacted_exception_traceback(error),
            )
            return
        logger.error(
            "{}_ERROR:{} exc_type={} http_status={} cause_types={}",
            tag,
            req_tag,
            type(error).__name__,
            http_status,
            ",".join(cause_types) if cause_types else None,
        )

    def _build_request_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
    ) -> JsonObject:
        """Build a provider request from the immutable profile."""
        request, reasoning = self._prepare_messages_reasoning(
            request, reasoning, model_info
        )
        return self._behavior.build_messages_body(request, reasoning=reasoning)

    def _prepare_messages_reasoning(
        self,
        request: MessagesRequest,
        reasoning: ReasoningPolicy,
        model_info: ProviderModelInfo | None,
    ) -> tuple[MessagesRequest, ReasoningPolicy]:
        return prepare_messages_reasoning(
            request,
            reasoning,
            model_info=model_info,
            can_disable=bool(self._behavior.reasoning_off_fields),
            normal_max_tokens=self._behavior.normal_max_tokens,
        )

    def _build_responses_request_body(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> ResponsesChatRequest:
        """Build a Chat body directly from Responses ingress."""
        validate_history(request.model_dump(mode="json"), "responses")
        try:
            translated = build_responses_chat_request(
                request,
                reasoning_replay=self._profile.request_policy.reasoning_replay,
                structured_reasoning_details=(
                    self._profile.structured_reasoning_details
                ),
            )
        except ResponsesConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc
        body = translated.body
        apply_openai_chat_body_policy(body, self._profile.request_policy)
        self._profile.apply_reasoning_to_body(body, reasoning)
        body = self._behavior.finalize_chat_body(body, reasoning=reasoning)
        encode_openai_chat_tool_names(body, translated.tool_names)
        return ResponsesChatRequest(
            body=body,
            tool_names=translated.tool_names,
            reserved_tool_ids=translated.reserved_tool_ids,
            tool_adapter=translated.tool_adapter,
        )

    async def _prepare_create(
        self,
        body: dict[str, Any],
        operation_kind: ProviderOperationKind,
        *,
        history: RequestHistory,
        endpoint: RequestEndpoint | None = None,
        request_client: OpenAIRequestClient | None = None,
        extra_headers: Mapping[str, str] | None = None,
        on_origin: Callable[[ReplayOrigin], None] | None = None,
        call_replay: tuple[CallReplay, ...] = (),
        native_chat: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any], AsyncOpenAI]:
        """Return the intended body, the body to send, and the client to send it."""
        body = self._output_caps.clamp(body)
        body, replaced = self._text_only.strip(body)
        if replaced:
            logger.warning(
                "{}_REQUEST: model {} rejected media earlier; replaced {} media "
                "part(s) with a note. The model will not see them.",
                self._provider_name,
                body.get("model"),
                replaced,
            )
        create_body = resolve_history_overrides(
            body if native_chat else self._behavior.prepare_create_body(body)
        )
        client = self._client
        if endpoint is not None:
            assert request_client is not None
            client = request_client.for_endpoint(self._client, await endpoint.resolve())
        elif self._api_key_provider is not None:
            # A borrowed view pins the same credential for provenance and
            # inference, without owning or closing the provider's HTTP pool.
            client = self._client.with_options(api_key=await self._api_key_provider())
        sdk_headers: dict[str, JsonTree[Omit]] | None = None
        if extra_headers or endpoint is not None:
            sdk_headers = {
                **json_object(create_body.get("extra_headers") or {}),
                **(extra_headers or {}),
                **(
                    request_client.openai_headers()
                    if request_client is not None
                    else {}
                ),
            }
        origin = replay_origin(
            self._provider_name,
            "chat",
            str(create_body["model"]),
            client=client,
            endpoint=endpoint.snapshot if endpoint is not None else None,
        )
        if on_origin is not None:
            on_origin(origin)
        prepared = history.prepare(
            create_body,
            origin,
            self._history_cache,
            reasoning_field=self._profile.request_policy.reasoning_replay.value,
            structured_details=self._profile.structured_reasoning_details,
            native_chat=native_chat,
            continuation=operation_kind is ProviderOperationKind.CONTINUATION,
            finalize=lambda projected, origin=origin: (
                self._behavior.prepare_history_body(
                    cast(
                        dict[str, Any],
                        apply_call_replay(projected, origin, call_replay),
                    )
                    if call_replay
                    else projected,
                    origin,
                )
            ),
        )
        sdk_body: _SDKRequestBody = dict(prepared)
        if sdk_headers is not None:
            sdk_body["extra_headers"] = sdk_headers
        return body, sdk_body, client

    async def _open_stream(
        self,
        scope: ProviderAttemptScope,
        client: AsyncOpenAI,
        create_body: dict[str, Any],
        body: dict[str, Any],
        *,
        history: RequestHistory,
        on_event: Callable[[ChatCompletionChunk], None] | None = None,
        native_chat: bool = False,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Open one streaming chat completion owned by an admitted attempt."""

        def observed(event: ChatCompletionChunk) -> None:
            if event.choices or event.usage is not None:
                history.accepted()
            if on_event is not None:
                on_event(event)

        stream = scope.retain(
            OpenAIStreamAdapter(
                await client.chat.completions.create(**create_body, stream=True),
                read_timeout_s=self._read_timeout_s,
                on_event=observed,
                transform_event=self._behavior.normalize_chunk,
            )
        )
        if native_chat:
            return stream
        return scope.adopt(self._behavior.normalize_stream(stream, body))

    def _corrector(
        self, reasoning: ReasoningCorrection | None = None
    ) -> RequestCorrector:
        return RequestCorrector(
            "chat", reasoning=reasoning, steps=self._correction_steps
        )

    def stream_chat(
        self,
        request: ChatRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
        endpoint_context: EndpointContext | None = None,
        extra_headers: Mapping[str, str] | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[StreamOutput]:
        raw = request.body()
        raw.pop("stream", None)
        if not request.stream:
            raw["stream_options"] = {
                **(request.stream_options or {}),
                "include_usage": True,
            }
        # Ingress is JSON wire data, not SDK kwargs. A literal extra_body or
        # extra_headers extension must stay in the JSON rather than overriding
        # request fields or HTTP headers inside the SDK.
        sdk_fields = CompletionCreateParamsStreaming.__annotations__
        body = {key: value for key, value in raw.items() if key in sdk_fields}
        extensions = {key: value for key, value in raw.items() if key not in sdk_fields}
        if extensions:
            body["extra_body"] = extensions
        # Explicit routing overrides have priority. Otherwise native controls stay
        # exactly as supplied, including vendor extensions and explicit nulls.
        if reasoning != DEFAULT_REASONING_POLICY:
            self._profile.apply_reasoning_to_body(body, reasoning)
        validate_history(body, "chat")
        public_model = response_model or request.model
        options = body.get("stream_options") or {}
        return _OpenAIChatStreamRunner(
            self,
            body=body,
            tool_names=OpenAIToolNameCodec.from_names(()),
            reserved_tool_ids=frozenset(),
            output_factory=lambda: NativeChatStreamOutput(
                model=public_model,
                input_tokens=input_tokens,
                include_usage=isinstance(options, dict)
                and bool(options.get("include_usage")),
            ),
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=public_model,
            reasoning=reasoning,
            endpoint_context=endpoint_context,
            extra_headers=extra_headers,
            continuation=continuation,
            native_chat=True,
        ).run()

    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
        endpoint_context: EndpointContext | None = None,
        extra_headers: Mapping[str, str] | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[StreamOutput]:
        """Stream response in Anthropic SSE format."""
        prepared_request, wire_reasoning = self._prepare_messages_reasoning(
            request, reasoning, model_info
        )
        body = self._build_request_body(prepared_request, reasoning=wire_reasoning)
        off_fields = self._behavior.reasoning_off_fields
        correction = (
            ReasoningCorrection(
                off_fields,
                self._profile.request_policy.max_tokens_field,
                self._behavior.normal_max_tokens,
                provider_rejection=self._behavior.reasoning_disable_rejected,
            )
            if reasoning.control is ReasoningControl.PREFER_OFF
            and wire_reasoning.control is ReasoningControl.OFF
            and off_fields
            else None
        )
        tool_names = OpenAIToolNameCodec.from_request(request)
        message_id = f"msg_{uuid.uuid4()}"
        runner = _OpenAIChatStreamRunner(
            self,
            body=body,
            tool_names=tool_names,
            reserved_tool_ids=_reserved_anthropic_tool_ids(request),
            constrain_tool_calls=has_tool_selection(request.tools, request.tool_choice),
            output_factory=lambda: AnthropicChatStreamOutput(
                message_id=message_id,
                model=request.model if response_model is None else response_model,
                input_tokens=input_tokens,
                log_raw_events=self._log_raw_sse_events,
            ),
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=(
                request.model if response_model is None else response_model
            ),
            reasoning=reasoning,
            endpoint_context=endpoint_context,
            extra_headers=extra_headers or {},
            reasoning_correction=correction,
            continuation=continuation,
        )
        return runner.run()

    def stream_gemini(
        self,
        request: GeminiRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
        endpoint_context: EndpointContext | None = None,
        extra_headers: Mapping[str, str] | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[StreamOutput]:
        try:
            prepared = build_gemini_chat_request(request)
        except GeminiConversionError as error:
            raise InvalidRequestError(str(error)) from error
        body = prepared.body
        body.pop("stream", None)
        policy = self._profile.request_policy
        if "max_tokens" not in body and policy.default_max_tokens is not None:
            body["max_tokens"] = policy.default_max_tokens
        apply_openai_chat_body_policy(body, policy)
        self._profile.apply_reasoning_to_body(body, reasoning)
        body = self._behavior.finalize_chat_body(body, reasoning=reasoning)
        public_model = request.model if response_model is None else response_model
        response_id = f"fcc_{uuid.uuid4().hex}"
        runner = _OpenAIChatStreamRunner(
            self,
            body=body,
            tool_names=prepared.tool_names,
            reserved_tool_ids=prepared.reserved_tool_ids,
            constrain_tool_calls=True,
            output_factory=lambda: GeminiChatStreamOutput(
                response_id=response_id,
                model=public_model,
                include_thoughts=request.include_thoughts,
                tool_names=prepared.declared_names,
                input_tokens=input_tokens,
            ),
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=public_model,
            reasoning=reasoning,
            endpoint_context=endpoint_context,
            extra_headers=extra_headers,
            continuation=continuation,
            call_replay=prepared.call_replay,
        )
        return runner.run()

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        endpoint_context: EndpointContext | None = None,
        extra_headers: Mapping[str, str] | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[StreamOutput]:
        """Stream a Chat upstream directly as OpenAI Responses SSE."""
        translated = self._build_responses_request_body(request, reasoning=reasoning)
        public_model = request.model if response_model is None else response_model
        runner = _OpenAIChatStreamRunner(
            self,
            body=translated.body,
            tool_names=translated.tool_names,
            reserved_tool_ids=translated.reserved_tool_ids,
            constrain_tool_calls=translated.tool_adapter.constrains_returned_calls,
            output_factory=lambda: ResponsesChatStreamOutput(
                translated.tool_adapter,
                input_tokens=input_tokens,
                response_model=public_model,
            ),
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=public_model,
            reasoning=reasoning,
            endpoint_context=endpoint_context,
            extra_headers=extra_headers or {},
            continuation=continuation,
        )
        return runner.run()


class _OpenAIChatStreamRunner:
    """Orchestrate one OpenAI-chat request and its recovery lifecycle."""

    def __init__(
        self,
        transport: OpenAIChatTransport,
        *,
        body: dict[str, Any],
        tool_names: OpenAIToolNameCodec,
        reserved_tool_ids: frozenset[str],
        output_factory: _ChatOutputFactory,
        constrain_tool_calls: bool = False,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        endpoint_context: EndpointContext | None = None,
        extra_headers: Mapping[str, str] | None = None,
        reasoning_correction: ReasoningCorrection | None = None,
        continuation: ContinuationSeed | None = None,
        call_replay: tuple[CallReplay, ...] = (),
        native_chat: bool = False,
    ) -> None:
        self._transport = transport
        self._native_chat = native_chat
        self._body = body
        self._continuation = continuation
        self._call_replay = call_replay
        self._tool_argument_aliases = transport._behavior.tool_argument_aliases(body)
        self._tool_names = tool_names
        self._reserved_tool_ids = reserved_tool_ids
        self._output_factory = output_factory
        self._constrain_tool_calls = constrain_tool_calls
        self._input_tokens = input_tokens
        self._request_id = request_id
        self._response_model = response_model
        self._reasoning = reasoning
        self._reasoning_correction = reasoning_correction
        self._extra_headers = dict(extra_headers or {})
        self._request_client = OpenAIRequestClient(transport._endpoint_transport)
        self._endpoint = (
            RequestEndpoint(endpoint_context) if endpoint_context is not None else None
        )

    async def run(self) -> AsyncIterator[StreamOutput]:
        """Convert the upstream OpenAI-chat stream into Anthropic SSE."""
        model: JsonValue = self._body["model"]
        extra_body = self._body.get("extra_body")
        # The SDK merges allowed extra_body fields over the top-level request.
        if isinstance(extra_body, dict) and isinstance(extra_body.get("model"), str):
            model = extra_body["model"]
        execution = self._transport._admission.start_execution(
            request_id=self._request_id, model=str(model)
        )
        provider_stream = self._run_execution(execution)
        try:
            async for event in provider_stream:
                yield event
        except asyncio.CancelledError:
            raise
        except Exception as error:
            execution.fail(error)
            raise
        else:
            execution.succeed()
        finally:
            try:
                await maybe_await_aclose(provider_stream)
            finally:
                try:
                    await self._request_client.aclose()
                finally:
                    execution.abandon()

    def _run_execution(
        self, execution: ProviderExecution
    ) -> AsyncIterator[StreamOutput]:
        """Run one provider execution while retaining transport-owned state."""
        tag = self._transport._provider_name
        req_tag = f" request_id={self._request_id}" if self._request_id else ""
        body = resolve_history_overrides(self._body)
        continuation_request = ContinuationRequest("chat")
        operation_kind = ProviderOperationKind.GENERATION
        if self._continuation is not None:
            body = continuation_request.build(body, self._continuation)
            operation_kind = ProviderOperationKind.CONTINUATION
        if not self._native_chat:
            request_stream_usage(body)
        messages = body.get("messages")
        tools = body.get("tools")
        return run_attempts(
            AttemptPlan(
                body=body,
                operation_kind=operation_kind,
                execution=execution,
                continuation=continuation_request,
                corrector=partial(
                    self._transport._corrector, self._reasoning_correction
                ),
                start=partial(_ChatAttempt, runner=self),
                reporting=FailureReporting(
                    read_timeout_s=self._transport._read_timeout_s,
                    failure_override=self._transport._behavior.failure_override,
                    trace_error_message=self._transport._log_api_error_tracebacks,
                    report_terminal_error=partial(
                        self._transport._log_stream_transport_error,
                        tag,
                        req_tag,
                        request_id=self._request_id,
                    ),
                ),
                provider_name=tag,
                transport="chat",
                gateway_model=self._response_model,
                refreshable=self._endpoint,
                sent_fields={
                    "message_count": len(messages) if isinstance(messages, list) else 0,
                    "tool_count": len(tools) if isinstance(tools, list) else 0,
                },
                sent_payload=lambda: {"body": provider_chat_body_snapshot(body)},
            )
        )

    def _new_stream_presenter(self, *, output_reasoning: bool) -> _ChatStreamPresenter:
        def extra_reasoning_events(
            delta: JsonObject, output: ChatStreamOutput
        ) -> Iterator[str]:
            yield from self._transport._behavior.extra_reasoning_events(
                delta,
                output,
                output_reasoning=output_reasoning,
            )

        output = self._output_factory()

        def record_extra_content(
            tool_call_id: str, extra_content: dict[str, Any]
        ) -> None:
            assert output.replay_origin is not None
            self._transport._behavior.record_tool_call_extra_content(
                tool_call_id, extra_content, origin=output.replay_origin
            )

        assembler_type = (
            partial(_NativeChatStreamPresenter, choices=int(self._body.get("n") or 1))
            if self._native_chat
            else _ChatStreamPresenter
        )
        return assembler_type(
            output=output,
            profile=self._transport._profile,
            provider_name=self._transport._provider_name,
            output_reasoning=output_reasoning,
            tool_names=self._tool_names,
            reserved_tool_ids=self._reserved_tool_ids,
            record_extra_content=record_extra_content,
            extra_reasoning_events=extra_reasoning_events,
        )


class _ChatAttempt(TransportAttempt):
    """One Chat Completions attempt."""

    def __init__(self, context: AttemptContext, *, runner: _OpenAIChatStreamRunner):
        super().__init__(context)
        self._runner = runner
        self._presenter = runner._new_stream_presenter(
            output_reasoning=runner._reasoning.output_enabled
        )
        self._client: AsyncOpenAI | None = None
        self._validating = False

    @property
    @override
    def normal_stop_seen(self) -> bool:
        return self._presenter.normal_stop_seen

    @override
    async def prepare(self) -> None:
        runner = self._runner
        (
            self.body,
            self.sent_body,
            self._client,
        ) = await runner._transport._prepare_create(
            self.body,
            self.context.operation_kind,
            history=self.context.history,
            endpoint=runner._endpoint,
            request_client=runner._request_client,
            extra_headers=runner._extra_headers,
            on_origin=partial(setattr, self._presenter.output, "replay_origin"),
            call_replay=runner._call_replay,
            native_chat=runner._native_chat,
        )

    @override
    async def stream(self, scope: ProviderAttemptScope) -> AsyncIterator[StreamOutput]:
        assert self._client is not None
        runner = self._runner
        transport = runner._transport
        presenter = self._presenter
        if transport._validate_prepared_body is not None:
            self._validating = True
            await transport._validate_prepared_body(self.sent_body)
            self._validating = False
        if runner._native_chat:
            self.sent_body = _sdk_create_body(self.sent_body)
        stream = await transport._open_stream(
            scope,
            self._client,
            self.sent_body,
            self.body,
            history=self.context.history,
            on_event=self._observe,
            native_chat=runner._native_chat,
        )
        presenter.bind_tools(
            runner._tool_argument_aliases,
            ToolCallScope.from_body(self.sent_body, "chat")
            if runner._constrain_tool_calls
            else None,
        )
        async for chunk in stream:
            if not scope.attempt.accepted:
                await scope.attempt.accept()
            for event in presenter.start_events():
                if event:
                    yield StreamOutput((event,))
            for output in presenter.feed(chunk):
                yield output
        for event in presenter.finish_upstream():
            if event:
                yield StreamOutput((event,))
        for event in presenter.prepare_completion(
            pad_empty=not self.context.continuing and presenter.output.pad_empty
        ):
            if event:
                yield StreamOutput((event,))
        completion = presenter.completion
        if completion.provider_input_tokens is not None:
            logger.debug(
                "TOKEN_ESTIMATE: our={} provider={} diff={:+d}",
                runner._input_tokens,
                completion.provider_input_tokens,
                completion.provider_input_tokens - runner._input_tokens,
            )
        usage = ChatStreamUsage(
            finish_reason=completion.finish_reason,
            reported={
                key: value
                for key, value in (
                    ("prompt_tokens", usage_int(presenter.usage_info, "prompt_tokens")),
                    (
                        "completion_tokens",
                        usage_int(presenter.usage_info, "completion_tokens"),
                    ),
                    ("total_tokens", usage_int(presenter.usage_info, "total_tokens")),
                    (
                        "cached_tokens",
                        runner._transport._behavior.cached_input_tokens(
                            presenter.usage_info
                        ),
                    ),
                    (
                        "reasoning_tokens",
                        nested_usage_int(
                            presenter.usage_info,
                            "completion_tokens_details",
                            "reasoning_tokens",
                        ),
                    ),
                )
                if value is not None
            },
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cached_tokens=runner._transport._behavior.cached_input_tokens(
                presenter.usage_info
            )
            or 0,
            cache_write_tokens=runner._transport._behavior.cache_write_input_tokens(
                presenter.usage_info
            ),
            reasoning_tokens=(
                nested_usage_int(
                    presenter.usage_info,
                    "completion_tokens_details",
                    "reasoning_tokens",
                )
                or 0
            ),
            anthropic_fields=runner._transport._behavior.anthropic_usage_fields(
                presenter.usage_info
            ),
        )
        for event in presenter.terminal_events(usage=usage):
            if event:
                yield StreamOutput((event,))

    @override
    def resolvable(self, error: Exception) -> bool:
        # A local check sent nothing upstream; the permit is abandoned without
        # teaching correction or recovery state about a rejection.
        return not self._validating

    @override
    def abandon(self, error: Exception) -> Iterator[StreamOutput]:
        return iter(self._presenter.abandon())

    @override
    def failure_payload(
        self, error: Exception, failure: ExecutionFailure
    ) -> JsonObject | None:
        return self._presenter.output.failure_payload(failure)

    @override
    def completed_fields(self) -> dict[str, Any]:
        completion = self._presenter.completion
        return {
            "finish_reason": completion.finish_reason,
            "output_tokens": completion.output_tokens,
            "prompt_tokens": completion.input_tokens,
            "prompt_tokens_estimate": self._runner._input_tokens,
        }

    def _observe(self, chunk: ChatCompletionChunk) -> None:
        self._presenter.observe_upstream_chunk(chunk)
        source = self.context.source
        continuing = self.context.continuing
        if self._runner._native_chat:
            source.observe(
                "chat",
                _native_chunk_payload(chunk),
                continuing=continuing,
            )
            return
        choices: list[JsonObject] = []
        # The SDK can leave choices unset on a usage-only trailer.
        for choice in chunk.choices or ():
            # The SDK can keep a provider's empty stop delta as a raw value.
            delta: object = choice.delta
            metadata: JsonObject = {}
            if isinstance(delta, ChoiceDelta):
                extras: JsonObject = delta.model_extra or {}
                metadata = {
                    "reasoning_details": extras.get("reasoning_details"),
                    "audio": bool(delta.audio),
                }
            choices.append({"delta": metadata})
        source.observe(
            "chat",
            {
                "usage": {
                    key: value
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if (value := usage_int(chunk.usage, key)) is not None
                },
                "choices": choices,
            },
            continuing=continuing,
        )
