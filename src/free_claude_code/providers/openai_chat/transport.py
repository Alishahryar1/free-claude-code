"""Shared Chat Completions transport and per-request stream execution."""

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import httpx2
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import ProviderCandidate
from free_claude_code.core.anthropic import (
    ContentBlockToolUse,
    ContentType,
    ThinkTagParser,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.streaming import (
    ToolSchema,
    map_stop_reason,
    tool_schemas_by_name,
)
from free_claude_code.core.diagnostics import (
    exception_cause_types,
    redacted_exception_traceback,
)
from free_claude_code.core.failures import (
    ExecutionFailure,
    FailureKind,
    UnsupportedRequestFeature,
)
from free_claude_code.core.history_replay import ReplayOrigin, prepare_history
from free_claude_code.core.json_types import JsonObject
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
from free_claude_code.core.recovery import CandidateIncompatible, RecoveryCheckpoint
from free_claude_code.core.recovery_request import continue_request
from free_claude_code.core.request_preservation import require_preserved_body
from free_claude_code.core.stream_events import (
    DecodedStreamEvent,
    RequestOutcome,
    StreamEvent,
)
from free_claude_code.core.tool_input import complete_json_object
from free_claude_code.core.trace import provider_chat_body_snapshot, trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
)
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.endpoint_types import EndpointContext
from free_claude_code.providers.failure_policy import (
    TruncatedProviderStreamError,
    context_window_exceeded_provider_failure,
    is_context_window_finish_reason,
    is_retryable_stream_error,
)
from free_claude_code.providers.history_replay import (
    replay_origin,
    validate_history,
)
from free_claude_code.providers.http import (
    ProviderAttemptScope,
)
from free_claude_code.providers.openai_client import OpenAIRequestClient
from free_claude_code.providers.openai_stream import OpenAIStreamAdapter
from free_claude_code.providers.reasoning_compatibility import (
    ReasoningCorrection,
    prepare_messages_reasoning,
)
from free_claude_code.providers.request_recovery import (
    RequestCorrections,
)
from free_claude_code.providers.stream_candidate import (
    StreamCandidate,
    content_progress,
)

from .behavior import OpenAIChatBehavior
from .output_cap import clamp_output_tokens, parse_output_token_cap
from .profiles import OpenAIChatProfile
from .reasoning_details import StructuredReasoningStream
from .request_policy import (
    apply_openai_chat_body_policy,
)
from .stream_output import (
    AnthropicChatStreamOutput,
    ChatStreamOutput,
    ChatStreamUsage,
    ResponsesChatStreamOutput,
)
from .tool_calls import (
    OpenAIToolCallAssembler,
    tool_call_extra_content,
)
from .usage import (
    clone_without_stream_usage,
    is_stream_usage_rejection,
    nested_usage_int,
    request_stream_usage,
    usage_int,
)

OpenAIAsyncCredentialProvider = Callable[[], Awaitable[str]]
_ExtraReasoningEvents = Callable[[Any, ChatStreamOutput], Iterator[StreamEvent]]
_ChatOutputFactory = Callable[[], ChatStreamOutput]


def _iter_visible_text_events(
    output: ChatStreamOutput,
    text: str,
) -> Iterator[StreamEvent]:
    yield from output.ensure_text_block()
    yield output.emit_text_delta(text)


@dataclass(frozen=True, slots=True)
class _OpenAIChatCompletion:
    finish_reason: Any
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


class _OpenAIChatStreamAssembler:
    """Own one discardable OpenAI-chat replay epoch."""

    def __init__(
        self,
        *,
        output: ChatStreamOutput,
        profile: OpenAIChatProfile,
        provider_name: str,
        output_reasoning: bool,
        tool_names: OpenAIToolNameCodec,
        tool_schemas: dict[str, ToolSchema],
        tool_calls: OpenAIToolCallAssembler,
        extra_reasoning_events: _ExtraReasoningEvents,
    ) -> None:
        self._output = output
        self._profile = profile
        self._provider_name = provider_name
        self._output_reasoning = output_reasoning
        self._tool_names = tool_names
        self._tool_schemas = tool_schemas
        self._tool_calls = tool_calls
        self._extra_reasoning_events = extra_reasoning_events
        self._think_parser = ThinkTagParser()
        self._structured_reasoning = (
            StructuredReasoningStream()
            if profile.structured_reasoning_details
            else None
        )
        if self._structured_reasoning is not None:
            self._output.reasoning_replay = self._structured_reasoning
        self._finish_reason: Any = None
        self._usage_info: Any = None
        self._native_reasoning_seen = False
        self._tool_argument_aliases: dict[str, dict[str, str]] = {}
        self._tool_argument_alias_buffers: dict[int, str] = {}
        self._tool_name_buffers: dict[int, str] = {}
        self._started = False
        self._aliases_bound = False
        self._upstream_finished = False
        self._completion: _OpenAIChatCompletion | None = None
        self._completed = False
        self.invalid_input = False

    @property
    def output(self) -> ChatStreamOutput:
        return self._output

    @property
    def usage_info(self) -> Any:
        return self._usage_info

    @property
    def native_reasoning_pending(self) -> bool:
        return (
            self._structured_reasoning is not None and self._structured_reasoning.active
        )

    @property
    def completion(self) -> _OpenAIChatCompletion:
        if self._completion is None:
            raise RuntimeError("stream completion has not been prepared")
        # Usage may arrive after the terminal choice and completed client calls.
        output_tokens = usage_int(self._usage_info, "completion_tokens")
        input_tokens = usage_int(self._usage_info, "prompt_tokens")
        return replace(
            self._completion,
            output_tokens=output_tokens
            if output_tokens is not None
            else self._completion.output_tokens,
            input_tokens=input_tokens
            if input_tokens is not None
            else self._completion.input_tokens,
            provider_input_tokens=input_tokens,
        )

    @property
    def ready_to_complete(self) -> bool:
        return self._finish_reason is not None and self._completion is None

    @property
    def content_completed(self) -> bool:
        return self._completion is not None

    def start_events(self) -> Iterator[StreamEvent]:
        if self._started:
            return
        self._started = True
        yield from self._output.start_events()

    def bind_tool_argument_aliases(self, aliases: dict[str, dict[str, str]]) -> None:
        if self._aliases_bound:
            raise RuntimeError("tool argument aliases already bound")
        self._aliases_bound = True
        self._tool_argument_aliases = aliases

    def feed(self, chunk: Any) -> Iterator[StreamEvent]:
        if not self._started:
            raise RuntimeError("stream assembler is not accepting chunks")

        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            self._usage_info = chunk_usage

        if not chunk.choices:
            return
        if self._upstream_finished:
            raise TruncatedProviderStreamError(
                "Provider sent another choice after completion."
            )

        if (
            self._output.replay_origin is not None
            and isinstance(getattr(chunk, "model", None), str)
            and chunk.model
        ):
            self._output.replay_origin = replace(
                self._output.replay_origin, model=chunk.model
            )
        choice = chunk.choices[0]
        delta = choice.delta
        if choice.finish_reason:
            self._finish_reason = choice.finish_reason
        if delta is None:
            return

        if choice.finish_reason:
            self._finish_reason = choice.finish_reason
            logger.debug(
                "{} finish_reason: {}",
                self._provider_name,
                self._finish_reason,
            )

        reasoning = self._profile.reasoning_delta(delta)
        if self._output_reasoning:
            if self._structured_reasoning is not None:
                yield from self._structured_reasoning.events(
                    delta,
                    self._output,
                    native_reasoning=reasoning,
                )
            elif reasoning is not None and (
                reasoning or not self._native_reasoning_seen
            ):
                # Preserve initial empty reasoning for replay; later empty fields
                # are placeholders and must not interrupt text or tool output.
                self._native_reasoning_seen = True
                yield from self._output.ensure_reasoning_block()
                if reasoning:
                    yield self._output.emit_reasoning_delta(reasoning)

        yield from self._extra_reasoning_events(delta, self._output)

        native_tool_calls = delta.tool_calls
        if delta.content:
            for part in self._think_parser.feed(delta.content):
                if part.type == ContentType.THINKING:
                    if not self._output_reasoning:
                        continue
                    yield from self._output.ensure_reasoning_block()
                    yield self._output.emit_reasoning_delta(part.content)
                else:
                    yield from _iter_visible_text_events(self._output, part.content)

        if native_tool_calls:
            yield from self._output.close_content_blocks()
            for tool_call in native_tool_calls:
                extra_content = tool_call_extra_content(tool_call)
                tool_call_info = {
                    "index": tool_call.index,
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                if extra_content:
                    tool_call_info["extra_content"] = extra_content
                yield from self._tool_calls.process_tool_call(
                    tool_call_info,
                    self._output,
                    tool_names=self._tool_names,
                    tool_name_buffers=self._tool_name_buffers,
                    tool_argument_aliases=self._tool_argument_aliases,
                    tool_argument_alias_buffers=self._tool_argument_alias_buffers,
                )

    def finish_upstream(self) -> Iterator[StreamEvent]:
        if self._upstream_finished:
            return
        if self._finish_reason is None:
            raise TruncatedProviderStreamError(
                "Provider stream ended without finish_reason."
            )
        if is_context_window_finish_reason(self._finish_reason):
            raise context_window_exceeded_provider_failure()
        if any(
            not self._tool_names.is_unchanged_name(name)
            for name in self._tool_name_buffers.values()
        ):
            raise TruncatedProviderStreamError(
                "Provider stream ended with an incomplete tool name."
            )

        remaining = self._think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                if self._output_reasoning:
                    yield from self._output.ensure_reasoning_block()
                    yield self._output.emit_reasoning_delta(remaining.content)
            else:
                yield from _iter_visible_text_events(self._output, remaining.content)

        yield from self._output.flush_reasoning_replay()
        self._upstream_finished = True

    def prepare_completion(self) -> Iterator[StreamEvent]:
        if not self._upstream_finished or self._completion is not None:
            raise RuntimeError("stream completion cannot be prepared")

        yield from self._tool_calls.flush_tool_name_buffers(
            self._output,
            tool_names=self._tool_names,
            tool_name_buffers=self._tool_name_buffers,
            tool_argument_aliases=self._tool_argument_aliases,
            tool_argument_alias_buffers=self._tool_argument_alias_buffers,
        )

        has_emitted_tool = self._output.has_emitted_tool_block()
        has_content_blocks = self._output.has_content_block()
        if not has_content_blocks or (
            not has_emitted_tool
            and not self._output.accumulated_text.strip()
            and self._output.accumulated_reasoning.strip()
        ):
            yield from self._output.ensure_text_block()
            yield self._output.emit_text_delta(" ")

        yield from self._tool_calls.flush_tool_argument_alias_buffers(
            self._output,
            self._tool_argument_aliases,
            self._tool_argument_alias_buffers,
        )
        for state in self._output.tool_states.values():
            if not complete_json_object(state.content):
                self.invalid_input = True
                state.open = False
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

    def terminal_events(self, *, usage: ChatStreamUsage) -> Iterator[StreamEvent]:
        if self._completed:
            return
        completion = self.completion
        yield from self._output.finish_success(
            stop_reason=map_stop_reason(completion.finish_reason),
            usage=usage,
        )
        self._completed = True


class OpenAIChatTransport:
    """Execute Chat requests while borrowing provider-owned HTTP resources."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        admission: ProviderAdmissionController,
        behavior: OpenAIChatBehavior,
        read_timeout_s: float,
        log_raw_sse_events: bool,
        log_api_error_tracebacks: bool,
        endpoint_transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = client
        self._admission = admission
        self._behavior = behavior
        self._profile = behavior.profile
        self._provider_name = self._profile.provider_name
        self._read_timeout_s = read_timeout_s
        self._log_raw_sse_events = log_raw_sse_events
        self._log_api_error_tracebacks = log_api_error_tracebacks
        self._endpoint_transport = endpoint_transport
        self._model_output_caps: dict[str, int] = {}

    def _log_stream_transport_error(
        self,
        tag: str,
        req_tag: str,
        error: Exception,
        *,
        request_id: str | None = None,
    ) -> None:
        """Log streaming transport failures (metadata-only unless verbose is enabled)."""
        response = getattr(error, "response", None)
        http_status = (
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
        preserve_features: bool = False,
    ) -> dict[str, Any]:
        """Build a provider request from the immutable profile."""
        request, reasoning = self._prepare_messages_reasoning(
            request, reasoning, model_info, preserve_features=preserve_features
        )
        return self._behavior.build_messages_body(
            request, reasoning=reasoning, preserve_features=preserve_features
        )

    def _prepare_messages_reasoning(
        self,
        request: MessagesRequest,
        reasoning: ReasoningPolicy,
        model_info: ProviderModelInfo | None,
        *,
        preserve_features: bool = False,
    ) -> tuple[MessagesRequest, ReasoningPolicy]:
        return prepare_messages_reasoning(
            request,
            reasoning,
            model_info=model_info,
            can_disable=bool(self._behavior.reasoning_off_fields),
            normal_max_tokens=self._behavior.normal_max_tokens,
            preserve_features=preserve_features,
        )

    def _build_responses_request_body(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        preserve_features: bool = False,
    ) -> ResponsesChatRequest:
        """Build a Chat body directly from Responses ingress."""
        validate_history(request.model_dump(mode="json"))
        try:
            translated = build_responses_chat_request(
                request,
                reasoning_replay=self._profile.request_policy.reasoning_replay,
                structured_reasoning_details=(
                    self._profile.structured_reasoning_details
                ),
                preserve_features=preserve_features,
            )
        except UnsupportedRequestFeature as error:
            raise CandidateIncompatible(str(error)) from error
        except ResponsesConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc
        body = translated.body
        apply_openai_chat_body_policy(
            body, self._profile.request_policy, preserve_features=preserve_features
        )
        self._profile.apply_reasoning_to_body(body, reasoning)
        unshaped = deepcopy(body) if preserve_features else None
        body = self._behavior.finalize_chat_body(body, reasoning=reasoning)
        if unshaped is not None:
            require_preserved_body(unshaped, body, "Chat provider adaptation")
        encode_openai_chat_tool_names(body, translated.tool_names)
        return ResponsesChatRequest(
            body=body,
            tool_names=translated.tool_names,
            tool_schemas=translated.tool_schemas,
            reserved_tool_ids=translated.reserved_tool_ids,
            tool_adapter=translated.tool_adapter,
        )

    def _next_chat_retry_body(
        self,
        error: Exception,
        body: dict,
        used_retry_kinds: set[str],
        *,
        sent_body: Mapping[str, Any] | None = None,
    ) -> dict | None:
        retry_body = self._retry_body_for_output_cap(error, body)
        if retry_body is not None:
            return retry_body

        if "stream_usage" not in used_retry_kinds and is_stream_usage_rejection(error):
            retry_body = clone_without_stream_usage(body)
            if retry_body is not None:
                used_retry_kinds.add("stream_usage")
                logger.warning(
                    "{}_STREAM: retrying without stream_options.include_usage "
                    "after upstream rejection",
                    self._provider_name,
                )
                return retry_body

        if "provider_specific" not in used_retry_kinds:
            retry_body = self._behavior.retry_request_body(
                error, dict(sent_body) if sent_body is not None else body
            )
            if retry_body is not None:
                used_retry_kinds.add("provider_specific")
                return retry_body

        return self._behavior.retry_after_standard_corrections(
            error, body, used_retry_kinds
        )

    def _apply_learned_output_cap(self, body: dict) -> dict:
        """Clamp output tokens to a previously learned cap for this model."""
        model = body.get("model")
        if not isinstance(model, str):
            return body
        cap = self._model_output_caps.get(model)
        if cap is None:
            return body
        clamped = clamp_output_tokens(body, cap)
        return clamped if clamped is not None else body

    def _retry_body_for_output_cap(self, error: Exception, body: dict) -> dict | None:
        """Learn an upstream output-token cap from a 400 and clamp for one retry."""
        cap = parse_output_token_cap(error)
        if cap is None:
            return None
        model = body.get("model")
        if isinstance(model, str):
            previous = self._model_output_caps.get(model)
            cap = cap if previous is None else min(previous, cap)
            self._model_output_caps[model] = cap
        clamped = clamp_output_tokens(body, cap)
        if clamped is None:
            return None
        logger.warning(
            "{}_STREAM: clamping output tokens to {} after upstream cap rejection",
            self._provider_name,
            cap,
        )
        return clamped

    def open_messages(
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
    ) -> AbstractAsyncContextManager[ProviderCandidate]:
        return self._open_candidate(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            model_info=model_info,
            endpoint_context=endpoint_context,
            extra_headers=extra_headers,
        )

    def open_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        endpoint_context: EndpointContext | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> AbstractAsyncContextManager[ProviderCandidate]:
        return self._open_candidate(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            model_info=None,
            endpoint_context=endpoint_context,
            extra_headers=extra_headers,
        )

    @asynccontextmanager
    async def _open_candidate(
        self,
        request: MessagesRequest | OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        model_info: ProviderModelInfo | None,
        endpoint_context: EndpointContext | None,
        extra_headers: Mapping[str, str] | None,
    ) -> AsyncIterator[ProviderCandidate]:
        candidate = _ChatCandidate(
            self,
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            model_info=model_info,
            endpoint_context=endpoint_context,
            extra_headers=extra_headers,
        )
        try:
            yield candidate
        finally:
            await candidate.aclose()


class _ChatCandidate(StreamCandidate):
    def __init__(
        self,
        transport: OpenAIChatTransport,
        request: MessagesRequest | OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        model_info: ProviderModelInfo | None,
        endpoint_context: EndpointContext | None,
        extra_headers: Mapping[str, str] | None,
    ) -> None:
        super().__init__(
            admission=transport._admission,
            provider_name=transport._provider_name,
            protocol="chat",
            read_timeout_s=transport._read_timeout_s,
            request_id=request_id,
            endpoint=RequestEndpoint(endpoint_context)
            if endpoint_context is not None
            else None,
            failure_override=transport._behavior.failure_override,
        )
        self._transport = transport
        self._request = request.model_copy(deep=True)
        self.original_body = request.model_dump(mode="json", exclude_unset=True)
        self.input_protocol = (
            "messages" if isinstance(request, MessagesRequest) else "responses"
        )
        self._input_tokens = input_tokens
        self._response_model = response_model
        self._reasoning = reasoning
        self._model_info = model_info
        self._extra_headers = dict(extra_headers or {})
        self._request_client = OpenAIRequestClient(transport._endpoint_transport)
        self._client = transport._client
        self._output_factory: _ChatOutputFactory
        self._tool_names: OpenAIToolNameCodec
        self._tool_schemas: dict[str, ToolSchema]
        self._reserved_tool_ids: frozenset[str]

    def _build_body(self, checkpoint: RecoveryCheckpoint) -> None:
        request = continue_request(self._request, checkpoint)
        correction = None
        if isinstance(request, MessagesRequest):
            prepared, wire_reasoning = self._transport._prepare_messages_reasoning(
                request,
                self._reasoning,
                self._model_info,
                preserve_features=self.preserve_features,
            )
            self.body = self._transport._build_request_body(
                prepared,
                reasoning=wire_reasoning,
                preserve_features=self.preserve_features,
            )
            off_fields = self._transport._behavior.reasoning_off_fields
            if (
                self._reasoning.control is ReasoningControl.PREFER_OFF
                and wire_reasoning.control is ReasoningControl.OFF
                and off_fields
            ):
                correction = ReasoningCorrection(
                    off_fields,
                    self._transport._profile.request_policy.max_tokens_field,
                    self._transport._behavior.normal_max_tokens,
                    provider_rejection=self._transport._behavior.reasoning_disable_rejected,
                )
            self._tool_names = OpenAIToolNameCodec.from_request(request)
            self._tool_schemas = tool_schemas_by_name(request)
            self._reserved_tool_ids = _reserved_anthropic_tool_ids(request)
            self._output_factory = lambda: AnthropicChatStreamOutput(
                message_id=f"msg_{uuid.uuid4()}",
                model=self._response_model,
                input_tokens=self._input_tokens,
                log_raw_events=self._transport._log_raw_sse_events,
            )
        else:
            translated = self._transport._build_responses_request_body(
                request,
                reasoning=self._reasoning,
                preserve_features=self.preserve_features,
            )
            self.body = cast(JsonObject, translated.body)
            self._tool_names = translated.tool_names
            self._tool_schemas = {
                name: ToolSchema(name=name, input_schema=schema)
                for name, schema in translated.tool_schemas.items()
            }
            self._reserved_tool_ids = translated.reserved_tool_ids
            self._output_factory = lambda: ResponsesChatStreamOutput(
                translated.tool_adapter,
                input_tokens=self._input_tokens,
                response_model=self._response_model,
            )
        self.body = self._transport._apply_learned_output_cap(self.body)
        # Corrections operate on the sent wire body, which excludes private
        # argument metadata. Keep the decoding map for this request revision.
        self._tool_argument_aliases = self._transport._behavior.tool_argument_aliases(
            self.body
        )
        request_stream_usage(self.body)
        self.corrections = RequestCorrections("chat", correction)

    async def _prepare_endpoint(self) -> ReplayOrigin:
        self._client = (
            self._request_client.for_endpoint(
                self._transport._client, await self.endpoint.resolve()
            )
            if self.endpoint is not None
            else self._transport._client
        )
        origin = replay_origin(
            self.provider_name,
            "chat",
            str(self.body["model"]),
            client=self._client,
            endpoint=self.endpoint.snapshot if self.endpoint is not None else None,
        )
        before = deepcopy(self.body) if self.preserve_features else None
        create_body = self._transport._behavior.prepare_create_body(self.body)
        if before is not None:
            require_preserved_body(before, create_body, "Chat dispatch adaptation")
        if self._extra_headers or self.endpoint is not None:
            create_body = {
                **create_body,
                "extra_headers": {
                    **(create_body.get("extra_headers") or {}),
                    **self._extra_headers,
                    **self._request_client.openai_headers(),
                },
            }
        self.sent_body = prepare_history(
            cast(JsonObject, create_body),
            origin,
            scope=self._transport._behavior.history_scope(self.body),
            reasoning_field=self._transport._profile.request_policy.reasoning_replay.value,
            structured_details=self._transport._profile.structured_reasoning_details,
            preserve_features=self.preserve_features,
        )
        return origin

    async def _read(
        self, scope: ProviderAttemptScope
    ) -> AsyncIterator[DecodedStreamEvent]:
        assert self.origin is not None
        output = self._output_factory()
        output.replay_origin = self.origin
        assembler = _OpenAIChatStreamAssembler(
            output=output,
            profile=self._transport._profile,
            provider_name=self.provider_name,
            output_reasoning=self._reasoning.output_enabled,
            tool_names=self._tool_names,
            tool_schemas=self._tool_schemas,
            tool_calls=OpenAIToolCallAssembler(
                reserved_tool_ids=self._reserved_tool_ids,
                record_extra_content=self._transport._behavior.record_tool_call_extra_content,
            ),
            extra_reasoning_events=lambda delta, target: (
                self._transport._behavior.extra_reasoning_events(
                    delta,
                    target,
                    output_reasoning=self._reasoning.output_enabled,
                )
            ),
        )
        assembler.bind_tool_argument_aliases(self._tool_argument_aliases)
        sdk = await self._client.chat.completions.create(
            **cast(dict[str, Any], self.sent_body), stream=True
        )
        source = OpenAIStreamAdapter(sdk)
        try:
            stream = self._transport._behavior.normalize_stream(source, self.body)
        except BaseException:
            scope.retain(source)
            raise
        scope.retain(stream)
        try:
            async for chunk in stream:
                if not scope.attempt.accepted:
                    await scope.attempt.accept()
                raw = cast(JsonObject, _chunk_payload(chunk))
                events = (*assembler.start_events(), *assembler.feed(chunk))
                if assembler.ready_to_complete:
                    events = (
                        *events,
                        *assembler.finish_upstream(),
                        *assembler.prepare_completion(),
                    )
                yield DecodedStreamEvent(
                    output.replay_origin or self.origin,
                    StreamEvent("chat.completion.chunk", raw),
                    output.project(events),
                    progress=content_progress("chat", "chat.completion.chunk", raw),
                    native_reasoning_pending=assembler.native_reasoning_pending,
                )
        except Exception as error:
            if not assembler.content_completed or not is_retryable_stream_error(error):
                raise
            trace_event(
                stage="provider",
                event="provider.usage_trailer.unavailable",
                source="provider",
                provider=self.provider_name,
                request_id=self.request_id,
                exc_type=type(error).__name__,
            )
        events = (
            ()
            if assembler.content_completed
            else (*assembler.finish_upstream(), *assembler.prepare_completion())
        )
        completion = assembler.completion
        if assembler.invalid_input and completion.finish_reason != "length":
            raise TruncatedProviderStreamError(
                "Provider completed a response with unfinished or invalid tool input."
            )
        usage = ChatStreamUsage(
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cached_tokens=self._transport._behavior.cached_input_tokens(
                assembler.usage_info
            )
            or 0,
            cache_write_tokens=self._transport._behavior.cache_write_input_tokens(
                assembler.usage_info
            ),
            reasoning_tokens=nested_usage_int(
                assembler.usage_info, "completion_tokens_details", "reasoning_tokens"
            )
            or 0,
            anthropic_fields=self._transport._behavior.anthropic_usage_fields(
                assembler.usage_info
            ),
        )
        events = (*events, *assembler.terminal_events(usage=usage))
        yield DecodedStreamEvent(
            output.replay_origin or self.origin,
            StreamEvent(
                "chat.completion.done", {"finish_reason": completion.finish_reason}
            ),
            output.project(events),
            outcome=RequestOutcome.INCOMPLETE
            if completion.finish_reason in {"length", "content_filter"}
            else RequestOutcome.SUCCESS,
            stop_reason=completion.finish_reason,
        )

    def _correction(self, error: Exception) -> JsonObject | None:
        body = self.corrections.next_body(
            error,
            self.body,
            sent_body=self.sent_body,
            reasoning_error=error,
            reasoning_sent_body=self.sent_body,
            after_common=partial(
                self._transport._next_chat_retry_body,
                error,
                self.body,
                sent_body=self.sent_body,
            ),
        )
        return (
            self._transport._apply_learned_output_cap(body)
            if body is not None
            else None
        )

    def _effective_error(self, error: Exception) -> Exception:
        if not isinstance(error, CandidateIncompatible):
            self._transport._log_stream_transport_error(
                self.provider_name,
                self.request_id or "",
                error,
                request_id=self.request_id,
            )
        if isinstance(error, ResponsesConversionError):
            return ExecutionFailure(
                FailureKind.UPSTREAM,
                502,
                "Provider tool output cannot be represented in the requested protocol.",
                False,
            )
        return error

    def _request_trace_fields(self) -> JsonObject:
        return {"body": provider_chat_body_snapshot(self.sent_body)}

    async def aclose(self) -> None:
        try:
            await self._request_client.aclose()
        finally:
            await super().aclose()


def _chunk_payload(value: Any) -> Any:
    """Preserve SDK fields and the native views produced by stream normalizers."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_unset=True, warnings=False)
    if isinstance(value, SimpleNamespace):
        return {key: _chunk_payload(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _chunk_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_chunk_payload(item) for item in value]
    return value
