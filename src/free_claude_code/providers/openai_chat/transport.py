"""Shared Chat Completions transport and per-request stream execution."""

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, assert_never, cast

import httpx2
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChoiceDelta
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
from free_claude_code.core.trace import provider_chat_body_snapshot, trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderAttempt,
    ProviderOperationKind,
)
from free_claude_code.providers.canonical_failures import (
    context_window_exceeded_provider_failure,
    is_context_window_finish_reason,
)
from free_claude_code.providers.continuation import (
    ContinuationRequest,
    SourceRecoveryState,
)
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.endpoint_types import EndpointContext
from free_claude_code.providers.failure_classification import classify
from free_claude_code.providers.history_negotiation import (
    HistoryModeCache,
    resolve_history_overrides,
)
from free_claude_code.providers.history_replay import (
    replay_origin,
    validate_history,
)
from free_claude_code.providers.http import (
    ProviderAttemptScope,
    close_provider_stream,
    maybe_await_aclose,
)
from free_claude_code.providers.openai_client import OpenAIRequestClient
from free_claude_code.providers.openai_stream import OpenAIStreamAdapter
from free_claude_code.providers.reasoning_compatibility import (
    ReasoningCorrection,
    prepare_messages_reasoning,
)
from free_claude_code.providers.request_recovery import (
    RequestCorrections,
    RequestRecovery,
)
from free_claude_code.providers.stream_failure_resolution import (
    Continue,
    Fail,
    Handoff,
    Retry,
    StreamFailure,
    StreamFailurePolicy,
    StreamFailureResolver,
)
from free_claude_code.providers.stream_recovery import (
    TruncatedProviderStreamError,
)

from .behavior import OpenAIChatBehavior
from .gemini_output import GeminiChatStreamOutput
from .native_output import NativeChatStreamOutput
from .output_cap import clamp_output_tokens, parse_output_token_cap
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
_ExtraReasoningEvents = Callable[[Any, ChatStreamOutput], Iterator[str]]
_ChatOutputFactory = Callable[[], ChatStreamOutput]


def _iter_visible_text_events(
    output: ChatStreamOutput,
    text: str,
) -> Iterator[str]:
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
        self._tool_buffer = ChatToolCallBuffer()
        self._reserved_tool_ids = set(reserved_tool_ids)
        self._record_extra_content = record_extra_content
        self._extra_reasoning_events = extra_reasoning_events
        self._think_parser = ThinkTagParser()
        self._structured_reasoning = StructuredReasoningStream(
            structured_details=profile.structured_reasoning_details
        )
        self._output.reasoning_replay = self._structured_reasoning
        self._finish_reason: Any = None
        self._raw_stop_seen = False
        self._usage_info: Any = None
        self._tool_argument_aliases: dict[str, dict[str, str]] = {}
        self._started = False
        self._aliases_bound = False
        self._upstream_finished = False
        self._completion: _OpenAIChatCompletion | None = None
        self._completed = False

    @property
    def output(self) -> ChatStreamOutput:
        return self._output

    @property
    def usage_info(self) -> Any:
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

    def observe_upstream_chunk(self, chunk: Any) -> None:
        if chunk.choices and chunk.choices[0].finish_reason is not None:
            self._raw_stop_seen = True

    def start_events(self) -> Iterator[str]:
        if self._started:
            return
        self._started = True
        yield from self._output.start_events()

    def bind_tool_argument_aliases(self, aliases: dict[str, dict[str, str]]) -> None:
        if self._aliases_bound:
            raise RuntimeError("tool argument aliases already bound")
        self._aliases_bound = True
        self._tool_argument_aliases = aliases

    def feed(self, chunk: Any) -> Iterator[StreamOutput]:
        if not self._started or self._upstream_finished:
            raise RuntimeError("stream presenter is not accepting chunks")

        if chunk.choices and chunk.choices[0].finish_reason is not None:
            self._finish_reason = chunk.choices[0].finish_reason
        yield from self._present(
            self._tool_buffer.feed(chunk.model_dump(mode="json", exclude_unset=True))
        )

    def abandon(self) -> Iterator[StreamOutput]:
        yield from self._present(self._tool_buffer.abandon())

    def _present(self, items) -> Iterator[StreamOutput]:
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
                extra = call.body.get("extra_content")
                if extra:
                    self._record_extra_content(call.body["id"], extra)
                frames = [
                    *self._output.close_content_blocks(),
                    *self._output.emit_complete_tool(
                        call.index,
                        CompletedToolCall(
                            call.body["id"],
                            function["name"],
                            function["arguments"],
                            extra,
                        ),
                    ),
                ]
                yield completed_output(frames)
            else:
                frames = tuple(self._feed_frame(item))
                if frames:
                    yield StreamOutput(frames)

    def _feed_frame(self, chunk: dict[str, Any]) -> Iterator[str]:

        chunk_usage = chunk.get("usage")
        if chunk_usage is not None:
            self._usage_info = chunk_usage

        if not chunk.get("choices"):
            return

        choice = chunk["choices"][0]
        raw_delta = choice.get("delta")
        if choice.get("finish_reason") is not None:
            self._finish_reason = choice["finish_reason"]
        if raw_delta is None:
            return
        # This is already SDK-decoded source data. Preserve its permissive extras
        # for provider reasoning hooks without validating unrelated usage fields.
        delta = ChoiceDelta.model_construct(**raw_delta)

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

        if delta.content:
            for part in self._think_parser.feed(delta.content):
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
        yield from self._output.finish_success(
            stop_reason=map_stop_reason(completion.finish_reason),
            usage=usage,
        )
        self._completed = True


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
    def generated_output(self) -> bool:
        return self._generated

    @property
    def normal_stop_seen(self) -> bool:
        return self._relay.normal_stop_seen

    def observe_upstream_chunk(self, chunk: Any) -> None:
        pass

    def start_events(self) -> Iterator[str]:
        self._started = True
        return iter(())

    def feed(self, chunk: Any) -> Iterator[StreamOutput]:
        payload = chunk.model_dump(mode="json", exclude_unset=True)
        if getattr(chunk, "usage", None) is not None:
            self._usage_info = chunk.usage
        for choice in payload.get("choices", []):
            if choice.get("finish_reason") is not None:
                self._finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            self._generated |= any(
                value for key, value in delta.items() if key != "role"
            )
        yield from self._present(self._tool_buffer.feed(payload))

    def _present(self, items) -> Iterator[StreamOutput]:
        for item in items:
            if isinstance(item, CompletedChatBatch):
                yield completed_output(self._relay.feed(frame) for frame in item.frames)
            else:
                yield StreamOutput((self._relay.feed(item),))

    def finish_upstream(self) -> Iterator[str]:
        if not self._relay.normal_stop_seen:
            raise TruncatedProviderStreamError(
                "Native Chat stream ended before all choices completed"
            )
        self._upstream_finished = True
        return iter(())

    def prepare_completion(self, *, pad_empty: bool) -> Iterator[str]:
        provider_input = usage_int(self._usage_info, "prompt_tokens")
        self._completion = _OpenAIChatCompletion(
            self._finish_reason,
            usage_int(self._usage_info, "completion_tokens") or 0,
            provider_input if provider_input is not None else self._output.input_tokens,
            provider_input,
        )
        return iter(())

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
    ) -> dict[str, Any]:
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
        validate_history(request.model_dump(mode="json"))
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

    async def _create_stream(
        self,
        body: dict,
        request_recovery: RequestRecovery,
        operation_kind: ProviderOperationKind,
        *,
        corrections: RequestCorrections | None = None,
        endpoint: RequestEndpoint | None = None,
        request_client: OpenAIRequestClient | None = None,
        extra_headers: Mapping[str, str] | None = None,
        on_event: Callable[[Any], None] | None = None,
        on_origin: Callable[[ReplayOrigin], None] | None = None,
        call_replay: tuple[CallReplay, ...] = (),
        native_chat: bool = False,
    ) -> tuple[Any, dict, ProviderAttempt, dict]:
        """Create a streaming chat completion with bounded request fallbacks."""
        execution = request_recovery.execution
        body = self._apply_learned_output_cap(body)
        if corrections is None:
            corrections = RequestCorrections("chat")

        while execution.can_attempt:
            attempt = await execution.open_attempt(operation_kind)
            stream: Any | None = None
            retain_attempt = False
            create_body = body
            validating = False
            try:
                create_body = cast(
                    dict[str, Any],
                    resolve_history_overrides(
                        body
                        if native_chat
                        else self._behavior.prepare_create_body(body)
                    ),
                )
                client = self._client
                if endpoint is not None:
                    assert request_client is not None
                    client = request_client.for_endpoint(
                        self._client, await endpoint.resolve()
                    )
                elif self._api_key_provider is not None:
                    # A borrowed view pins the same credential for provenance and
                    # inference, without owning or closing the provider's HTTP pool.
                    client = self._client.with_options(
                        api_key=await self._api_key_provider()
                    )
                if extra_headers or endpoint is not None:
                    create_body = create_body.copy()
                    create_body["extra_headers"] = {
                        **(create_body.get("extra_headers") or {}),
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
                create_body = cast(
                    dict[str, Any],
                    corrections.prepare_history(
                        create_body,
                        origin,
                        self._history_cache,
                        reasoning_field=self._profile.request_policy.reasoning_replay.value,
                        structured_details=self._profile.structured_reasoning_details,
                        native_chat=native_chat,
                        continuation=operation_kind
                        is ProviderOperationKind.CONTINUATION,
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
                    ),
                )
                if self._validate_prepared_body is not None:
                    validating = True
                    await self._validate_prepared_body(create_body)
                    validating = False

                def observed(event: Any) -> None:
                    if (
                        getattr(event, "choices", None)
                        or getattr(event, "usage", None) is not None
                    ):
                        corrections.accepted_history()
                    if on_event is not None:
                        on_event(event)

                if native_chat:
                    sdk_fields = CompletionCreateParamsStreaming.__annotations__
                    extras = {
                        key: value
                        for key, value in create_body.items()
                        if key not in sdk_fields
                        and key not in {"extra_body", "extra_headers"}
                    }
                    create_body = {
                        key: value
                        for key, value in create_body.items()
                        if key in sdk_fields or key in {"extra_body", "extra_headers"}
                    }
                    if extras:
                        create_body["extra_body"] = {
                            **extras,
                            **create_body.get("extra_body", {}),
                        }
                stream = OpenAIStreamAdapter(
                    await client.chat.completions.create(
                        **create_body,
                        stream=True,
                    ),
                    read_timeout_s=self._read_timeout_s,
                    on_event=observed,
                    transform_event=self._behavior.normalize_chunk,
                )
                if not native_chat:
                    stream = self._behavior.normalize_stream(stream, body)
                retain_attempt = True
                return stream, body, attempt, create_body
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if validating:
                    # No upstream response occurred. Abandon the permit without
                    # teaching correction or recovery state about a rejection.
                    raise
                retry_body = await request_recovery.retry_request(
                    error,
                    classify(error).auth_status,
                    attempt,
                    body,
                    operation_kind=operation_kind,
                    propose_correction=partial(
                        corrections.next_body,
                        error,
                        body,
                        sent_body=create_body,
                        reasoning_error=error,
                        reasoning_sent_body=create_body,
                        after_common=partial(
                            self._next_chat_retry_body,
                            error,
                            body,
                            sent_body=create_body,
                        ),
                    ),
                )
                if retry_body is not None:
                    body = self._apply_learned_output_cap(retry_body)
                    continue
                decision = await attempt.fail(
                    error,
                    provider_failure_override=self._behavior.failure_override,
                )
                if not decision.retry_allowed:
                    if decision.error is not error:
                        raise decision.error from error
                    raise
            finally:
                if not retain_attempt:
                    try:
                        if stream is not None:
                            await close_provider_stream(
                                stream,
                                active_error=sys.exception(),
                                provider_name=self._provider_name,
                                request_id=execution.request_id,
                            )
                    finally:
                        await attempt.aclose()

        if execution.last_failure is not None:
            raise execution.last_failure
        raise RuntimeError("provider execution ended without a final error")

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

        retry_kind = self._behavior.retry_request_kind(error, body, used_retry_kinds)
        if retry_kind is not None and retry_kind not in used_retry_kinds:
            retry_body = self._behavior.retry_request_body(error, body)
            if retry_body is not None:
                used_retry_kinds.add(retry_kind)
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
        validate_history(body)
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
        for key in policy.unsupported_body_keys:
            if key in body:
                raise InvalidRequestError(
                    f"{self._provider_name} cannot represent Gemini control {key}"
                )
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
        self._input_tokens = input_tokens
        self._request_id = request_id
        self._response_model = response_model
        self._reasoning = reasoning
        self._reasoning_correction = reasoning_correction
        self._extra_headers = dict(extra_headers or {})
        self._terminal_failure: ExecutionFailure | None = None
        self._request_client = OpenAIRequestClient(transport._endpoint_transport)
        self._endpoint = (
            RequestEndpoint(endpoint_context) if endpoint_context is not None else None
        )

    async def run(self) -> AsyncIterator[StreamOutput]:
        """Convert the upstream OpenAI-chat stream into Anthropic SSE."""
        model = self._body["model"]
        extra_body = self._body.get("extra_body")
        # The SDK merges allowed extra_body fields over the top-level request.
        if isinstance(extra_body, dict) and isinstance(extra_body.get("model"), str):
            model = extra_body["model"]
        execution = self._transport._admission.start_execution(
            request_id=self._request_id, model=str(model)
        )
        request_recovery = RequestRecovery(execution, endpoint=self._endpoint)
        provider_stream = self._run_execution(request_recovery)
        try:
            async for event in provider_stream:
                yield event
        except asyncio.CancelledError:
            raise
        except Exception as error:
            execution.fail(error)
            raise
        else:
            if self._terminal_failure is None:
                execution.succeed()
            else:
                execution.fail(self._terminal_failure)
        finally:
            try:
                await maybe_await_aclose(provider_stream)
            finally:
                try:
                    await self._request_client.aclose()
                finally:
                    execution.abandon()

    async def _run_execution(
        self,
        request_recovery: RequestRecovery,
    ) -> AsyncIterator[StreamOutput]:
        """Run one provider execution while retaining transport-owned state."""
        tag = self._transport._provider_name
        req_tag = f" request_id={self._request_id}" if self._request_id else ""
        execution = request_recovery.execution

        def packet(event: str) -> Iterator[StreamOutput]:
            if event:
                yield StreamOutput((event,))

        body = resolve_history_overrides(self._body)
        continuation_request = ContinuationRequest("chat")
        operation_kind = ProviderOperationKind.GENERATION
        if self._continuation is not None:
            body = continuation_request.build(body, self._continuation)
            operation_kind = ProviderOperationKind.CONTINUATION
        if not self._native_chat:
            request_stream_usage(body)
        output_reasoning = self._reasoning.output_enabled
        corrections = RequestCorrections("chat", self._reasoning_correction)
        trace_event(
            lambda: {"body": provider_chat_body_snapshot(body)},
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=tag,
            request_id=self._request_id,
            execution_id=execution.execution_id,
            gateway_model=self._response_model,
            downstream_model=body.get("model"),
            message_count=len(body.get("messages", [])),
            tool_count=len(body.get("tools", [])),
        )

        resolver = StreamFailureResolver(
            StreamFailurePolicy(
                provider_name=tag,
                read_timeout_s=self._transport._read_timeout_s,
                request_id=self._request_id,
                failure_override=self._transport._behavior.failure_override,
                trace_error_message=self._transport._log_api_error_tracebacks,
                report_terminal_error=partial(
                    self._transport._log_stream_transport_error,
                    tag,
                    req_tag,
                    request_id=self._request_id,
                ),
            ),
            recovery=request_recovery,
            continuation=continuation_request,
        )
        source: SourceRecoveryState | None = None
        while True:
            presenter = self._new_stream_presenter(output_reasoning=output_reasoning)
            source = SourceRecoveryState(execution, previous=source)
            sent_body = body

            def observe_chunk(
                chunk: Any,
                presenter: _ChatStreamPresenter = presenter,
                source: SourceRecoveryState = source,
                operation_kind: ProviderOperationKind = operation_kind,
            ) -> None:
                presenter.observe_upstream_chunk(chunk)
                if self._native_chat:
                    source.observe(
                        "chat",
                        chunk.model_dump(mode="json", exclude_unset=True),
                        continuing=operation_kind is ProviderOperationKind.CONTINUATION,
                    )
                    return
                source.observe(
                    "chat",
                    {
                        "usage": {
                            key: value
                            for key in (
                                "prompt_tokens",
                                "completion_tokens",
                                "total_tokens",
                            )
                            if (value := usage_int(getattr(chunk, "usage", None), key))
                            is not None
                        },
                        "choices": [
                            {
                                "delta": {
                                    "reasoning_details": getattr(
                                        getattr(choice, "delta", None),
                                        "reasoning_details",
                                        None,
                                    ),
                                    "audio": getattr(
                                        getattr(choice, "delta", None), "audio", None
                                    ),
                                }
                            }
                            for choice in (getattr(chunk, "choices", None) or ())
                        ],
                    },
                    continuing=operation_kind is ProviderOperationKind.CONTINUATION,
                )

            scope: ProviderAttemptScope | None = None
            try:
                stream, body, attempt, sent_body = await self._transport._create_stream(
                    body,
                    request_recovery,
                    operation_kind,
                    corrections=corrections,
                    endpoint=self._endpoint,
                    request_client=self._request_client,
                    extra_headers=self._extra_headers,
                    on_event=observe_chunk,
                    on_origin=partial(setattr, presenter.output, "replay_origin"),
                    call_replay=self._call_replay,
                    native_chat=self._native_chat,
                )
                scope = ProviderAttemptScope(
                    attempt,
                    provider_name=tag,
                    request_id=self._request_id,
                )
                stream = scope.retain(stream)
                presenter.bind_tool_argument_aliases(self._tool_argument_aliases)
                async for chunk in stream:
                    if not scope.attempt.accepted:
                        await scope.attempt.accept()
                    for event in presenter.start_events():
                        for out_event in packet(event):
                            yield out_event
                    for output in presenter.feed(chunk):
                        yield output

                for event in presenter.finish_upstream():
                    for out_event in packet(event):
                        yield out_event
                break

            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as error:
                if not (
                    isinstance(error, ExecutionFailure)
                    and error.stream_context
                    and error.stream_context.source.unsafe_reason
                ):
                    for output in presenter.abandon():
                        yield output
                resolution = await resolver.resolve(
                    StreamFailure(
                        error=error,
                        scope=scope,
                        body=body,
                        operation_kind=operation_kind,
                        normal_stop_seen=presenter.normal_stop_seen,
                        source=source,
                        propose_correction=partial(
                            corrections.next_body,
                            error,
                            body,
                            sent_body=sent_body,
                            reasoning_error=error,
                            reasoning_sent_body=sent_body,
                        ),
                        failure_payload=presenter.output.failure_payload,
                    )
                )
                match resolution:
                    case Retry(body=next_body):
                        body = next_body
                        continue
                    case Continue(body=next_body):
                        body = next_body
                        operation_kind = ProviderOperationKind.CONTINUATION
                        corrections = RequestCorrections(
                            "chat", self._reasoning_correction
                        )
                        continue
                    case Handoff():
                        return
                    case Fail(error=failure):
                        raise failure from error
                    case _:
                        assert_never(resolution)
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())

        for event in presenter.prepare_completion(
            pad_empty=operation_kind is ProviderOperationKind.GENERATION
            and presenter.output.pad_empty
        ):
            for out_event in packet(event):
                yield out_event
        completion = presenter.completion
        if completion.provider_input_tokens is not None:
            logger.debug(
                "TOKEN_ESTIMATE: our={} provider={} diff={:+d}",
                self._input_tokens,
                completion.provider_input_tokens,
                completion.provider_input_tokens - self._input_tokens,
            )
        trace_event(
            stage="provider",
            event="provider.response.completed",
            source="provider",
            provider=tag,
            request_id=self._request_id,
            finish_reason=(
                None
                if completion.finish_reason is None
                else str(completion.finish_reason)
            ),
            output_tokens=completion.output_tokens,
            prompt_tokens=completion.input_tokens,
            prompt_tokens_estimate=self._input_tokens,
        )
        usage = ChatStreamUsage(
            finish_reason=str(completion.finish_reason)
            if completion.finish_reason is not None
            else None,
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
                        self._transport._behavior.cached_input_tokens(
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
            cached_tokens=self._transport._behavior.cached_input_tokens(
                presenter.usage_info
            )
            or 0,
            cache_write_tokens=self._transport._behavior.cache_write_input_tokens(
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
            anthropic_fields=self._transport._behavior.anthropic_usage_fields(
                presenter.usage_info
            ),
        )
        for event in presenter.terminal_events(usage=usage):
            for out_event in packet(event):
                yield out_event

    def _new_stream_presenter(self, *, output_reasoning: bool) -> _ChatStreamPresenter:
        def extra_reasoning_events(
            delta: Any, output: ChatStreamOutput
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
