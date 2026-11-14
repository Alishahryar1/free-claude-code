"""Native Messages execution using the common admission and attempt owners."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from dataclasses import replace
from functools import partial
from typing import assert_never

import httpx

from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    validate_messages_json,
)
from free_claude_code.core.anthropic.passthrough import NativeMessagesPassthrough
from free_claude_code.core.anthropic.tool_calls import (
    MessagesToolCallBuffer,
    validate_tool_identity,
)
from free_claude_code.core.diagnostics import (
    extract_upstream_error_detail,
    format_execution_failure_message,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_output import StreamOutput, completed_output
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.continuation import (
    ContinuationRequest,
    SourceRecoveryState,
    public_stream_failure,
)
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.failure_reporting import finalize_provider_failure
from free_claude_code.providers.history_negotiation import HistoryModeCache
from free_claude_code.providers.history_replay import replay_origin
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.recovery_errors import RetryableProviderProtocolError
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

from .wire import (
    check_messages_failure,
    is_messages_stop,
    messages_events,
    messages_status_error,
)


async def stream_native_messages(
    client: httpx.AsyncClient,
    admission: ProviderAdmissionController,
    *,
    history_cache: HistoryModeCache | None = None,
    base_url: str,
    headers: Mapping[str, str],
    body: JsonObject,
    public_model: str,
    provider_name: str,
    read_timeout_s: float,
    request_id: str | None,
    continuation: ContinuationSeed | None = None,
) -> AsyncIterator[StreamOutput]:
    execution = admission.start_execution(
        request_id=request_id, model=str(body["model"])
    )
    streaming = body.get("stream", False) is True
    source = SourceRecoveryState(execution)
    normal_stop_seen = False

    cache = history_cache if history_cache is not None else HistoryModeCache()
    corrections = RequestCorrections("messages")
    request_recovery = RequestRecovery(execution)
    endpoint = HttpEndpoint(base_url=base_url, api_key=None, headers=dict(headers))

    try:
        continuation_request = ContinuationRequest("messages")
        operation_kind = ProviderOperationKind.GENERATION
        if continuation is not None:
            body = continuation_request.build(body, continuation)
            operation_kind = ProviderOperationKind.CONTINUATION
        resolver = StreamFailureResolver(
            StreamFailurePolicy(
                provider_name=provider_name,
                read_timeout_s=read_timeout_s,
                request_id=request_id,
                # The outer handler finalizes with the native error detail.
                finalize=False,
            ),
            recovery=request_recovery,
            continuation=continuation_request,
        )
        while execution.can_attempt:
            relay = None
            tool_buffer = MessagesToolCallBuffer()
            normal_stop_seen = False
            source = SourceRecoveryState(execution, previous=source)
            scope = None
            sent_body = body
            try:
                attempt = await execution.open_attempt(operation_kind)
                scope = ProviderAttemptScope(
                    attempt, provider_name=provider_name, request_id=request_id
                )
                origin = replay_origin(
                    provider_name, "messages", str(body["model"]), endpoint=endpoint
                )
                sent_body = corrections.prepare_history(
                    body,
                    origin,
                    cache,
                    continuation=operation_kind is ProviderOperationKind.CONTINUATION,
                )
                response = scope.retain(
                    await client.send(
                        client.build_request(
                            "POST",
                            base_url.rstrip("/") + "/messages",
                            json=sent_body,
                            headers=headers,
                        ),
                        stream=True,
                    )
                )
                if not response.is_success:
                    raise await messages_status_error(response)
                if not streaming:
                    await response.aread()
                    try:
                        message = response.json()
                        validate_messages_json(message)
                    except (ValueError, NativeMessagesError) as error:
                        raise RetryableProviderProtocolError(
                            "Messages upstream returned invalid JSON."
                        ) from error
                    if (
                        not isinstance(message, dict)
                        or message.get("type") != "message"
                        or message.get("stop_reason") is None
                    ):
                        raise RetryableProviderProtocolError(
                            "Messages upstream did not return a Message object."
                        )
                    normal_stop_seen = True
                    if message["stop_reason"] not in {
                        "tool_use",
                        "end_turn",
                        "stop_sequence",
                    }:
                        message["content"] = [
                            block
                            for block in message.get("content", [])
                            if block.get("type") != "tool_use"
                        ]
                    call_ids: set[str] = set()
                    for block in message.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            call_ids.add(validate_tool_identity(block, call_ids))
                    corrections.accepted_history()
                    await attempt.accept()
                    execution.succeed()
                    yield StreamOutput(
                        (
                            json.dumps(
                                {**message, "model": public_model}, ensure_ascii=False
                            ),
                        )
                    )
                    return
                if (
                    "text/event-stream"
                    not in response.headers.get("content-type", "").lower()
                ):
                    raise RetryableProviderProtocolError(
                        "Messages upstream did not return an SSE stream."
                    )
                relay = NativeMessagesPassthrough(public_model)
                tool_buffer = MessagesToolCallBuffer()
                async with aclosing(
                    messages_events(response, read_timeout_s=read_timeout_s)
                ) as events:
                    async for kind, payload in events:
                        normal_stop_seen |= is_messages_stop(kind, payload)
                        check_messages_failure(kind, payload, native=True)
                        source.observe(
                            "messages",
                            payload,
                            continuing=operation_kind
                            is ProviderOperationKind.CONTINUATION,
                        )
                        corrections.accepted_history()
                        if not attempt.accepted:
                            await attempt.accept()
                        for group in tool_buffer.feed(kind, payload):
                            yield completed_output(
                                filter(
                                    None,
                                    (
                                        relay.feed(event, value)
                                        for event, value in group
                                    ),
                                )
                            )
                        if relay.completed:
                            break
                if not relay.completed:
                    raise RetryableProviderProtocolError(
                        "Messages stream ended without message_stop."
                    )
                execution.succeed()
                return
            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as raw_error:
                if relay is not None and not (
                    isinstance(raw_error, ExecutionFailure)
                    and raw_error.stream_context
                    and raw_error.stream_context.source.unsafe_reason
                ):
                    for group in tool_buffer.abandon():
                        yield completed_output(
                            filter(
                                None,
                                (relay.feed(event, value) for event, value in group),
                            )
                        )
                error = (
                    RetryableProviderProtocolError(str(raw_error))
                    if isinstance(raw_error, NativeMessagesError)
                    else raw_error
                )
                resolution = await resolver.resolve(
                    StreamFailure(
                        error=error,
                        scope=scope,
                        body=body,
                        operation_kind=operation_kind,
                        normal_stop_seen=normal_stop_seen,
                        source=source,
                        propose_correction=partial(
                            corrections.next_body,
                            raw_error,
                            body,
                            sent_body=sent_body,
                            reasoning_error=error,
                        ),
                    )
                )
                match resolution:
                    case Retry(body=next_body):
                        body = next_body
                        continue
                    case Continue(body=next_body):
                        body = next_body
                        corrections = RequestCorrections("messages")
                        operation_kind = ProviderOperationKind.CONTINUATION
                        continue
                    case Handoff():
                        execution.succeed()
                        return
                    case Fail(error=failure) if failure is raw_error:
                        raise
                    case Fail(error=failure):
                        raise failure from raw_error
                    case _:
                        assert_never(resolution)
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())
        raise RuntimeError("Messages execution ended without a terminal result.")
    except asyncio.CancelledError, GeneratorExit:
        raise
    except Exception as error:
        if isinstance(error, ExecutionFailure):
            # Stream errors carry their native evidence before classification.
            failure = replace(
                error,
                message=format_execution_failure_message(
                    error,
                    extract_upstream_error_detail(error),
                    upstream_name=provider_name,
                    request_id=request_id,
                ),
            )
        else:
            failure = finalize_provider_failure(
                error,
                provider_name=provider_name,
                read_timeout_s=read_timeout_s,
                request_id=request_id,
            )
        if streaming and execution.delivery is not None:
            failure = public_stream_failure(
                failure,
                source=source,
                body=body,
                protocol="messages",
                normal_stop_seen=normal_stop_seen,
            )
        execution.fail(failure)
        raise failure from error
    finally:
        execution.abandon()
