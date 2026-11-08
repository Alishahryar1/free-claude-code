"""Native Messages execution using the common admission and attempt owners."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from dataclasses import replace
from functools import partial

import httpx

from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    validate_messages_json,
)
from free_claude_code.core.anthropic.passthrough import NativeMessagesPassthrough
from free_claude_code.core.diagnostics import (
    extract_upstream_error_detail,
    format_execution_failure_message,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.continuation import (
    ContinuationRequest,
    SourceRecoveryState,
    public_recovery,
    public_stream_failure,
)
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    classify_provider_failure,
    is_retryable_stream_error,
)
from free_claude_code.providers.history_negotiation import HistoryModeCache
from free_claude_code.providers.history_replay import replay_origin
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.request_recovery import (
    RequestCorrections,
    RequestRecovery,
)
from free_claude_code.providers.stream_recovery import can_retry_undelivered_stream

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
) -> AsyncIterator[str]:
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
        committed = False
        continuation_request = ContinuationRequest("messages")
        operation_kind = ProviderOperationKind.GENERATION
        if continuation is not None:
            body = continuation_request.build(body, continuation)
            operation_kind = ProviderOperationKind.CONTINUATION
        while execution.can_attempt:
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
                    ):
                        raise RetryableProviderProtocolError(
                            "Messages upstream did not return a Message object."
                        )
                    corrections.accepted_history()
                    await attempt.accept()
                    execution.succeed()
                    yield json.dumps(
                        {**message, "model": public_model}, ensure_ascii=False
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
                        output = relay.feed(kind, payload)
                        if output is None:
                            continue
                        corrections.accepted_history()
                        if not attempt.accepted:
                            await attempt.accept()
                        committed = True
                        yield output
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
                error = (
                    RetryableProviderProtocolError(str(raw_error))
                    if isinstance(raw_error, NativeMessagesError)
                    else raw_error
                )
                if scope is not None and (
                    not committed or execution.delivery is not None
                ):
                    corrected = await request_recovery.retry_request(
                        error,
                        None,
                        scope.attempt,
                        body,
                        operation_kind=operation_kind,
                        normal_stop_seen=normal_stop_seen,
                        propose_correction=partial(
                            corrections.next_body,
                            raw_error,
                            body,
                            sent_body=sent_body,
                            reasoning_error=error,
                        ),
                    )
                    if corrected is not None:
                        body = corrected
                        continue
                if scope is not None:
                    hidden_stop = (
                        normal_stop_seen
                        and execution.delivery is not None
                        and not execution.delivery.content_released
                    )
                    if hidden_stop:
                        await scope.attempt.accept()
                    else:
                        decision = await scope.attempt.fail(error)
                        if (
                            not committed and decision.retry_allowed
                        ) or can_retry_undelivered_stream(
                            execution.delivery,
                            retryable=is_retryable_stream_error(error),
                            attempts_remaining=execution.attempts_remaining,
                            normal_stop_seen=normal_stop_seen,
                        ):
                            continue
                if scope is not None:
                    await scope.aclose(active_error=error)
                recovered = public_recovery(
                    execution,
                    body=body,
                    request=continuation_request,
                    retryable=is_retryable_stream_error(error),
                    normal_stop_seen=normal_stop_seen,
                    source=source,
                )
                if recovered is not None:
                    if recovered.body is None:
                        execution.succeed()
                        return
                    body = recovered.body
                    corrections = RequestCorrections("messages")
                    operation_kind = ProviderOperationKind.CONTINUATION
                    continue
                if error is not raw_error:
                    raise error from raw_error
                raise
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
            failure = classify_provider_failure(
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
