"""Native Messages execution using the common admission and attempt owners."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace

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
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    classify_provider_failure,
)
from free_claude_code.providers.http import ProviderAttemptScope

from .wire import check_messages_failure, messages_events, messages_status_error


async def stream_native_messages(
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
) -> AsyncIterator[str]:
    execution = admission.start_execution(request_id=request_id)
    streaming = body.get("stream", False) is True

    async def complete() -> str:
        async with client.stream(
            "POST", base_url.rstrip("/") + "/messages", json=body, headers=headers
        ) as response:
            if not response.is_success:
                raise await messages_status_error(response)
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
            return json.dumps({**message, "model": public_model}, ensure_ascii=False)

    try:
        if not streaming:
            yield await execution.run_call(
                complete, operation_kind=ProviderOperationKind.GENERATION
            )
            return
        committed = False
        while execution.can_attempt:
            scope = None
            try:
                attempt = await execution.open_attempt(ProviderOperationKind.GENERATION)
                scope = ProviderAttemptScope(
                    attempt, provider_name=provider_name, request_id=request_id
                )
                response = scope.retain(
                    await client.send(
                        client.build_request(
                            "POST",
                            base_url.rstrip("/") + "/messages",
                            json=body,
                            headers=headers,
                        ),
                        stream=True,
                    )
                )
                if not response.is_success:
                    raise await messages_status_error(response)
                if (
                    "text/event-stream"
                    not in response.headers.get("content-type", "").lower()
                ):
                    raise RetryableProviderProtocolError(
                        "Messages upstream did not return an SSE stream."
                    )
                relay = NativeMessagesPassthrough(public_model)
                async for kind, payload in messages_events(response):
                    check_messages_failure(kind, payload, native=True)
                    output = relay.feed(kind, payload)
                    if output is None:
                        continue
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
                if scope is not None:
                    decision = await scope.attempt.fail(error)
                    if not committed and decision.retry_allowed:
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
        execution.fail(failure)
        raise failure from error
    finally:
        execution.abandon()
