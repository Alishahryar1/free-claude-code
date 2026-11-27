"""Application workflow for supported local Anthropic web-tool requests."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from urllib.parse import urlparse, urlsplit

from loguru import logger

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor, TokenCounter
from free_claude_code.application.response_collection import collect_response_packets
from free_claude_code.application.routing import RoutedMessagesRequest
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    ContentBlockServerToolUse,
    aggregate_anthropic_sse_to_message,
    anthropic_status_for_error_type,
)
from free_claude_code.core.anthropic.server_tool_sse import (
    ServerToolResponseContext,
    server_tool_completion_frames,
    server_tool_finish_frames,
    server_tool_message_start_frames,
    server_tool_result_frames,
    server_tool_start_frames,
    server_tool_use_frames,
    web_fetch_result_block,
    web_search_result_block,
    web_tool_error_block,
)
from free_claude_code.core.anthropic.stream_contracts import SSEEvent
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.gemini.web_search_replay import signed_gemini_turn
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_output import (
    CompletedCall,
    StreamOutput,
    call_references,
)
from free_claude_code.core.trace import trace_event
from free_claude_code.core.web_tools import WebSearchResult

from .ports import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
    WebToolsPort,
    web_fetch_allowed_scheme_set,
)
from .request import (
    AutomaticWebSearchPlan,
    WebSearchDomainFilter,
    extract_query,
    extract_url,
    forced_server_tool_name,
    forced_tool_turn_text,
    plan_automatic_web_search,
    unsupported_server_tool_error,
    web_tool_limits,
)


class WebToolService:
    """Coordinate one tool request using the originating settings and executor."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: WebToolsPort,
        executor: ProviderExecutor,
        token_counter: TokenCounter,
    ) -> None:
        self._client = client
        self._executor = executor
        self._token_counter = token_counter
        self._enabled = settings.enable_web_server_tools
        self._verbose_client_errors = settings.log_api_error_tracebacks
        self._egress = WebFetchEgressPolicy(
            allow_private_network_targets=settings.web_fetch_allow_private_networks,
            allowed_schemes=web_fetch_allowed_scheme_set(
                settings.web_fetch_allowed_schemes
            ),
        )

    def try_stream_messages(
        self, routed: RoutedMessagesRequest, *, request_id: str
    ) -> AsyncIterator[StreamOutput] | None:
        """Validate synchronously; return a lazy body only for handled requests."""
        request = routed.request
        plan = plan_automatic_web_search(request, web_tools_enabled=self._enabled)
        if plan is None:
            error = unsupported_server_tool_error(
                request, web_tools_enabled=self._enabled
            )
            if error is not None:
                raise InvalidRequestError(error)
        tool_name = forced_server_tool_name(request)
        if plan is None and tool_name is None:
            return None
        input_tokens = self._token_counter(
            request.messages, request.system, request.tools
        )
        if plan is not None:
            return self._stream_automatic_search(
                routed, plan, request_id=request_id, fallback_input_tokens=input_tokens
            )
        assert tool_name is not None
        trace_event(
            stage="routing",
            event="free_claude_code.api.optimization.web_server_tool",
            source="api",
            model=routed.resolved.original_model,
        )
        tool = next(tool for tool in request.named_tools if tool.name == tool_name)
        max_uses, domains = web_tool_limits(tool)
        if tool_name == "web_fetch" and (domains.allowed or domains.blocked):
            raise InvalidRequestError(
                "Local web_fetch cannot enforce domain filters across redirects"
            )
        if max_uses is not None:
            used = 0
            for message in reversed(request.messages):
                blocks = message.content
                if message.role == "user" and not (
                    isinstance(blocks, list)
                    and blocks
                    and blocks[0].type == "tool_result"
                ):
                    break
                if isinstance(blocks, list):
                    used += sum(
                        isinstance(block, ContentBlockServerToolUse)
                        and block.name == tool_name
                        for block in blocks
                    )
            if used >= max_uses:
                raise InvalidRequestError(f"{tool_name}.max_uses has been reached")
        text = forced_tool_turn_text(request)
        if not text.strip():
            raise InvalidRequestError(
                f"The latest user message has no text for {tool_name}"
            )
        tool_input = (
            {"query": extract_query(text)}
            if tool_name == "web_search"
            else {"url": extract_url(text)}
        )
        return self._stream_local_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            input_tokens=input_tokens,
            response_model=routed.resolved.original_model,
            domains=domains,
        )

    async def _stream_automatic_search(
        self,
        routed: RoutedMessagesRequest,
        plan: AutomaticWebSearchPlan,
        *,
        request_id: str,
        fallback_input_tokens: int,
    ) -> AsyncIterator[StreamOutput]:
        """Keep one provider decision private, then replay it or execute search."""
        translated = replace(routed, request=plan.request)
        trace_event(
            stage="routing",
            event="free_claude_code.api.web_search.automatic_recognized",
            source="api",
            request_id=request_id,
            model=routed.resolved.original_model,
        )
        provider_stream = self._executor.stream_messages(
            translated,
            raw_log_payload=plan.request.model_dump,
            request_id=request_id,
        )
        packets = await collect_response_packets(provider_stream, protocol="messages")
        chunks = tuple(frame for packet in packets for frame in packet.frames)
        stop_reason = _provider_stop_reason(chunks)
        references = {ref.call_id: ref for ref in call_references(chunks)}
        if stop_reason in ("refusal", "max_tokens", "stop_sequence", "pause_turn"):
            # Tool arguments from a discarded candidate can be incomplete.
            output = _without_tool_candidates(chunks) if references else iter(packets)
            for packet in output:
                yield packet
            return

        message, stream_error, _complete = await aggregate_anthropic_sse_to_message(
            _iterate_chunks(chunks)
        )
        if stream_error is not None:
            raise _execution_failure_from_stream_error(stream_error)
        tool_calls = _tool_calls(message)
        if tool_calls and stop_reason != "tool_use":
            raise _protocol_failure(
                "Upstream model returned tool candidates without a tool-use stop.",
                request_id=request_id,
            )
        completed = {call for packet in packets for call in packet.completed_calls}
        names = {tool.name for tool in plan.request.direct_client_tools()}
        searches: dict[int, str] = {}
        client_calls: dict[int, CompletedCall] = {}
        seen_ids: set[str] = set()
        for call in tool_calls:
            name = call.get("name")
            call_id = call.get("id")
            if not isinstance(name, str) or name not in names:
                raise _protocol_failure(
                    "Upstream model returned an unexpected tool for automatic WebSearch.",
                    request_id=request_id,
                )
            reference = references.get(call_id) if isinstance(call_id, str) else None
            if not call_id or reference not in completed or call_id in seen_ids:
                raise _protocol_failure(
                    "Upstream model returned an incomplete or ambiguous tool call for automatic WebSearch.",
                    request_id=request_id,
                )
            assert reference is not None and isinstance(call_id, str)
            seen_ids.add(call_id)
            index = reference.position[0]
            if name == plan.private_name:
                searches[index] = _search_query(
                    call.get("input"), request_id=request_id
                )
            else:
                client_calls[index] = reference

        if not tool_calls and (
            stop_reason == "tool_use"
            or (
                (plan.request.tool_choice or {}).get("type") == "any"
                and stop_reason == "end_turn"
            )
        ):
            raise _protocol_failure(
                "Upstream model ended the response without a required tool call.",
                request_id=request_id,
            )
        if not searches:
            trace_event(
                stage="execution",
                event="free_claude_code.api.web_search.automatic_declined",
                source="api",
                request_id=request_id,
                model=routed.resolved.original_model,
            )
            for packet in packets:
                yield packet
            return
        provider_usage = _provider_usage(message)
        message_content: list[JsonObject] = message["content"]
        # A turn can finish provider-hosted work alongside a local selection.
        # Keep its blocks and ordering even when no client function is pending.
        replay_provider_turn = bool(client_calls) or any(
            block.get("type")
            not in ("text", "thinking", "redacted_thinking", "tool_use")
            for block in message_content
        )
        preserve_signed_turn = (
            bool(client_calls) and signed_gemini_turn(message_content) is not None
        )
        input_tokens = _integer_field(provider_usage, "input_tokens")
        if input_tokens is None:
            input_tokens = fallback_input_tokens

        trace_event(
            stage="execution",
            event="free_claude_code.api.web_search.automatic_selected",
            source="api",
            request_id=request_id,
            model=routed.resolved.original_model,
        )
        message_id = f"msg_{uuid.uuid4()}"
        contexts = [
            ServerToolResponseContext(
                message_id=message_id,
                tool_id=f"srvtoolu_{uuid.uuid4().hex}",
                model=routed.resolved.original_model,
                tool_name="web_search",
                tool_input={"query": query},
                input_tokens=input_tokens,
                provider_usage=provider_usage,
            )
            for query in searches.values()
        ]
        executed = (
            len(contexts)
            if plan.remaining_uses is None
            else min(len(contexts), plan.remaining_uses)
        )
        if not replay_provider_turn:
            for frame in server_tool_message_start_frames(contexts[0]):
                yield StreamOutput((frame,))

        limit = asyncio.Semaphore(4)

        async def search_one(
            context: ServerToolResponseContext,
        ) -> tuple[JsonObject, str]:
            async with limit:
                return await self._search_result(context, plan.domains)

        summaries: list[str] = []
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(search_one(context))
                for context in contexts[:executed]
            ]

            async def search_frames(
                slot: int, index: int
            ) -> AsyncIterator[StreamOutput]:
                context = contexts[slot]
                for frame in server_tool_use_frames(context, index=index):
                    yield StreamOutput((frame,))
                if slot < executed:
                    result_block, summary = await tasks[slot]
                else:
                    result_block = web_tool_error_block(
                        context, error_code="max_uses_exceeded"
                    )
                    summary = (
                        f"Web search limit reached for: {context.tool_input['query']}"
                    )
                for frame in server_tool_result_frames(result_block, index=index + 1):
                    yield StreamOutput((frame,))
                summaries.append(summary)

            if client_calls and not preserve_signed_turn:
                # Keep search results ahead of calls awaiting a client reply.
                first_call = min(searches.keys() | client_calls.keys())
                schedule = {first_call: tuple(range(len(contexts)))}
            else:
                schedule = {index: (slot,) for slot, index in enumerate(searches)}
            indices: dict[int, int] = {}
            next_index = 0
            for event in _messages_events(chunks):
                data = event.data
                kind = data.get("type")
                if kind in {"message_delta", "message_stop"}:
                    continue
                index = data.get("index")
                if isinstance(index, int) and kind == "content_block_start":
                    for slot in schedule.get(index, ()):
                        async for packet in search_frames(slot, next_index):
                            yield packet
                        next_index += 2
                if not replay_provider_turn or (
                    isinstance(index, int) and index in searches
                ):
                    continue
                proof = ()
                if isinstance(index, int):
                    if kind == "content_block_start":
                        indices[index] = next_index
                        next_index += 1
                        if index in client_calls:
                            proof = (
                                replace(
                                    client_calls[index], position=(indices[index],)
                                ),
                            )
                    data["index"] = indices.get(index, index)
                yield StreamOutput((format_sse_event(event.event, data),), proof)

        for frame in server_tool_finish_frames(
            contexts[0],
            summary=None if client_calls else "\n\n".join(summaries),
            index=next_index,
            request_count=executed,
            stop_reason="tool_use" if client_calls else "end_turn",
        ):
            yield StreamOutput((frame,))
        trace_event(
            stage="execution",
            event="free_claude_code.api.web_search.automatic_completed",
            source="api",
            request_id=request_id,
            model=routed.resolved.original_model,
        )

    async def _stream_local_tool(
        self,
        *,
        tool_name: str,
        tool_input: dict[str, str],
        input_tokens: int,
        response_model: str,
        provider_usage: Mapping[str, object] | None = None,
        domains: WebSearchDomainFilter | None = None,
    ) -> AsyncIterator[StreamOutput]:
        context = ServerToolResponseContext(
            message_id=f"msg_{uuid.uuid4()}",
            tool_id=f"srvtoolu_{uuid.uuid4().hex}",
            model=response_model,
            tool_name=tool_name,
            tool_input=tool_input,
            input_tokens=input_tokens,
            provider_usage=provider_usage,
        )
        for frame in server_tool_start_frames(context):
            yield StreamOutput((frame,))

        if tool_name == "web_search":
            result_block, summary = await self._search_result(context, domains)
        else:
            try:
                fetched = await self._client.fetch(
                    tool_input["url"], egress=self._egress
                )
                result_block = web_fetch_result_block(
                    context, fetched, retrieved_at=datetime.now(UTC).isoformat()
                )
                summary = fetched.data
            except Exception as error:
                _log_web_tool_failure(tool_name, error, fetch_url=tool_input["url"])
                result_block = web_tool_error_block(context)
                summary = _web_tool_client_error_summary(
                    tool_name, error, verbose=self._verbose_client_errors
                )

        for frame in server_tool_completion_frames(
            context, result_block, summary=summary
        ):
            yield StreamOutput((frame,))

    async def _search_result(
        self,
        context: ServerToolResponseContext,
        domains: WebSearchDomainFilter | None,
    ) -> tuple[JsonObject, str]:
        query = context.tool_input["query"]
        try:
            results = await self._client.search(query)
            if domains is not None:
                results = _filter_results(results, domains)
            return web_search_result_block(context, results), _search_summary(
                query, results
            )
        except Exception as error:
            _log_web_tool_failure("web_search", error, fetch_url=None)
            return web_tool_error_block(context), _web_tool_client_error_summary(
                "web_search", error, verbose=self._verbose_client_errors
            )


def _search_summary(query: str, results: list[WebSearchResult]) -> str:
    if not results:
        return f"No web search results found for: {query}"
    lines = [f"Search results for: {query}"]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}\n{result.url}")
    return "\n\n".join(lines)


async def _iterate_chunks(chunks: tuple[str, ...]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


def _messages_events(
    chunks: Iterable[str], *, event_names: frozenset[str] | None = None
) -> Iterator[SSEEvent]:
    decoder = AnthropicSSEDecoder(event_names=event_names)
    for chunk in chunks:
        yield from decoder.feed(chunk)


def _provider_stop_reason(chunks: Iterable[str]) -> str | None:
    reason = None
    for event in _messages_events(chunks, event_names=frozenset({"message_delta"})):
        delta = event.data.get("delta")
        if isinstance(delta, dict) and isinstance(
            stop := delta.get("stop_reason"), str
        ):
            reason = stop
    return reason


def _without_tool_candidates(chunks: Iterable[str]) -> Iterator[StreamOutput]:
    """Keep the provider's terminal and discard calls it did not finish with tool_use."""
    kept_indices: dict[int, int] = {}
    for event in _messages_events(chunks):
        data = event.data
        index = data.get("index")
        if isinstance(index, int):
            if data.get("type") == "content_block_start":
                block = data.get("content_block", {})
                # This carrier describes a native Gemini turn that still has
                # the discarded function calls in it.
                if isinstance(block, dict) and (
                    block.get("type") == "tool_use" or signed_gemini_turn((block,))
                ):
                    continue
                kept_indices[index] = len(kept_indices)
            if index not in kept_indices:
                continue
            data["index"] = kept_indices[index]
        yield StreamOutput((format_sse_event(event.event, data),))


def _tool_calls(message: Mapping[str, object]) -> list[dict[str, object]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    calls: list[dict[str, object]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        calls.append({str(key): value for key, value in block.items()})
    return calls


def _search_query(arguments: object, *, request_id: str) -> str:
    if not isinstance(arguments, dict) or set(arguments) != {"query"}:
        raise _protocol_failure(
            "Upstream model returned malformed arguments for automatic WebSearch.",
            request_id=request_id,
        )
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise _protocol_failure(
            "Upstream model returned an empty query for automatic WebSearch.",
            request_id=request_id,
        )
    return query.strip()


def _provider_usage(message: Mapping[str, object]) -> dict[str, object]:
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {str(key): value for key, value in usage.items()}


def _integer_field(values: Mapping[str, object], key: str) -> int | None:
    value = values.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _filter_results(
    results: list[WebSearchResult],
    domains: WebSearchDomainFilter,
) -> list[WebSearchResult]:
    if not domains.allowed and not domains.blocked:
        return results
    return [result for result in results if _hostname_is_allowed(result.url, domains)]


def _hostname_is_allowed(url: str, domains: WebSearchDomainFilter) -> bool:
    hostname = urlsplit(url).hostname
    if hostname is None:
        return False
    try:
        normalized = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return False
    if domains.allowed and not any(
        _is_hostname_or_subdomain(normalized, domain) for domain in domains.allowed
    ):
        return False
    return not any(
        _is_hostname_or_subdomain(normalized, domain) for domain in domains.blocked
    )


def _is_hostname_or_subdomain(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith(f".{domain}")


def _execution_failure_from_stream_error(
    error: Mapping[str, object],
) -> ExecutionFailure:
    raw_type = error.get("type")
    error_type = raw_type if isinstance(raw_type, str) else "api_error"
    raw_message = error.get("message")
    message = (
        raw_message
        if isinstance(raw_message, str) and raw_message.strip()
        else "Provider request failed unexpectedly."
    )
    status = anthropic_status_for_error_type(error_type)
    kind = {
        400: FailureKind.INVALID_REQUEST,
        401: FailureKind.AUTHENTICATION,
        402: FailureKind.PERMISSION,
        403: FailureKind.PERMISSION,
        404: FailureKind.INVALID_REQUEST,
        413: FailureKind.INVALID_REQUEST,
        429: FailureKind.RATE_LIMIT,
        504: FailureKind.TIMEOUT,
        529: FailureKind.OVERLOADED,
    }.get(status, FailureKind.UPSTREAM)
    return ExecutionFailure(
        kind=kind,
        status_code=status,
        message=message,
        retryable=False,
    )


def _protocol_failure(message: str, *, request_id: str) -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.UPSTREAM,
        status_code=500,
        message=f"{message}\n\nRequest ID: {request_id}",
        retryable=False,
    )


def _safe_public_host_for_logs(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host[:253]


def _log_web_tool_failure(
    tool_name: str,
    error: BaseException,
    *,
    fetch_url: str | None = None,
) -> None:
    exc_type = type(error).__name__
    if isinstance(error, WebFetchEgressViolation):
        host = _safe_public_host_for_logs(fetch_url) if fetch_url else ""
        logger.warning(
            "web_tool_egress_rejected tool={} exc_type={} host={!r}",
            tool_name,
            exc_type,
            host,
        )
        return
    if tool_name == "web_fetch" and fetch_url:
        logger.warning(
            "web_tool_failure tool={} exc_type={} host={!r}",
            tool_name,
            exc_type,
            _safe_public_host_for_logs(fetch_url),
        )
    else:
        logger.warning("web_tool_failure tool={} exc_type={}", tool_name, exc_type)


def _web_tool_client_error_summary(
    tool_name: str,
    error: BaseException,
    *,
    verbose: bool,
) -> str:
    if verbose:
        return f"{tool_name} failed: {type(error).__name__}"
    return "Web tool request failed."
