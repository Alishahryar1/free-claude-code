"""Application workflow for supported local Anthropic web-tool requests."""

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from urllib.parse import urlparse, urlsplit

from loguru import logger

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor, TokenCounter
from free_claude_code.application.routing import RoutedMessagesRequest
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
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
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.trace import close_stream_input, trace_event
from free_claude_code.core.web_tools import WebSearchResult

from .ports import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
    WebToolsPort,
    web_fetch_allowed_scheme_set,
)
from .request import (
    HIDDEN_WEB_SEARCH_NAME,
    AutomaticWebSearchPlan,
    WebSearchDomainFilter,
    extract_query,
    extract_url,
    forced_server_tool_name,
    forced_tool_turn_text,
    plan_automatic_web_search,
    unsupported_server_tool_error,
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
    ) -> AsyncIterator[str] | None:
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
        text = forced_tool_turn_text(request)
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
        )

    async def _stream_automatic_search(
        self,
        routed: RoutedMessagesRequest,
        plan: AutomaticWebSearchPlan,
        *,
        request_id: str,
        fallback_input_tokens: int,
    ) -> AsyncIterator[str]:
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
            raw_log_payload=plan.request.model_dump(),
            request_id=request_id,
        )
        chunks: list[str] = []
        try:
            chunks.extend([chunk async for chunk in provider_stream])
        finally:
            await close_stream_input(
                provider_stream,
                owner="automatic_web_search",
                source="api",
                preserved_error=sys.exception(),
            )

        message, stream_error, _complete = await aggregate_anthropic_sse_to_message(
            _iterate_chunks(chunks)
        )
        if stream_error is not None:
            raise _execution_failure_from_stream_error(stream_error)
        tool_calls = _tool_calls(message)
        if not tool_calls:
            trace_event(
                stage="execution",
                event="free_claude_code.api.web_search.automatic_declined",
                source="api",
                request_id=request_id,
                model=routed.resolved.original_model,
            )
            for chunk in chunks:
                yield chunk
            return
        queries: list[str] = []
        for call in tool_calls:
            if call.get("name") != HIDDEN_WEB_SEARCH_NAME:
                raise _protocol_failure(
                    "Upstream model returned an unexpected tool for automatic WebSearch.",
                    request_id=request_id,
                )
            arguments = call.get("input")
            if not isinstance(arguments, dict) or set(arguments) != {"query"}:
                raise _protocol_failure(
                    "Upstream model returned malformed arguments for automatic WebSearch.",
                    request_id=request_id,
                )
            raw_query = arguments.get("query")
            if not isinstance(raw_query, str) or not raw_query.strip():
                raise _protocol_failure(
                    "Upstream model returned an empty query for automatic WebSearch.",
                    request_id=request_id,
                )
            queries.append(raw_query.strip())
        provider_usage = _provider_usage(message)
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
            for query in queries
        ]
        executed = min(len(contexts), plan.max_uses or len(contexts))
        for frame in server_tool_message_start_frames(contexts[0]):
            yield frame

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
            for index, context in enumerate(contexts):
                for frame in server_tool_use_frames(context, index=index * 2):
                    yield frame
                if index < executed:
                    result_block, summary = await tasks[index]
                else:
                    result_block = web_tool_error_block(
                        context, error_code="max_uses_exceeded"
                    )
                    summary = (
                        f"Web search limit reached for: {context.tool_input['query']}"
                    )
                for frame in server_tool_result_frames(
                    result_block, index=index * 2 + 1
                ):
                    yield frame
                summaries.append(summary)

        for frame in server_tool_finish_frames(
            contexts[0],
            summary="\n\n".join(summaries),
            index=len(contexts) * 2,
            request_count=executed,
        ):
            yield frame
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
    ) -> AsyncIterator[str]:
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
            yield frame

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
            yield frame

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


async def _iterate_chunks(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


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
