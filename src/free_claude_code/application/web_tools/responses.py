"""Application-owned Codex text web workflow over a bound provider route."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from urllib.parse import urlsplit

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.responses_execution import (
    ExecutionChunk,
    ExecutionProgress,
    ResponsesBinding,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    WebResponsePresenter,
    WebSearchSpec,
    encoded_records,
    prepare_web_history,
    prepare_web_request,
    public_action,
    replay_item,
    validate_action,
)
from free_claude_code.core.sse import SSEDecoder
from free_claude_code.core.token_estimation import estimate_text_tokens
from free_claude_code.core.trace import close_stream_input
from free_claude_code.core.web_domains import domain_matches

from .ports import WebFetchEgressPolicy, WebToolsPort

_ACTION_TIMEOUT = 20.0


class ResponsesWebTools:
    def __init__(
        self,
        client: WebToolsPort | None = None,
        *,
        enabled: bool = True,
        egress: WebFetchEgressPolicy = WebFetchEgressPolicy(
            False, frozenset({"http", "https"})
        ),
    ) -> None:
        self._client = client
        self._enabled = enabled
        self._egress = egress

    async def stream(
        self,
        bound: ResponsesBinding,
        request: OpenAIResponsesRequest,
        *,
        token_counter: Callable[[OpenAIResponsesRequest], int],
    ) -> AsyncIterator[ExecutionChunk]:
        presenter: WebResponsePresenter | None = None
        records: list[JsonObject] = []
        try:
            history = prepare_web_history(request)
            working = history.request
            spec = None
            if bound.egress != "responses":
                working, spec = prepare_web_request(working)
            if spec is None:
                stream = bound.stream(working, input_tokens=token_counter(working))
                try:
                    async for frame in stream:
                        yield frame
                finally:
                    await close_stream_input(
                        stream,
                        owner="responses_web",
                        source="application",
                        preserved_error=sys.exception(),
                    )
                return
            if working.tool_choice != "none" and (
                not self._enabled or self._client is None
            ):
                raise InvalidRequestError(
                    "Local web search is disabled or unavailable. Enable local web tools or disable Codex web search."
                )
            presenter = WebResponsePresenter(request, spec)
            pages = _retained_pages(history.records)
            for record in history.records:
                _remember_sources(presenter.sources, record["result"])
            remaining = request.max_output_tokens
            used = 0
            exhausted = False
            while True:
                presenter.begin_turn()
                if remaining is not None:
                    working.max_output_tokens = remaining
                decoder = SSEDecoder()
                stream = bound.stream(working, input_tokens=token_counter(working))
                try:
                    async for chunk in stream:
                        if chunk:
                            yield ExecutionProgress()
                        for event in decoder.feed(chunk):
                            for frame in presenter.feed(event):
                                yield frame
                    for event in decoder.finish():
                        for frame in presenter.feed(event):
                            yield frame
                finally:
                    await close_stream_input(
                        stream,
                        owner="responses_web",
                        source="application",
                        preserved_error=sys.exception(),
                    )
                if presenter.terminal is None:
                    raise ResponsesConversionError(
                        "Provider ended without a terminal response."
                    )
                if presenter.terminal.get("status") != "completed":
                    for slot in presenter.web_slots():
                        for frame in presenter.complete_web(
                            slot, {}, status="incomplete"
                        ):
                            yield frame
                    if records:
                        for frame in presenter.append_replay(replay_item(records)):
                            yield frame
                    yield presenter.finish()
                    return
                private = presenter.private_output()
                if remaining is not None:
                    usage = presenter.terminal.get("usage")
                    count = (
                        usage.get("output_tokens") if isinstance(usage, dict) else None
                    )
                    if (
                        not isinstance(count, int)
                        or isinstance(count, bool)
                        or count < 0
                    ):
                        count = estimate_text_tokens(
                            json.dumps(private, ensure_ascii=False)
                        )
                    remaining = max(0, remaining - count)
                slots = presenter.web_slots()
                ordinary = any(
                    item.get("type")
                    in ("function_call", "custom_tool_call", "tool_search_call")
                    and item not in [slot.private for slot in slots]
                    for item in private
                )
                if slots and not spec.allowed:
                    raise ResponsesConversionError(
                        "Provider called local web despite the selected ordinary/disabled tool choice."
                    )
                outputs: list[JsonValue] = []
                for slot in slots:
                    if exhausted:
                        raise ResponsesConversionError(
                            "Provider called a removed web tool after its budget was exhausted."
                        )
                    # Separate model responses may reuse IDs. Only FCC-owned calls
                    # need new identities in the combined private conversation.
                    call = next(item for item in private if item == slot.private)
                    call["call_id"] = slot.identity
                    call["id"] = "fc_" + slot.identity.removeprefix("ws_")
                    arguments: JsonObject = {}
                    page: JsonObject | None = None
                    if used >= spec.max_calls:
                        result = _error(
                            "budget_exhausted",
                            "The local web action limit has been reached; finish with the available information.",
                        )
                    else:
                        used += 1
                        try:
                            arguments = validate_action(
                                json.loads(str(slot.private.get("arguments", "")))
                            )
                        except (ValueError, TypeError) as exc:
                            result = _error("invalid_arguments", str(exc))
                        else:
                            yield presenter.searching(slot)
                            yield ExecutionProgress("local")
                            try:
                                async with asyncio.timeout(_ACTION_TIMEOUT):
                                    result, page = await self._action(
                                        arguments, spec, pages
                                    )
                            except TimeoutError:
                                result = _error("timeout", "The web action timed out.")
                            except Exception:
                                result = _error(
                                    "lookup_failed",
                                    "The web lookup failed or the page was blocked by the fetch policy.",
                                )
                            yield ExecutionProgress()
                    record: JsonObject = {
                        "id": slot.identity,
                        "action": public_action(arguments) if arguments else {},
                        "result": result,
                    }
                    if page is not None:
                        record["page"] = page
                    try:
                        encoded_records([*records, record])
                    except ResponsesConversionError:
                        result = _error(
                            "context_limit",
                            "The retained web context is full; use the existing results.",
                        )
                        record = {
                            "id": slot.identity,
                            "action": {},
                            "result": result,
                        }
                        # If even the bounded error cannot fit, fail before giving
                        # it to the model; the outer failure retains prior records.
                        encoded_records([*records, record])
                    records.append(record)
                    outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": slot.identity,
                            "output": json.dumps(result, ensure_ascii=False),
                        }
                    )
                    for frame in presenter.complete_web(slot, result):
                        yield frame
                if not slots or ordinary or remaining == 0:
                    if records:
                        for frame in presenter.append_replay(replay_item(records)):
                            yield frame
                    yield presenter.finish(
                        incomplete=remaining == 0 and bool(slots) and not ordinary
                    )
                    return
                previous = (
                    working.input
                    if isinstance(working.input, list)
                    else [{"role": "user", "content": working.input}]
                )
                working = working.model_copy(
                    update={"input": [*previous, *private, *outputs]}, deep=True
                )
                if working.tool_choice == "required" or isinstance(
                    working.tool_choice, dict
                ):
                    working.tool_choice = "auto"
                if used >= spec.max_calls:
                    exhausted = True
                    working.tools = [
                        tool
                        for tool in working.tools or []
                        if tool.get("name") != spec.name
                    ]
        except (ResponsesConversionError, ExecutionFailure, InvalidRequestError) as exc:
            if presenter is None or presenter.response is None:
                if isinstance(exc, ResponsesConversionError):
                    raise InvalidRequestError(str(exc)) from exc
                raise
            failure = (
                exc
                if isinstance(exc, ExecutionFailure)
                else ExecutionFailure(FailureKind.UPSTREAM, 502, str(exc), False)
            )
            for frame in presenter.fail(failure):
                yield frame
            if records:
                for frame in presenter.append_replay(replay_item(records)):
                    yield frame
            yield presenter.finish()

    async def _action(
        self,
        action: JsonObject,
        spec: WebSearchSpec,
        pages: dict[str, JsonObject],
    ) -> tuple[JsonObject, JsonObject | None]:
        if self._client is None:
            raise RuntimeError("No web client")
        if action["action"] == "search":
            found = await self._client.search(str(action["query"]))
            sources: list[JsonValue] = []
            allowance = spec.context_chars
            for result in found[:10]:
                if not _allowed(result.url, spec):
                    continue
                snippet = result.snippet[: max(0, allowance)]
                allowance -= len(snippet)
                sources.append(
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": snippet,
                        "truncated": len(snippet) < len(result.snippet),
                    }
                )
            return {
                "results": sources,
                "sources": sources,
                "content_kind": "index_snippets",
            }, None
        url = str(action["url"])
        if not _allowed(url, spec):
            return _error(
                "blocked_domain", "The URL is outside the requested domain filter."
            ), None
        page: JsonObject | None = pages.get(url)
        if page is None:
            if not spec.live:
                return _error(
                    "cached_page_unavailable",
                    "Only index snippets are available for this URL; fresh page access is disabled.",
                ), None
            fetched = await self._client.fetch(
                url, egress=replace(self._egress, allowed_domains=spec.domains)
            )
            page = {
                "url": fetched.url,
                "title": fetched.title,
                "text": fetched.data[:24000],
                "truncated": fetched.truncated or len(fetched.data) > 24000,
            }
            pages[url] = page
            pages[fetched.url] = page
        if not _allowed(str(page["url"]), spec):
            return _error(
                "blocked_domain", "The page is outside the requested domain filter."
            ), None
        text = str(page.get("text", ""))
        sources = [{"url": page["url"], "title": page["title"]}]
        if action["action"] == "open_page":
            return {
                "url": page["url"],
                "title": page["title"],
                "text": text[: spec.context_chars],
                "truncated": bool(page.get("truncated"))
                or len(text) > spec.context_chars,
                "sources": sources,
            }, page
        pattern = str(action["pattern"])
        matches: list[JsonValue] = []
        start = 0
        allowance = spec.context_chars
        while (
            len(matches) < 20
            and allowance > 0
            and (index := text.find(pattern, start)) >= 0
        ):
            excerpt = text[max(0, index - 120) : index + len(pattern) + 120][:allowance]
            matches.append(excerpt)
            allowance -= len(excerpt)
            start = index + len(pattern)
        return {
            "url": page["url"],
            "matches": matches,
            "sources": sources,
            "searched_complete_page": not bool(page.get("truncated")),
            "pattern": pattern,
        }, page


def _error(code: str, message: str) -> JsonObject:
    return {"error": {"code": code, "message": message}}


def _allowed(url: str, spec: WebSearchSpec) -> bool:
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    return bool(host) and (
        not spec.domains or any(domain_matches(host, domain) for domain in spec.domains)
    )


def _retained_pages(records: tuple[JsonObject, ...]) -> dict[str, JsonObject]:
    pages: dict[str, JsonObject] = {}
    for record in records:
        page, action = record.get("page"), record.get("action")
        if (
            isinstance(page, dict)
            and isinstance(page.get("url"), str)
            and isinstance(page.get("text"), str)
        ):
            pages[str(page["url"])] = page
            if isinstance(action, dict) and isinstance(action.get("url"), str):
                pages[str(action["url"])] = page
    return pages


def _remember_sources(target: dict[str, str], result: JsonValue) -> None:
    sources = result.get("sources") if isinstance(result, dict) else None
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict) and isinstance(source.get("url"), str):
                target[str(source["url"])] = str(source.get("title") or source["url"])
