"""Recognize the narrow Anthropic web-tool requests FCC can execute locally."""

import json
import re
from dataclasses import dataclass

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic import MessagesRequest, Tool
from free_claude_code.core.anthropic.conversion import (
    unrepresentable_client_tool_constraints,
)
from free_claude_code.core.anthropic.models import (
    ContentBlockServerToolUse,
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    ContentBlockWebFetchToolResult,
    ContentBlockWebSearchToolResult,
    Message,
)
from free_claude_code.core.anthropic.server_tool_types import HIDDEN_WEB_SEARCH_NAME
from free_claude_code.core.gemini.web_search_replay import signed_gemini_turn
from free_claude_code.core.history_replay import HistoryReplayError

WEB_SEARCH_TYPE = "web_search_20250305"
WEB_FETCH_TYPE = "web_fetch_20250910"

_SERVER_TOOL_TYPES = {
    "web_search": WEB_SEARCH_TYPE,
    "web_fetch": WEB_FETCH_TYPE,
}
_SERVER_TOOL_PREFIXES = ("web_search_", "web_fetch_")
_SERVER_HISTORY_TYPES = {
    "server_tool_use",
    "web_search_tool_result",
    "web_fetch_tool_result",
}
_WEB_SEARCH_OPTION_FIELDS = {"max_uses", "allowed_domains", "blocked_domains"}
_DOMAIN_PATTERN = re.compile(
    r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
)


@dataclass(frozen=True, slots=True)
class WebSearchDomainFilter:
    """Literal hostname constraints supplied by Claude Code."""

    allowed: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AutomaticWebSearchPlan:
    """One exact Claude Code automatic-search request translated for a provider."""

    request: MessagesRequest
    private_name: str
    domains: WebSearchDomainFilter
    remaining_uses: int | None


def forced_tool_turn_text(request: MessagesRequest) -> str:
    """Text for parsing forced server-tool inputs: latest user turn only."""
    if not request.messages:
        return ""

    for message in reversed(request.messages):
        if message.role == "user":
            return content_text(message.content)
    return ""


def forced_server_tool_name(request: MessagesRequest) -> str | None:
    """Return a forced supported server tool, never an ordinary same-name function."""
    choice = request.tool_choice
    if not isinstance(choice, dict) or choice.get("type") != "tool":
        return None
    name = choice.get("name")
    if not isinstance(name, str) or name not in _SERVER_TOOL_TYPES:
        return None
    expected_type = _SERVER_TOOL_TYPES[name]
    if any(
        tool.name == name and tool.type == expected_type for tool in request.tools or []
    ):
        return name
    return None


def has_tool_named(request: MessagesRequest, name: str) -> bool:
    return any(tool.name == name for tool in request.tools or [])


def is_web_server_tool_request(request: MessagesRequest) -> bool:
    """True only when the client forces one supported Anthropic server tool."""
    return forced_server_tool_name(request) is not None


def is_anthropic_server_tool_definition(tool: Tool) -> bool:
    """Whether ``tool.type`` identifies an Anthropic web server-tool family."""
    tool_type = tool.type
    return isinstance(tool_type, str) and tool_type.startswith(_SERVER_TOOL_PREFIXES)


def has_listed_anthropic_server_tools(request: MessagesRequest) -> bool:
    """True when tools include a typed Anthropic web server-tool definition."""
    return any(
        is_anthropic_server_tool_definition(tool) for tool in request.tools or []
    )


def plan_automatic_web_search(
    request: MessagesRequest,
    *,
    web_tools_enabled: bool,
) -> AutomaticWebSearchPlan | None:
    """Translate one WebSearch definition while preserving ordinary client tools."""
    choice = request.tool_choice
    if choice is not None and (
        choice.get("type") not in ("auto", "any")
        or choice.get("disable_parallel_tool_use", False) is not False
        or set(choice) - {"type", "disable_parallel_tool_use"}
    ):
        return None

    tools = request.tools or []
    server_tools = [tool for tool in tools if is_anthropic_server_tool_definition(tool)]
    if not server_tools:
        return None
    if len(server_tools) != 1:
        return None

    tool = server_tools[0]
    if tool.type != WEB_SEARCH_TYPE:
        return None
    for other in tools:
        if other is tool:
            continue
        if other.type not in (None, "custom") or other.input_schema is None:
            return None
        if constraints := unrepresentable_client_tool_constraints(other):
            raise InvalidRequestError(
                f"Automatic web_search cannot enforce {constraints} for tool {other.name!r}."
            )
    names = {definition.name for definition in tools}
    if len(names) != len(tools):
        raise InvalidRequestError("Automatic web_search requires unique tool names.")
    if not web_tools_enabled:
        raise InvalidRequestError(
            "Anthropic web_search is disabled (ENABLE_WEB_SERVER_TOOLS=false). "
            "Enable local web tools or remove web_search from the request."
        )
    if tool.name != "web_search":
        raise InvalidRequestError(
            f"Server tool type {WEB_SEARCH_TYPE!r} must use name 'web_search'."
        )
    if tool.description is not None or tool.input_schema is not None:
        raise InvalidRequestError(
            "Anthropic web_search must not define description or input_schema."
        )

    options = tool.model_extra or {}
    unsupported = sorted(set(options) - _WEB_SEARCH_OPTION_FIELDS)
    if unsupported:
        raise InvalidRequestError(
            "Anthropic web_search contains unsupported option(s): "
            + ", ".join(unsupported)
            + "."
        )
    max_uses = _validate_max_uses(options.get("max_uses"))
    allowed = _parse_domains(
        options.get("allowed_domains"), field="web_search.allowed_domains"
    )
    blocked = _parse_domains(
        options.get("blocked_domains"), field="web_search.blocked_domains"
    )
    if allowed and blocked:
        raise InvalidRequestError(
            "web_search may define allowed_domains or blocked_domains, not both."
        )

    private_name = _private_search_name(names, request.messages)
    hidden_tool = Tool(
        name=private_name,
        description="Search the public web for current information.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    try:
        messages, used_this_turn = _normalized_search_history(request.messages)
    except HistoryReplayError as error:
        raise InvalidRequestError(str(error)) from error
    translated = request.model_copy(
        update={
            "tools": [hidden_tool if other is tool else other for other in tools],
            "messages": messages,
        },
        deep=True,
    )
    return AutomaticWebSearchPlan(
        request=translated,
        private_name=private_name,
        domains=WebSearchDomainFilter(allowed=allowed, blocked=blocked),
        remaining_uses=None if max_uses is None else max(0, max_uses - used_this_turn),
    )


def unsupported_server_tool_error(
    request: MessagesRequest, *, web_tools_enabled: bool
) -> str | None:
    """Return the user-facing error for a server-tool shape FCC does not handle."""
    forced = forced_server_tool_name(request)
    if forced and not web_tools_enabled:
        return (
            f"tool_choice forces Anthropic server tool {forced!r}, but local web "
            "server tools are disabled (ENABLE_WEB_SERVER_TOOLS=false). Enable "
            "them or remove the forced server tool."
        )
    if not forced and has_listed_anthropic_server_tools(request):
        return (
            f"FCC supports one {WEB_SEARCH_TYPE} alongside custom tools with "
            "automatic or required tool choice."
        )
    return None


def _private_search_name(names: set[str], messages: list[Message]) -> str:
    names.update(
        block.name
        for message in messages
        if message.role == "assistant" and isinstance(message.content, list)
        for block in message.content
        if isinstance(block, ContentBlockToolUse)
    )
    private_name = HIDDEN_WEB_SEARCH_NAME
    suffix = 0
    while private_name in names:
        suffix += 1
        private_name = f"{HIDDEN_WEB_SEARCH_NAME}_{suffix}"
    return private_name


_INCOMPLETE_HISTORY = (
    "Automatic web_search does not accept incomplete or unsupported prior "
    "Anthropic server-tool history."
)


def _normalized_search_history(messages: list[Message]) -> tuple[list[Message], int]:
    """Carry forward completed web results and count current-turn search attempts."""
    normalized: list[Message] = []
    used_this_turn = 0
    for message in messages:
        content = message.content
        if message.role == "user" and not (
            isinstance(content, list)
            and content
            and all(isinstance(block, ContentBlockToolResult) for block in content)
        ):
            used_this_turn = 0
        if not isinstance(content, list) or not any(
            getattr(block, "type", None) in _SERVER_HISTORY_TYPES for block in content
        ):
            normalized.append(message)
            continue
        if message.role != "assistant":
            raise InvalidRequestError(_INCOMPLETE_HISTORY)

        uses: dict[str, ContentBlockServerToolUse] = {}
        results: dict[
            str,
            tuple[
                int, ContentBlockWebSearchToolResult | ContentBlockWebFetchToolResult
            ],
        ] = {}
        for index, block in enumerate(content):
            if isinstance(block, ContentBlockServerToolUse):
                argument = {"web_search": "query", "web_fetch": "url"}.get(block.name)
                value = block.input.get(argument) if argument is not None else None
                if (
                    argument is None
                    or not block.id
                    or block.id in uses
                    or set(block.input) != {argument}
                    or not isinstance(value, str)
                    or not value.strip()
                ):
                    raise InvalidRequestError(_INCOMPLETE_HISTORY)
                uses[block.id] = block
            elif isinstance(
                block, (ContentBlockWebSearchToolResult, ContentBlockWebFetchToolResult)
            ):
                if not block.tool_use_id or block.tool_use_id in results:
                    raise InvalidRequestError(_INCOMPLETE_HISTORY)
                results[block.tool_use_id] = (index, block)
        if not uses or uses.keys() != results.keys():
            raise InvalidRequestError(_INCOMPLETE_HISTORY)

        signed_turn = any(use.name == "web_search" for use in uses.values()) and (
            signed_gemini_turn(content) is not None
        )
        projected = []
        for block in content:
            if isinstance(block, ContentBlockServerToolUse):
                result_index, result = results[block.id]
                if block.name == "web_search" and isinstance(
                    result, ContentBlockWebSearchToolResult
                ):
                    quoted = _search_history_text(block, result)
                    projected.append(block if signed_turn else quoted)
                    if (
                        not isinstance(result.content, dict)
                        or result.content.get("error_code") != "max_uses_exceeded"
                    ):
                        used_this_turn += 1
                elif block.name == "web_fetch" and isinstance(
                    result, ContentBlockWebFetchToolResult
                ):
                    next_index = result_index + 1
                    following = (
                        content[next_index] if next_index < len(content) else None
                    )
                    projected.append(
                        _fetch_history_text(
                            block,
                            result,
                            following.text
                            if isinstance(following, ContentBlockText)
                            else None,
                        )
                    )
                else:
                    raise InvalidRequestError(_INCOMPLETE_HISTORY)
            elif (
                signed_turn and isinstance(block, ContentBlockWebSearchToolResult)
            ) or not isinstance(
                block, (ContentBlockWebSearchToolResult, ContentBlockWebFetchToolResult)
            ):
                projected.append(block)
        normalized.append(message.model_copy(update={"content": projected}, deep=True))
    return normalized, used_this_turn


def _search_history_text(
    use: ContentBlockServerToolUse, result: ContentBlockWebSearchToolResult
) -> ContentBlockText:
    content = result.content
    if isinstance(content, list):
        hits: list[dict[str, str]] = []
        for hit in content:
            if (
                not isinstance(hit, dict)
                or hit.get("type") != "web_search_result"
                or not isinstance(hit.get("title"), str)
                or not isinstance(hit.get("url"), str)
            ):
                raise InvalidRequestError(_INCOMPLETE_HISTORY)
            hits.append({"title": hit["title"], "url": hit["url"]})
        record: dict[str, object] = {"query": use.input["query"], "results": hits}
    elif (
        isinstance(content, dict)
        and content.get("type") == "web_search_tool_result_error"
        and isinstance(content.get("error_code"), str)
    ):
        record = {"query": use.input["query"], "error_code": content["error_code"]}
    else:
        raise InvalidRequestError(_INCOMPLETE_HISTORY)
    return ContentBlockText(
        type="text",
        text="[Earlier web search data]\n"
        + json.dumps(record, ensure_ascii=False, separators=(",", ":")),
    )


def _fetch_history_text(
    use: ContentBlockServerToolUse,
    result: ContentBlockWebFetchToolResult,
    following_text: str | None,
) -> ContentBlockText:
    content = result.content
    if not isinstance(content, dict):
        raise InvalidRequestError(_INCOMPLETE_HISTORY)
    match content.get("type"):
        case "web_fetch_tool_error" | "web_fetch_tool_result_error" if isinstance(
            content.get("error_code"), str
        ):
            record = {"url": use.input["url"], "error_code": content["error_code"]}
        case "web_fetch_result":
            document = content.get("content")
            url = content.get("url")
            if not isinstance(document, dict) or document.get("type") != "document":
                raise InvalidRequestError(_INCOMPLETE_HISTORY)
            source = document.get("source")
            if (
                not isinstance(url, str)
                or not isinstance(source, dict)
                or source.get("type") != "text"
                or not isinstance(source.get("data"), str)
            ):
                raise InvalidRequestError(_INCOMPLETE_HISTORY)
            record = {"url": url}
            if isinstance(document.get("title"), str):
                record["title"] = document["title"]
            if source["data"] != following_text:
                record["text"] = source["data"]
        case _:
            raise InvalidRequestError(_INCOMPLETE_HISTORY)
    return ContentBlockText(
        type="text",
        text="[Earlier web fetch data]\n"
        + json.dumps(record, ensure_ascii=False, separators=(",", ":")),
    )


def _validate_max_uses(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidRequestError("web_search.max_uses must be a positive integer.")
    return value


def _parse_domains(value: object, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidRequestError(f"{field} must be a list of literal hostnames.")

    domains: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry or entry != entry.strip():
            raise InvalidRequestError(f"{field} entries must be non-empty hostnames.")
        if any(character in entry for character in ":/\\?#@*"):
            raise InvalidRequestError(
                f"{field} entry {entry!r} must be a literal hostname without a "
                "scheme, port, path, query, fragment, user info, or wildcard."
            )
        try:
            domain = entry.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise InvalidRequestError(
                f"{field} entry {entry!r} is not a valid hostname."
            ) from exc
        if _DOMAIN_PATTERN.fullmatch(domain) is None:
            raise InvalidRequestError(
                f"{field} entry {entry!r} is not a valid hostname."
            )
        domains.append(domain)

    if len(domains) != len(set(domains)):
        raise InvalidRequestError(f"{field} must not contain duplicate hostnames.")
    return tuple(domains)


def content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def extract_query(text: str) -> str:
    match = re.search(r"query:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().strip("\"'")
    return text.strip()


def extract_url(text: str) -> str:
    match = re.search(r"https?://\S+", text)
    return match.group(0).rstrip(").,]") if match else text.strip()
