"""Recognize the narrow Anthropic web-tool requests FCC can execute locally."""

import re
from dataclasses import dataclass

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic import MessagesRequest, Tool
from free_claude_code.core.web_domains import WebSearchDomainFilter, parse_domains

WEB_SEARCH_TYPE = "web_search_20250305"
WEB_FETCH_TYPE = "web_fetch_20250910"
HIDDEN_WEB_SEARCH_NAME = "fcc_web_search"

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


@dataclass(frozen=True, slots=True)
class AutomaticWebSearchPlan:
    """One exact Claude Code automatic-search request translated for a provider."""

    request: MessagesRequest
    domains: WebSearchDomainFilter


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
    """Translate only Claude Code's server-tool-only automatic WebSearch request."""
    choice = request.tool_choice
    if choice is not None and choice != {"type": "auto"}:
        return None

    tools = request.tools or []
    server_tools = [tool for tool in tools if is_anthropic_server_tool_definition(tool)]
    if not server_tools:
        return None
    if len(tools) != 1 or len(server_tools) != 1:
        return None

    tool = server_tools[0]
    if tool.type != WEB_SEARCH_TYPE:
        return None
    if not web_tools_enabled:
        raise InvalidRequestError(
            "Anthropic web_search is disabled (ENABLE_WEB_SERVER_TOOLS=false). "
            "Enable local web tools or remove web_search from the request."
        )
    if _has_server_tool_history(request):
        raise InvalidRequestError(
            "Automatic web_search does not accept prior Anthropic server-tool history."
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
    _validate_max_uses(options.get("max_uses"))
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

    hidden_tool = Tool(
        name=HIDDEN_WEB_SEARCH_NAME,
        description="Search the public web for current information.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    translated = request.model_copy(
        update={"tools": [hidden_tool]},
        deep=True,
    )
    return AutomaticWebSearchPlan(
        request=translated,
        domains=WebSearchDomainFilter(allowed=allowed, blocked=blocked),
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
            "FCC supports automatic Anthropic web_search only for Claude Code's "
            f"{WEB_SEARCH_TYPE} server-tool-only request. Automatic web_fetch, "
            "mixed tools, and other server-tool choices are not supported."
        )
    return None


def _has_server_tool_history(request: MessagesRequest) -> bool:
    for message in request.messages:
        if not isinstance(message.content, list):
            continue
        for block in message.content:
            if getattr(block, "type", None) in _SERVER_HISTORY_TYPES:
                return True
    return False


def _validate_max_uses(value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidRequestError("web_search.max_uses must be a positive integer.")


def _parse_domains(value: object, *, field: str) -> tuple[str, ...]:
    try:
        return parse_domains(value, field=field)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc


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
