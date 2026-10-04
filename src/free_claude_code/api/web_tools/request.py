"""Detect forced Anthropic web server tool requests."""

from dataclasses import dataclass

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic import MessagesRequest, Tool

from .egress import normalize_web_domain
from .parsers import content_text


def forced_tool_turn_text(request: MessagesRequest) -> str:
    """Text for parsing forced server-tool inputs: latest user turn only (avoids stale history)."""
    if not request.messages:
        return ""

    for message in reversed(request.messages):
        if message.role == "user":
            return content_text(message.content)
    return ""


def forced_server_tool_name(request: MessagesRequest) -> str | None:
    """Return web_search or web_fetch only when tool_choice forces that server tool."""
    tc = request.tool_choice
    if not isinstance(tc, dict):
        return None
    if tc.get("type") != "tool":
        return None
    name = tc.get("name")
    if name in {"web_search", "web_fetch"}:
        return str(name)
    return None


def has_tool_named(request: MessagesRequest, name: str) -> bool:
    return any(tool.name == name for tool in request.tools or [])


def is_web_server_tool_request(request: MessagesRequest) -> bool:
    """True when the client forces a web server tool via tool_choice (not merely listed)."""
    forced = forced_server_tool_name(request)
    if forced is None:
        return False
    return has_tool_named(request, forced)


def is_anthropic_server_tool_definition(tool: Tool) -> bool:
    """Whether ``tool`` refers to an Anthropic server tool (web_search / web_fetch family)."""
    name = (tool.name or "").strip()
    if name in ("web_search", "web_fetch"):
        return True
    typ = tool.type
    if isinstance(typ, str):
        return typ.startswith("web_search") or typ.startswith("web_fetch")
    return False


def has_listed_anthropic_server_tools(request: MessagesRequest) -> bool:
    """True when tools include web_search / web_fetch-style entries (listed, forced or not)."""
    return any(is_anthropic_server_tool_definition(t) for t in (request.tools or []))


@dataclass(frozen=True, slots=True)
class LocalWebTool:
    name: str
    max_uses: int
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()


_SUPPORTED_TYPES = {
    "web_search": frozenset({"web_search_20250305"}),
    "web_fetch": frozenset({"web_fetch_20250910"}),
}
_SUPPORTED_OPTIONS = frozenset(
    {"max_uses", "allowed_domains", "blocked_domains", "cache_control"}
)


def _domains_option(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise InvalidRequestError(
            "Local web-tool domain options must be lists of domains."
        )
    try:
        return tuple(normalize_web_domain(str(item)) for item in value)
    except ValueError as exc:
        raise InvalidRequestError(
            "Local web-tool domain options contain an invalid domain."
        ) from exc


def local_web_tools(request: MessagesRequest) -> tuple[LocalWebTool, ...]:
    """Validate the local fallback's supported subset before any side effects."""
    tools: list[LocalWebTool] = []
    for tool in request.tools or ():
        if (
            tool.name not in _SUPPORTED_TYPES
            or tool.type not in _SUPPORTED_TYPES[tool.name]
        ):
            raise InvalidRequestError(
                "Local web tools support web_search_20250305 and web_fetch_20250910 only; "
                "hosted dynamic-filtering tools and mixed client/server tools are unsupported."
            )
        extra = tool.model_extra or {}
        unknown = set(extra) - _SUPPORTED_OPTIONS
        if unknown or tool.input_schema is not None:
            raise InvalidRequestError(
                "Local web tools cannot honour these definition fields: "
                f"{sorted(unknown | ({'input_schema'} if tool.input_schema is not None else set()))}."
            )
        max_uses = extra.get("max_uses", 4)
        if not isinstance(max_uses, int) or isinstance(max_uses, bool) or max_uses <= 0:
            raise InvalidRequestError(
                "Local web-tool max_uses must be a positive integer."
            )
        allowed = _domains_option(extra.get("allowed_domains"))
        blocked = _domains_option(extra.get("blocked_domains"))
        if allowed and blocked:
            raise InvalidRequestError(
                "Specify allowed_domains or blocked_domains, not both."
            )
        if any(item.name == tool.name for item in tools):
            raise InvalidRequestError(
                "Duplicate local web-tool definitions are unsupported."
            )
        tools.append(LocalWebTool(tool.name, min(max_uses, 4), allowed, blocked))
    if not tools:
        raise InvalidRequestError(
            "A forced web tool must have a matching supported definition."
        )
    choice = request.tool_choice
    if choice is not None:
        if set(choice) - {"type", "name", "disable_parallel_tool_use"}:
            raise InvalidRequestError("Unsupported local web-tool choice fields.")
        if choice.get("type") not in {"auto", "any", "none", "tool"}:
            raise InvalidRequestError("Unsupported local web-tool choice type.")
        if "disable_parallel_tool_use" in choice and not isinstance(
            choice["disable_parallel_tool_use"], bool
        ):
            raise InvalidRequestError("disable_parallel_tool_use must be a boolean.")
        if choice.get("type") != "tool" and "name" in choice:
            raise InvalidRequestError("Only a named tool choice may specify name.")
        if choice.get("type") == "tool" and choice.get("name") not in {
            item.name for item in tools
        }:
            raise InvalidRequestError(
                "A forced web tool must have a matching supported definition."
            )
    return tuple(tools)


def is_local_web_tool_request(request: MessagesRequest) -> bool:
    """Detect a web-only tool set without interpreting user text as tool intent."""
    return bool(request.tools) and all(
        is_anthropic_server_tool_definition(tool) for tool in request.tools
    )


def unsupported_server_tool_error(
    request: MessagesRequest, *, web_tools_enabled: bool
) -> str | None:
    """Reject unsupported hosted tools before they can reach OpenAI Chat upstreams."""
    forced = forced_server_tool_name(request)
    if not forced and not has_listed_anthropic_server_tools(request):
        return None
    if not web_tools_enabled:
        if forced:
            return (
                f"tool_choice forces Anthropic server tool {forced!r}, but local web server tools are "
                "disabled (ENABLE_WEB_SERVER_TOOLS=false). Enable them or remove the forced server tool."
            )
        return (
            "FCC cannot pass listed Anthropic server tools to OpenAI Chat upstreams. "
            "Enable local handling with ENABLE_WEB_SERVER_TOOLS=true, or remove these tools."
        )
    try:
        local_web_tools(request)
    except InvalidRequestError as exc:
        return exc.message
    return None
