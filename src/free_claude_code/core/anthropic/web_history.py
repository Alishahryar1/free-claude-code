"""Public portions of recorded Anthropic web calls and results."""

from ..history_replay import HistoryReplayError, tool_history_context
from ..json_types import JsonObject
from .content import get_block_attr, get_block_type


def _fields(value: object, *names: str) -> JsonObject:
    if not isinstance(value, dict):
        raise HistoryReplayError("Unsupported web tool history in Messages")
    return {name: value[name] for name in names if name in value}


def public_search_result(content: object) -> list[JsonObject] | JsonObject:
    if isinstance(content, list):
        hits: list[JsonObject] = []
        for hit in content:
            public = _fields(hit, "type", "url", "title", "page_age")
            if (
                public.get("type") != "web_search_result"
                or not isinstance(public.get("url"), str)
                or not isinstance(public.get("title"), str)
            ):
                raise HistoryReplayError(
                    "Unsupported web search result in Messages history"
                )
            hits.append(public)
        return hits
    error = _fields(content, "type", "error_code")
    if error.get("type") != "web_search_tool_result_error" or not isinstance(
        error.get("error_code"), str
    ):
        raise HistoryReplayError("Unsupported web search error in Messages history")
    return error


def _public_fetch_result(content: object) -> JsonObject:
    result = _fields(content, "type", "url", "retrieved_at", "content", "error_code")
    if result.get("type") in ("web_fetch_tool_error", "web_fetch_tool_result_error"):
        return _fields(result, "type", "error_code")
    if result.get("type") != "web_fetch_result":
        raise HistoryReplayError("Unsupported web fetch result in Messages history")
    if result.get("content") is not None:
        document = _fields(result["content"], "type", "title", "context", "source")
        source = _fields(document.get("source"), "type", "media_type", "data")
        if (
            document.get("type") != "document"
            or source.get("type") != "text"
            or not isinstance(source.get("data"), str)
        ):
            raise HistoryReplayError(
                "This web fetch document requires native Messages egress"
            )
        document["source"] = source
        result["content"] = document
    return result


def web_history_text(block: object) -> str:
    kind = get_block_type(block)
    if kind == "server_tool_use":
        name = get_block_attr(block, "name")
        if name not in ("web_search", "web_fetch"):
            raise HistoryReplayError(
                "This server tool history requires native Messages egress"
            )
        return tool_history_context(
            {
                "type": kind,
                "id": get_block_attr(block, "id"),
                "name": name,
                "input": _fields(get_block_attr(block, "input"), "query", "q", "url"),
            }
        )
    content = get_block_attr(block, "content")
    if kind == "web_search_tool_result":
        public = public_search_result(content)
    elif kind == "web_fetch_tool_result":
        public = _public_fetch_result(content)
    else:
        raise HistoryReplayError("Unsupported web tool in Messages history")
    return tool_history_context(
        {
            "type": kind,
            "tool_use_id": get_block_attr(block, "tool_use_id"),
            "content": public,
        }
    )
