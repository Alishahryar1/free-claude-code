"""Model-visible evidence from completed Responses tools."""

from ..json_types import JsonObject, JsonValue, json_objects
from .errors import ResponsesConversionError
from .tool_search import public_search_tools


def _fields(value: JsonValue, *names: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ResponsesConversionError("Expected an object in Responses tool history")
    return {name: value[name] for name in names if name in value}


def _entries(value: JsonValue) -> list[JsonObject]:
    try:
        return list(json_objects(value))
    except TypeError as error:
        raise ResponsesConversionError(
            "Expected a list of objects in Responses tool history"
        ) from error


def _keep_entry_fields(record: JsonObject, field: str, *names: str) -> None:
    if (value := record.get(field)) is not None:
        record[field] = [_fields(entry, *names) for entry in _entries(value)]


def public_tool_record(kind: str, item: JsonObject) -> JsonObject | None:
    """Select public fields before a provider tool record becomes conversation text."""
    data: JsonObject = {}
    match kind:
        case "tool_search_output":
            data = _fields(item, "tools")
            if data.get("tools") is not None:
                data["tools"] = public_search_tools(data["tools"])
        case "tool_search_call":
            data = _fields(item, "arguments")
        case "web_search_call":
            action = item.get("action")
            if not isinstance(action, dict) or action.get("type") not in (
                "search",
                "find_in_page",
                "open_page",
            ):
                raise ResponsesConversionError(
                    "This web search action requires native Responses"
                )
            public_action = _fields(
                action, "type", "query", "queries", "pattern", "url", "sources"
            )
            _keep_entry_fields(public_action, "sources", "url")
            data["action"] = public_action
        case "file_search_call":
            data = _fields(item, "queries", "results")
            _keep_entry_fields(data, "results", "filename", "text")
        case "mcp_call":
            data = _fields(item, "name", "server_label", "arguments", "output", "error")
            if (error := data.get("error")) is not None:
                if not isinstance(error, dict):
                    raise ResponsesConversionError(
                        "Invalid MCP error in Responses history"
                    )
                if error.get("type") == "mcp_tool_execution_error":
                    data["error"] = _fields(error, "type", "content")
                elif error.get("type") in ("mcp_protocol_error", "http_error"):
                    data["error"] = _fields(error, "type", "code", "message")
                else:
                    raise ResponsesConversionError(
                        "This MCP error requires native Responses"
                    )
        case "code_interpreter_call":
            data = _fields(item, "code", "outputs")
            if (raw_outputs := data.get("outputs")) is not None:
                entries = _entries(raw_outputs)
                if any(
                    entry.get("type") != "logs"
                    or not isinstance(entry.get("logs"), str)
                    for entry in entries
                ):
                    raise ResponsesConversionError(
                        "This code interpreter output requires native Responses"
                    )
                data["outputs"] = [_fields(entry, "type", "logs") for entry in entries]
        case "local_shell_call" | "shell_call":
            data["action"] = _fields(
                item.get("action"), "type", "commands", "command", "working_directory"
            )
        case "local_shell_call_output" | "apply_patch_call_output":
            data.update(_fields(item, "output"))
        case "shell_call_output":
            outputs: list[JsonObject] = []
            for entry in _entries(item.get("output")):
                output = _fields(entry, "stdout", "stderr")
                output["outcome"] = _fields(entry.get("outcome"), "type", "exit_code")
                outputs.append(output)
            data["output"] = outputs
        case "apply_patch_call":
            data["operation"] = _fields(item.get("operation"), "type", "path", "diff")
        case "computer_call":
            action = item.get("action")
            if (
                not isinstance(action, dict)
                or action.get("type") != "screenshot"
                or item.get("pending_safety_checks")
                or item.get("actions")
            ):
                raise ResponsesConversionError(
                    "This computer action requires native Responses"
                )
            data["action"] = "screenshot"
        case _:
            return None
    return {**_fields(item, "type", "call_id", "id", "status"), **data}
