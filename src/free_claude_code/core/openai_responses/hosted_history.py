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
        return list(json_objects(value)) if value is not None else []
    except TypeError as error:
        raise ResponsesConversionError(
            "Expected a list of objects in Responses tool history"
        ) from error


def public_tool_record(kind: str, item: JsonObject) -> JsonObject | None:
    """Select public fields before a provider tool record becomes conversation text."""
    data = _fields(item, "type", "call_id", "id", "status")
    match kind:
        case "tool_search_output":
            data["tools"] = public_search_tools(item.get("tools"))
        case "tool_search_call":
            data["arguments"] = item.get("arguments")
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
            data["action"] = _fields(
                action, "type", "query", "queries", "pattern", "url"
            )
            data["sources"] = [
                _fields(source, "url") for source in _entries(action.get("sources"))
            ]
        case "file_search_call":
            data["queries"] = item.get("queries")
            data["results"] = [
                _fields(entry, "filename", "text")
                for entry in _entries(item.get("results"))
            ]
        case "mcp_call":
            data.update(_fields(item, "name", "server_label", "arguments", "output"))
            if item.get("error") is not None:
                data["error"] = _fields(item["error"], "type", "code", "message")
        case "code_interpreter_call":
            data["code"] = item.get("code")
            logs: list[str] = []
            for output in _entries(item.get("outputs")):
                if output.get("type") != "logs" or not isinstance(
                    log := output.get("logs"), str
                ):
                    raise ResponsesConversionError(
                        "This code interpreter output requires native Responses"
                    )
                logs.append(log)
            data["logs"] = logs
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
    return data
