"""Safe metadata summaries for HTTP 422 validation logging (no raw text content)."""

from free_claude_code.core.json_types import JsonObject


def summarize_request_validation_body(
    body: object,
) -> tuple[list[JsonObject], list[str]]:
    """Return message shape summary and tool name list for debug logs."""
    messages = body.get("messages") if isinstance(body, dict) else None
    message_summary: list[JsonObject] = []
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                message_summary.append({"message_kind": type(msg).__name__})
                continue
            content = msg.get("content")
            item: JsonObject = {
                "role": msg.get("role"),
                "content_kind": type(content).__name__,
            }
            if isinstance(content, list):
                item["block_types"] = [
                    block.get("type", "dict")
                    if isinstance(block, dict)
                    else type(block).__name__
                    for block in content[:12]
                ]
                item["block_keys"] = [
                    sorted(str(key) for key in block)[:12]
                    for block in content[:5]
                    if isinstance(block, dict)
                ]
            elif isinstance(content, str):
                item["content_length"] = len(content)
            message_summary.append(item)

    tool_names: list[str] = []
    if isinstance(body, dict) and isinstance(body.get("tools"), list):
        tool_names = [
            str(tool.get("name", ""))
            for tool in body["tools"]
            if isinstance(tool, dict)
        ]

    return message_summary, tool_names
