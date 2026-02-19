"""Message and tool format converters."""

import json
from typing import Any


def get_block_attr(block: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from object or dict."""
    if hasattr(block, attr):
        return getattr(block, attr)
    if isinstance(block, dict):
        return block.get(attr, default)
    return default


def get_block_type(block: Any) -> str | None:
    """Get block type from object or dict."""
    return get_block_attr(block, "type")


class AnthropicToOpenAIConverter:
    """Converts Anthropic message format to OpenAI format."""

    @staticmethod
    def convert_messages(
        messages: list[Any],
        *,
        include_reasoning_for_openrouter: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert a list of Anthropic messages to OpenAI format.

        When include_reasoning_for_openrouter is True, assistant messages with
        thinking blocks get reasoning_content added for OpenRouter multi-turn
        reasoning continuation.
        """
        result = []

        for msg in messages:
            role = msg.role
            content = msg.content

            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    result.extend(
                        AnthropicToOpenAIConverter._convert_assistant_message(
                            content,
                            include_reasoning_for_openrouter=include_reasoning_for_openrouter,
                        )
                    )
                elif role == "user":
                    result.extend(
                        AnthropicToOpenAIConverter._convert_user_message(content)
                    )
            else:
                result.append({"role": role, "content": str(content)})

        return result

    @staticmethod
    def _convert_assistant_message(
        content: list[Any],
        *,
        include_reasoning_for_openrouter: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert assistant message blocks, preserving interleaved thinking+text order."""
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content:
            block_type = get_block_type(block)

            if block_type == "text":
                content_parts.append(get_block_attr(block, "text", ""))
            elif block_type == "thinking":
                thinking = get_block_attr(block, "thinking", "")
                content_parts.append(f"<think>\n{thinking}\n</think>")
                if include_reasoning_for_openrouter:
                    thinking_parts.append(thinking)
            elif block_type == "tool_use":
                tool_input = get_block_attr(block, "input", {})
                tool_calls.append(
                    {
                        "id": get_block_attr(block, "id"),
                        "type": "function",
                        "function": {
                            "name": get_block_attr(block, "name"),
                            "arguments": json.dumps(tool_input)
                            if isinstance(tool_input, dict)
                            else str(tool_input),
                        },
                    }
                )

        content_str = "\n\n".join(content_parts)

        # Ensure content is never an empty string for assistant messages
        # NIM (especially Mistral models) requires non-empty content if there are no tool calls
        if not content_str and not tool_calls:
            content_str = " "

        msg: dict[str, Any] = {
            "role": "assistant",
            "content": content_str,
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if include_reasoning_for_openrouter and thinking_parts:
            msg["reasoning_content"] = "\n".join(thinking_parts)

        return [msg]

    @staticmethod
    def _convert_user_message(content: list[Any]) -> list[dict[str, Any]]:
        """Convert user message blocks (including tool results), preserving order."""
        result: list[dict[str, Any]] = []
        text_parts: list[str] = []

        def flush_text() -> None:
            if text_parts:
                result.append({"role": "user", "content": "\n".join(text_parts)})
                text_parts.clear()

        for block in content:
            block_type = get_block_type(block)

            if block_type == "text":
                text_parts.append(get_block_attr(block, "text", ""))
            elif block_type == "tool_result":
                flush_text()
                tool_content = get_block_attr(block, "content", "")
                if isinstance(tool_content, list):
                    tool_content = "\n".join(
                        item.get("text", str(item))
                        if isinstance(item, dict)
                        else str(item)
                        for item in tool_content
                    )
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": get_block_attr(block, "tool_use_id"),
                        "content": str(tool_content) if tool_content else "",
                    }
                )

        flush_text()
        return result

    @staticmethod
    def convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
        """Convert Anthropic tools to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    @staticmethod
    def convert_tool_choice(tool_choice: Any) -> Any:
        """Convert Anthropic tool_choice to OpenAI format."""
        if isinstance(tool_choice, dict):
            tc_type = tool_choice.get("type")
            if tc_type == "tool":
                return {
                    "type": "function",
                    "function": {"name": tool_choice.get("name")},
                }
            elif tc_type in ("auto", "any"):
                return "required" if tc_type == "any" else "auto"
            else:
                return tc_type
        return tool_choice

    @staticmethod
    def convert_system_prompt(system: Any) -> dict[str, str] | None:
        """Convert Anthropic system prompt to OpenAI format."""
        if isinstance(system, str):
            return {"role": "system", "content": system}
        elif isinstance(system, list):
            text_parts = [
                get_block_attr(block, "text", "")
                for block in system
                if get_block_type(block) == "text"
            ]
            if text_parts:
                return {"role": "system", "content": "\n\n".join(text_parts).strip()}
        return None
