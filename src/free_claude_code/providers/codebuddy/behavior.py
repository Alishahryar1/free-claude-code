"""CodeBuddy-specific Chat Completions body adaptation."""

from typing import Any

from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.openai_chat import OpenAIChatBehavior

from .sanitize import (
    NEUTRAL_SYSTEM_PROMPT,
    normalize_tool_choice,
    sanitize_system_text,
    sanitize_tool_parameters,
)

_DEVELOPER_ROLE = "developer"
_SYSTEM_ROLE = "system"


class CodeBuddyChatBehavior(OpenAIChatBehavior):
    """Neutralize the request shapes CodeBuddy's upstream rejects.

    CodeBuddy only accepts streaming chat, requires the first message to be
    a system prompt, accepts ``tool_choice`` only as a plain string, and its
    WAF/schema validator rejects foreign channel fingerprints and advanced
    JSON Schema constructs in tool parameters.
    """

    def finalize_chat_body(
        self,
        body: dict[str, Any],
        *,
        reasoning: ReasoningPolicy,
    ) -> dict[str, Any]:
        """Apply CodeBuddy behavior that is independent of client protocol.

        CodeBuddy only accepts streaming chat; the shared transport always
        requests a stream, so the body itself must not repeat the field.
        """
        body.pop("stream", None)

        messages = body.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                if isinstance(role, str) and role.lower() == _DEVELOPER_ROLE:
                    message["role"] = _SYSTEM_ROLE
                if message.get("role") == _SYSTEM_ROLE:
                    self._sanitize_system_message(message)
            first = messages[0] if messages else None
            first_role = (
                str(first.get("role", "")).lower() if isinstance(first, dict) else ""
            )
            if first_role != _SYSTEM_ROLE:
                body["messages"] = [
                    {"role": _SYSTEM_ROLE, "content": NEUTRAL_SYSTEM_PROMPT},
                    *messages,
                ]

        normalize_tool_choice(body)

        tools = body.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                function = tool.get("function")
                if isinstance(function, dict) and "parameters" in function:
                    function["parameters"] = sanitize_tool_parameters(
                        function["parameters"]
                    )
        return body

    @staticmethod
    def _sanitize_system_message(message: dict[str, Any]) -> None:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = sanitize_system_text(content)
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    part["text"] = sanitize_system_text(part["text"])
