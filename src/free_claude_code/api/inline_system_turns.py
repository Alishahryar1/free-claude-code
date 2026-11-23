"""Explain system turns inside Messages ``messages`` to the people who send them."""

import re
from collections.abc import Mapping

from free_claude_code.application.errors import InvalidRequestError

_CLAUDE_CODE_VERSION = re.compile(r"\bclaude-cli/(\d+(?:\.\d+)+)")
# Claude Code retries without system turns only when the refusal carries this code.
MID_CONVERSATION_SYSTEM_NOT_SUPPORTED = "mid_conversation_system_not_supported"


def reject_inline_system_turns(
    body: Mapping[str, object], headers: Mapping[str, str] | None
) -> None:
    """Raise an actionable error when ``messages`` holds a ``system`` turn.

    Current Claude Code recognizes the error code and resends without these
    turns. Older builds cannot recover, so the message says how to fix it.
    """

    messages = body.get("messages")
    if not isinstance(messages, list) or not any(
        isinstance(message, Mapping) and message.get("role") == "system"
        for message in messages
    ):
        return
    user_agent = headers.get("user-agent", "") if headers is not None else ""
    if match := _CLAUDE_CODE_VERSION.search(user_agent):
        raise InvalidRequestError(
            f"Claude Code {match.group(1)} sent a system message inside `messages`, "
            "which Free Claude Code can't accept. Run `claude update`, then "
            "restart Claude Code.",
            error_code=MID_CONVERSATION_SYSTEM_NOT_SUPPORTED,
        )
    raise InvalidRequestError(
        "System messages belong in the top-level `system` field, not in "
        "`messages`. If you're using Claude Code, run `claude update`, then "
        "restart it.",
        error_code=MID_CONVERSATION_SYSTEM_NOT_SUPPORTED,
    )
