"""DeepSeek Anthropic-to-OpenAI chat request policy."""

from collections.abc import Mapping
from typing import Any

from loguru import logger

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import (
    ReasoningReplayMode,
    dump_messages_request,
    serialize_tool_result_content,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.openai_chat import (
    OpenAIChatRequestPolicy,
    build_openai_chat_request_body,
)

DEEPSEEK_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="DEEPSEEK",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    include_extra_body=True,
)

_UNSUPPORTED_MESSAGE_BLOCK_TYPES = frozenset(
    {
        "image",
        "document",
    }
)
_STRIPPABLE_MESSAGE_BLOCK_TYPES = frozenset({"image", "document"})
_FORWARDABLE_ATTACHMENT_BLOCK_TYPES = frozenset({"image"})
_OMITTED_ATTACHMENT_TEXT = (
    "[attachment omitted: DeepSeek does not support image or document inputs]"
)
_OMITTED_ATTACHMENT_BLOCK = {"type": "text", "text": _OMITTED_ATTACHMENT_TEXT}

# DeepSeek-V4.1-Flash moved vision into the primary API, so the ``vision`` name
# heuristic below no longer catches it. Only the first-party GA id is listed:
# legacy ``deepseek-v4-flash`` is deliberately excluded because gateways still
# serve the older, text-only model under that name.
_VISION_CAPABLE_MODEL_IDS = frozenset({"deepseek-flash"})


def _is_vision_capable_model(model: str | None) -> bool:
    """True when the DeepSeek model accepts OpenAI image parts.

    DeepSeek's vision models (e.g. ``deepseek-v4-flash-vision-exp``) support
    image inputs natively, as does DeepSeek-V4.1-Flash (``deepseek-flash``).
    Document blocks are still stripped because the shared OpenAI conversion
    path has no document parts.
    """
    if not model:
        return False
    name = str(model).rsplit("/", 1)[-1].lower()
    return "vision" in name or name in _VISION_CAPABLE_MODEL_IDS


def build_deepseek_request_body(
    request_data: MessagesRequest, *, reasoning: ReasoningPolicy
) -> dict:
    """Build a DeepSeek Chat Completions body from an Anthropic request."""
    logger.debug(
        "DEEPSEEK_REQUEST: chat build model={} msgs={}",
        request_data.model,
        len(request_data.messages),
    )

    vision_capable = _is_vision_capable_model(request_data.model)
    data = dump_messages_request(request_data)
    if "messages" in data:
        data["messages"] = _strip_unsupported_attachment_blocks(
            data["messages"], allow_attachments=vision_capable
        )
    _validate_deepseek_request_dict(data, allow_attachments=vision_capable)
    if "messages" in data:
        data["messages"] = _normalize_tool_result_content(data["messages"])

    sanitized_request = MessagesRequest.model_validate(data)
    body = build_openai_chat_request_body(
        sanitized_request,
        reasoning=reasoning,
        policy=DEEPSEEK_REQUEST_POLICY,
        postprocessors=(
            lambda body, _request, _policy: finalize_deepseek_chat_body(
                body, reasoning
            ),
        ),
    )

    logger.debug(
        "DEEPSEEK_REQUEST: build done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body


def finalize_deepseek_chat_body(
    body: dict[str, Any], reasoning: ReasoningPolicy
) -> None:
    """Apply source-independent DeepSeek policy to one Chat body."""
    if body.get("tools") and reasoning.control is not ReasoningControl.OFF:
        for message in body.get("messages", []):
            if (
                message.get("role") == "assistant"
                and message.get("reasoning_content") is None
            ):
                message["reasoning_content"] = message.get("reasoning") or ""

    _downgrade_chat_forced_tool_choice(body)
    _apply_deepseek_chat_extras(body, reasoning)
    if body.get("max_tokens") is None:
        body["max_tokens"] = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def _downgrade_chat_forced_tool_choice(body: dict[str, Any]) -> None:
    tool_choice = body.get("tool_choice")
    if not isinstance(tool_choice, dict) or tool_choice.get("type") != "function":
        return
    function = tool_choice.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        return
    logger.debug(
        "DEEPSEEK_REQUEST: downgrading forced tool_choice to auto for unsupported "
        "native request shape tool={}",
        function["name"],
    )
    body["tool_choice"] = "auto"


def _strip_unsupported_attachment_blocks(
    messages: Any, *, allow_attachments: bool = False
) -> Any:
    if not isinstance(messages, list):
        return messages

    stripped: list[Any] = []
    top_level_dropped: dict[str, int] = {}
    nested_dropped: dict[str, int] = {}
    placeholder_replacements = 0

    for message in messages:
        if not isinstance(message, dict):
            stripped.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            stripped.append(message)
            continue

        new_content: list[Any] = []
        message_dropped_attachment = False
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype in _STRIPPABLE_MESSAGE_BLOCK_TYPES:
                    if (
                        allow_attachments
                        and btype in _FORWARDABLE_ATTACHMENT_BLOCK_TYPES
                    ):
                        new_content.append(block)
                        continue
                    top_level_dropped[btype] = top_level_dropped.get(btype, 0) + 1
                    message_dropped_attachment = True
                    continue
                if btype == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, list):
                        filtered_inner: list[Any] = []
                        for sub in inner:
                            if (
                                isinstance(sub, dict)
                                and sub.get("type") in _STRIPPABLE_MESSAGE_BLOCK_TYPES
                            ):
                                sub_type = sub["type"]
                                if (
                                    allow_attachments
                                    and sub_type in _FORWARDABLE_ATTACHMENT_BLOCK_TYPES
                                ):
                                    filtered_inner.append(sub)
                                    continue
                                nested_dropped[sub_type] = (
                                    nested_dropped.get(sub_type, 0) + 1
                                )
                                continue
                            filtered_inner.append(sub)
                        if not filtered_inner:
                            filtered_inner = [_OMITTED_ATTACHMENT_BLOCK]
                            placeholder_replacements += 1
                        new_block = dict(block)
                        new_block["content"] = filtered_inner
                        new_content.append(new_block)
                        continue
            new_content.append(block)
        if not new_content and message_dropped_attachment:
            new_content = [_OMITTED_ATTACHMENT_BLOCK]
            placeholder_replacements += 1
        new_msg = dict(message)
        new_msg["content"] = new_content
        stripped.append(new_msg)

    if top_level_dropped or nested_dropped:
        logger.warning(
            "DEEPSEEK_REQUEST: stripped unsupported attachment blocks "
            "(top_level={} nested_in_tool_result={} placeholder_tool_results={}). "
            "This model does not accept these attachment types; it will not see "
            "this content.",
            dict(top_level_dropped),
            dict(nested_dropped),
            placeholder_replacements,
        )
    return stripped


def _is_server_listed_tool(tool: Mapping[str, Any]) -> bool:
    name = (tool.get("name") or "").strip()
    if name in ("web_search", "web_fetch"):
        return True
    typ = tool.get("type")
    if isinstance(typ, str):
        return typ.startswith("web_search") or typ.startswith("web_fetch")
    return False


def _walk_block_list_for_unsupported(
    blocks: Any, *, where: str, allow_attachments: bool = False
) -> None:
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in _UNSUPPORTED_MESSAGE_BLOCK_TYPES:
            if allow_attachments and btype in _FORWARDABLE_ATTACHMENT_BLOCK_TYPES:
                continue
            raise InvalidRequestError(
                f"DeepSeek native does not support {btype!r} blocks ({where})."
            )
        if btype == "tool_result" and "content" in block:
            _walk_block_list_for_unsupported(
                block["content"],
                where=f"{where} (tool_result content)",
                allow_attachments=allow_attachments,
            )


def _validate_deepseek_request_dict(
    data: dict[str, Any], *, allow_attachments: bool = False
) -> None:
    mcp = data.get("mcp_servers")
    if mcp:
        raise InvalidRequestError("DeepSeek does not support mcp_servers on requests.")

    for tool in data.get("tools") or ():
        if not isinstance(tool, dict):
            continue
        if _is_server_listed_tool(tool):
            raise InvalidRequestError(
                "DeepSeek does not support listed Anthropic server tools "
                "(web_search / web_fetch). Remove them or use a different provider."
            )

    for i, message in enumerate(data.get("messages") or ()):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            _walk_block_list_for_unsupported(
                content,
                where=f"messages[{i}].content",
                allow_attachments=allow_attachments,
            )

    system = data.get("system")
    if isinstance(system, list):
        _walk_block_list_for_unsupported(
            system, where="system", allow_attachments=allow_attachments
        )


def _normalize_tool_result_content(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages

    normalized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue

        new_content: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue

            if block.get("type") == "tool_result":
                normalized_block = dict(block)
                normalized_block["content"] = serialize_tool_result_content(
                    block.get("content")
                )
                new_content.append(normalized_block)
            else:
                new_content.append(block)

        new_msg = dict(message)
        new_msg["content"] = new_content
        normalized.append(new_msg)

    return normalized


def _apply_deepseek_chat_extras(body: dict[str, Any], policy: ReasoningPolicy) -> None:
    extra_body = body.setdefault("extra_body", {})
    if not isinstance(extra_body, dict):
        return
    if policy.control is ReasoningControl.OFF:
        extra_body["thinking"] = {"type": "disabled"}
        return
    if policy.effort in {ReasoningEffort.XHIGH, ReasoningEffort.MAX}:
        body["reasoning_effort"] = "max"
    elif policy.effort is not None:
        body["reasoning_effort"] = "high"
    elif policy.requests_reasoning:
        extra_body["thinking"] = {"type": "enabled"}
