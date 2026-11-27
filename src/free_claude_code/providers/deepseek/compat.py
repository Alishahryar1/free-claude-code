"""DeepSeek Anthropic-to-OpenAI chat request policy."""

from loguru import logger

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import (
    ReasoningReplayMode,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import (
    JsonObject,
    json_array,
    json_objects,
)
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


def build_deepseek_request_body(
    request_data: MessagesRequest, *, reasoning: ReasoningPolicy
) -> JsonObject:
    """Build a DeepSeek Chat Completions body from an Anthropic request."""
    logger.debug(
        "DEEPSEEK_REQUEST: chat build model={} msgs={}",
        request_data.model,
        len(request_data.messages),
    )

    body = build_openai_chat_request_body(
        request_data,
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
        len(json_array(body.get("messages", []))),
        len(json_array(body.get("tools", []))),
    )
    return body


def finalize_deepseek_chat_body(body: JsonObject, reasoning: ReasoningPolicy) -> None:
    """Apply source-independent DeepSeek policy to one Chat body."""
    if body.get("tools") and reasoning.control is not ReasoningControl.OFF:
        for message in json_objects(body.get("messages", [])):
            if (
                message.get("role") == "assistant"
                and message.get("reasoning_content") is None
            ):
                message["reasoning_content"] = message.get("reasoning") or ""

    _apply_deepseek_chat_extras(body, reasoning)
    if body.get("max_tokens") is None:
        body["max_tokens"] = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def _apply_deepseek_chat_extras(body: JsonObject, policy: ReasoningPolicy) -> None:
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
