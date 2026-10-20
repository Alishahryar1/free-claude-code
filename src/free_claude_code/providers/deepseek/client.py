"""DeepSeek provider implementation (OpenAI-compatible Chat Completions)."""

from collections.abc import Mapping
from typing import Any

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.history_replay import HistoryScope
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatProvider,
    usage_int,
)

from .compat import (
    DEEPSEEK_REQUEST_POLICY,
    build_deepseek_request_body,
    finalize_deepseek_chat_body,
)

_PROFILE = OpenAIChatProfile(
    DEEPSEEK_REQUEST_POLICY,
    NO_REASONING,
    history_scope=HistoryScope.ALL,
)


def _deepseek_cache_partition(
    usage_info: object,
) -> tuple[int, int, int | None] | None:
    cache_hit_tokens = usage_int(usage_info, "prompt_cache_hit_tokens")
    cache_miss_tokens = usage_int(usage_info, "prompt_cache_miss_tokens")
    if (
        cache_hit_tokens is None
        or cache_hit_tokens < 0
        or cache_miss_tokens is None
        or cache_miss_tokens < 0
    ):
        return None

    prompt_tokens = usage_int(usage_info, "prompt_tokens")
    if prompt_tokens is None or prompt_tokens < 0:
        return cache_hit_tokens, cache_miss_tokens, None
    if prompt_tokens != cache_hit_tokens + cache_miss_tokens:
        return None
    return cache_hit_tokens, cache_miss_tokens, prompt_tokens


class DeepSeekChatBehavior(OpenAIChatBehavior):
    """DeepSeek Chat adaptation without HTTP ownership."""

    def history_scope(self, body: Mapping[str, Any]) -> HistoryScope:
        return HistoryScope.ALL if body.get("tools") else HistoryScope.UNKNOWN

    @property
    def reasoning_off_fields(self) -> tuple[tuple[str, ...], ...]:
        return (("extra_body", "thinking"),)

    @property
    def normal_max_tokens(self) -> int:
        return ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS

    def build_messages_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict:
        return build_deepseek_request_body(
            request,
            reasoning=reasoning,
        )

    def finalize_chat_body(
        self,
        body: dict[str, Any],
        *,
        reasoning: ReasoningPolicy,
    ) -> dict[str, Any]:
        """Apply DeepSeek policy after either client-protocol translation."""
        finalize_deepseek_chat_body(body, reasoning)
        return body

    def cached_input_tokens(self, usage_info: object) -> int | None:
        cache_partition = _deepseek_cache_partition(usage_info)
        if cache_partition is None:
            return None
        cache_hit_tokens, _, prompt_tokens = cache_partition
        return cache_hit_tokens if prompt_tokens is not None else None

    def anthropic_usage_fields(self, usage_info: Any) -> dict[str, int]:
        cache_partition = _deepseek_cache_partition(usage_info)
        if cache_partition is None:
            return {}
        cache_hit_tokens, cache_miss_tokens, _ = cache_partition
        return {
            "input_tokens": cache_miss_tokens,
            "cache_read_input_tokens": cache_hit_tokens,
        }


class DeepSeekProvider(OpenAIChatProvider):
    """DeepSeek using ``https://api.deepseek.com`` Chat Completions."""

    def __init__(
        self, config: ProviderConfig, *, admission: ProviderAdmissionController
    ):
        super().__init__(
            config,
            behavior=DeepSeekChatBehavior(_PROFILE),
            admission=admission,
        )
