"""Provider-specific Chat adaptation, independent of HTTP resource ownership."""

from collections.abc import Iterator, Mapping
from typing import Any

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.usage import anthropic_input_usage_fields
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.history_replay import HistoryScope
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningPolicy,
)

from .profiles import OpenAIChatProfile
from .request_policy import (
    build_openai_chat_request_body,
)
from .stream_output import (
    ChatStreamOutput,
)
from .usage import (
    nested_usage_int,
    usage_int,
)


class OpenAIChatBehavior:
    """Default Chat behavior; exceptional providers override only their quirks."""

    def __init__(self, profile: OpenAIChatProfile) -> None:
        self.profile = profile

    def build_messages_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict[str, Any]:
        """Build a provider request from the immutable profile."""
        body = build_openai_chat_request_body(
            request,
            reasoning=reasoning,
            policy=self.profile.request_policy,
            postprocessors=self.profile.request_postprocessors,
        )
        return self.finalize_chat_body(body, reasoning=reasoning)

    @property
    def reasoning_off_fields(self) -> tuple[tuple[str, ...], ...]:
        field = self.profile.reasoning.off_field
        return (field,) if field is not None else ()

    @property
    def normal_max_tokens(self) -> int | None:
        return self.profile.request_policy.default_max_tokens

    def finalize_chat_body(
        self,
        body: dict[str, Any],
        *,
        reasoning: ReasoningPolicy,
    ) -> dict[str, Any]:
        """Apply provider behavior that is independent of client protocol."""
        return body

    def history_scope(self, body: Mapping[str, Any]) -> HistoryScope:
        return self.profile.history_scope

    def extra_reasoning_events(
        self, delta: Any, output: ChatStreamOutput, *, output_reasoning: bool
    ) -> Iterator[str]:
        """Hook for provider-specific reasoning."""
        return iter(())

    def retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Return a modified request body for one retry, or None."""
        return None

    def reasoning_disable_rejected(self, error: Exception) -> bool:
        """Recognize provider-owned OFF rejection formats without learning state."""
        return False

    def failure_override(self, error: Exception) -> ExecutionFailure | None:
        """Return provider-specific failure semantics, or defer to shared policy."""
        return None

    def prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return the body passed to the upstream OpenAI-compatible client."""
        return body

    def normalize_stream(self, stream: Any, _body: Mapping[str, Any]) -> Any:
        """Return the provider-specific stream view consumed by the base runner."""
        return stream

    def record_tool_call_extra_content(
        self, tool_call_id: str, extra_content: dict[str, Any]
    ) -> None:
        """Hook for providers that must replay OpenAI tool-call metadata later."""

    def tool_argument_aliases(self, body: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Return provider-specific per-tool argument aliases for this request."""
        return {}

    def cached_input_tokens(self, usage_info: object) -> int | None:
        """Return the provider's cached-input count from final Chat usage."""
        return nested_usage_int(
            usage_info,
            "prompt_tokens_details",
            "cached_tokens",
        )

    def cache_write_input_tokens(self, usage_info: object) -> int | None:
        """Return the provider's cache-write count from final Chat usage."""
        return nested_usage_int(
            usage_info,
            "prompt_tokens_details",
            "cache_write_tokens",
        )

    def anthropic_usage_fields(self, usage_info: Any) -> dict[str, int]:
        """Split standard prompt cache counts for final Anthropic usage."""
        return anthropic_input_usage_fields(
            usage_int(usage_info, "prompt_tokens"),
            cache_read_tokens=self.cached_input_tokens(usage_info),
            cache_creation_tokens=self.cache_write_input_tokens(usage_info),
        )

    def retry_after_standard_corrections(
        self, error: Exception, body: dict[str, Any], used_retry_kinds: set[str]
    ) -> dict[str, Any] | None:
        """Offer a provider correction after the standard correction sequence."""
        return None
