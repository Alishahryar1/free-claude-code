"""Chat source observations, independent of the public response protocol."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from .history_replay import ReplayRecord
from .json_types import JsonObject


@dataclass(frozen=True, slots=True)
class ChatStreamUsage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cache_write_tokens: int | None = None
    reasoning_tokens: int = 0
    anthropic_fields: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChatToolObservation:
    tool_id: str
    name: str
    arguments: str = ""
    extra_content: JsonObject | None = None


@dataclass(frozen=True, slots=True)
class ChatChange:
    """A source boundary, fragment, completed call or structured reasoning record."""

    kind: str
    text: str = ""
    tool_index: int = 0
    tool: ChatToolObservation | None = None
    group_id: str = ""
    record: ReplayRecord | None = None
    usage: ChatStreamUsage | None = None


def responses_usage(usage: ChatStreamUsage) -> JsonObject:
    cached_tokens = usage.cached_tokens
    if (
        not isinstance(cached_tokens, int)
        or isinstance(cached_tokens, bool)
        or not 0 <= cached_tokens <= usage.input_tokens
    ):
        cached_tokens = 0
    input_details = {"cached_tokens": cached_tokens}
    written = usage.cache_write_tokens
    if (
        isinstance(written, int)
        and not isinstance(written, bool)
        and 0 <= written <= usage.input_tokens - cached_tokens
    ):
        input_details["cache_write_tokens"] = written
    return {
        "input_tokens": usage.input_tokens,
        "input_tokens_details": input_details,
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": max(0, min(usage.reasoning_tokens, usage.output_tokens))
        },
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }
