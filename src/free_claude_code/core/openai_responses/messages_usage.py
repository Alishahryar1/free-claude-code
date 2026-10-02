"""Map reported Messages token usage to Responses usage."""

from collections.abc import Mapping

from free_claude_code.core.json_types import JsonObject, JsonValue


class NativeMessagesUsage:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._cache_creation: JsonObject | None = None
        self._thinking_tokens: int | None = None
        self._final_output_seen = False

    def update(self, value: JsonValue, *, final: bool = False) -> None:
        if final:
            self._thinking_tokens = None
            output = value.get("output_tokens") if isinstance(value, Mapping) else None
            self._final_output_seen = (
                isinstance(output, int) and not isinstance(output, bool) and output >= 0
            )
        if not isinstance(value, Mapping):
            return
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "output_tokens",
        ):
            count = value.get(key)
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                self._counts[key] = count
        creation = value.get("cache_creation")
        if isinstance(creation, Mapping):
            self._cache_creation = {
                key: count
                for key, count in creation.items()
                if isinstance(count, int) and not isinstance(count, bool) and count >= 0
            }
        details = value.get("output_tokens_details")
        if isinstance(details, Mapping):
            thinking = details.get("thinking_tokens")
            if (
                isinstance(thinking, int)
                and not isinstance(thinking, bool)
                and thinking >= 0
            ):
                self._thinking_tokens = thinking

    def payload(self, *, require_final: bool = False) -> JsonObject | None:
        if require_final and not self._final_output_seen:
            return None
        if "input_tokens" not in self._counts or "output_tokens" not in self._counts:
            return None
        cached = self._counts.get("cache_read_input_tokens", 0)
        created = self._counts.get("cache_creation_input_tokens", 0)
        input_tokens = self._counts["input_tokens"] + cached + created
        output_tokens = self._counts["output_tokens"]
        details: JsonObject = {}
        if "cache_read_input_tokens" in self._counts:
            details["cached_tokens"] = cached
        if "cache_creation_input_tokens" in self._counts:
            details["cache_creation_tokens"] = created
        if self._cache_creation is not None:
            details["cache_creation"] = dict(self._cache_creation)
        usage: JsonObject = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if details:
            usage["input_tokens_details"] = details
        if self._thinking_tokens is not None and self._thinking_tokens <= output_tokens:
            usage["output_tokens_details"] = {"reasoning_tokens": self._thinking_tokens}
        return usage
