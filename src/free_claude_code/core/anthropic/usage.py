"""Anthropic terminal usage accounting helpers."""


def anthropic_input_usage_fields(
    total_tokens: int | None,
    *,
    cache_read_tokens: int | None,
    cache_creation_tokens: int | None,
) -> dict[str, int]:
    """Partition an inclusive input total into trustworthy Anthropic fields."""
    if (
        not isinstance(total_tokens, int)
        or isinstance(total_tokens, bool)
        or total_tokens < 0
    ):
        return {}

    fields: dict[str, int] = {}
    remaining_tokens = total_tokens
    if (
        isinstance(cache_read_tokens, int)
        and not isinstance(cache_read_tokens, bool)
        and 0 <= cache_read_tokens <= remaining_tokens
    ):
        fields["cache_read_input_tokens"] = cache_read_tokens
        remaining_tokens -= cache_read_tokens

    if (
        isinstance(cache_creation_tokens, int)
        and not isinstance(cache_creation_tokens, bool)
        and 0 <= cache_creation_tokens <= remaining_tokens
    ):
        fields["cache_creation_input_tokens"] = cache_creation_tokens
        remaining_tokens -= cache_creation_tokens

    if not fields:
        return {}
    return {"input_tokens": remaining_tokens, **fields}
