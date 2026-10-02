"""Small checks used where request builders consume or adapt source fields."""

from collections.abc import Mapping

from .failures import UnsupportedRequestFeature


def require_output_limit(
    original: Mapping[str, object], body: Mapping[str, object]
) -> None:
    """An explicit accepted output cap remains an upper bound on every retry."""
    fields = ("max_tokens", "max_output_tokens", "max_completion_tokens")
    limit = next(
        (original[key] for key in fields if original.get(key) is not None), None
    )
    if limit is None:
        return
    actual = next((body[key] for key in fields if body.get(key) is not None), None)
    if not (isinstance(limit, int) and isinstance(actual, int) and 0 < actual <= limit):
        raise UnsupportedRequestFeature(
            "Recovery cannot preserve the output token limit."
        )


def require_supported_fields(
    value: Mapping[str, object], supported: set[str], context: str
) -> None:
    # Unknown null controls are still unknown. Callers establish neutrality for
    # known fields in the branch that actually consumes them.
    unsupported = value.keys() - supported
    if unsupported:
        raise UnsupportedRequestFeature(
            f"{context} cannot preserve fields: {', '.join(sorted(unsupported))}."
        )


def require_preserved_body(
    before: Mapping[str, object], after: Mapping[str, object], context: str
) -> None:
    """Provider shaping may add defaults or clamp a cap, but cannot discard input."""
    for key, value in before.items():
        if key in {"max_tokens", "max_completion_tokens", "max_output_tokens"}:
            actual = after.get(key, after.get("max_completion_tokens"))
            if (
                isinstance(value, int)
                and isinstance(actual, int)
                and 0 < actual <= value
            ):
                continue
        if key not in after or after[key] != value:
            raise UnsupportedRequestFeature(f"{context} cannot preserve {key!r}.")
