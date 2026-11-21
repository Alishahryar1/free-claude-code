"""Small shared protocol utility helpers."""


def set_if_not_none[T](body: dict[str, T], key: str, value: T | None) -> None:
    """Set ``body[key]`` only when value is not None."""
    if value is not None:
        body[key] = value
