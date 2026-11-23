"""Preserve the type of indirectly parametrized fixture inputs."""

import pytest


def fixture_parameter[T](request: pytest.FixtureRequest, expected: type[T]) -> T:
    value: object = getattr(request, "param", None)
    assert isinstance(value, expected), (
        f"Expected a {expected.__name__} fixture parameter"
    )
    return value
