"""Native launch descriptions; execution belongs to each caller."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class NativeCheck:
    """A fixed native probe; its adapter owns output interpretation."""

    args: tuple[str, ...]
    accepts: Callable[[str], bool]
    failure_message: str
    timeout_seconds: float = 5.0


@dataclass(frozen=True, slots=True)
class PreparedLaunch:
    command: list[str] = field(repr=False)
    env: Mapping[str, str] = field(repr=False)
    activation_check: NativeCheck | None = None
