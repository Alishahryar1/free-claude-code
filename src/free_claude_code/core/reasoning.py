"""Provider-neutral reasoning intent."""

from dataclasses import dataclass
from enum import StrEnum


class ReasoningControl(StrEnum):
    """Whether a request explicitly controls reasoning computation."""

    DEFAULT = "default"
    OFF = "off"
    PREFER_OFF = "prefer_off"
    ON = "on"


class ReasoningCapability(StrEnum):
    """Explicit provider facts; absent computation controls remain unknown."""

    UNKNOWN = "unknown"
    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


class ReasoningEffort(StrEnum):
    """Named reasoning effort understood at the FCC application boundary."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"

    @property
    def budget_tokens(self) -> int:
        """Return FCC's numeric token budget for this effort."""

        return _EFFORT_BUDGET_TOKENS[self]


_EFFORT_BUDGET_TOKENS = {
    ReasoningEffort.MINIMAL: 512,
    ReasoningEffort.LOW: 512,
    ReasoningEffort.MEDIUM: 1_024,
    ReasoningEffort.HIGH: 2_048,
    ReasoningEffort.XHIGH: 4_096,
    ReasoningEffort.MAX: 8_192,
}


@dataclass(frozen=True, slots=True)
class ReasoningPolicy:
    """Resolved client and configuration intent passed to one provider.

    ``control`` and ``effort`` remain independent because clients may set an
    overall effort while separately disabling extended thinking. Providers
    translate the representable subset without changing the original intent.
    """

    control: ReasoningControl = ReasoningControl.DEFAULT
    effort: ReasoningEffort | None = None
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.budget_tokens is not None and (
            not isinstance(self.budget_tokens, int)
            or isinstance(self.budget_tokens, bool)
            or self.budget_tokens <= 0
        ):
            raise ValueError("Reasoning budget must be a positive integer.")
        if self.budget_tokens is not None and self.control is not ReasoningControl.ON:
            raise ValueError("A reasoning budget requires reasoning control to be on.")

    @classmethod
    def provider_default(cls) -> ReasoningPolicy:
        """Leave reasoning computation to the provider."""

        return cls()

    @classmethod
    def off(cls) -> ReasoningPolicy:
        """Explicitly disable reasoning computation and output."""

        return cls(control=ReasoningControl.OFF)

    @classmethod
    def prefer_off(cls) -> ReasoningPolicy:
        """Prefer disabling computation, allowing provider defaults if necessary."""
        return cls(control=ReasoningControl.PREFER_OFF)

    @classmethod
    def on(
        cls,
        *,
        effort: ReasoningEffort | None = None,
        budget_tokens: int | None = None,
    ) -> ReasoningPolicy:
        """Explicitly enable reasoning with optional client controls."""

        return cls(
            control=ReasoningControl.ON,
            effort=effort,
            budget_tokens=budget_tokens,
        )

    @property
    def output_enabled(self) -> bool:
        """Allow normalized reasoning; classifier ingress owns its public filter."""

        return self.control is not ReasoningControl.OFF

    @property
    def requests_reasoning(self) -> bool:
        """Return whether the request explicitly asks the provider to reason."""

        return self.control is not ReasoningControl.OFF and (
            self.control is ReasoningControl.ON
            or self.effort is not None
            or self.budget_tokens is not None
        )

    @property
    def numeric_budget_tokens(self) -> int | None:
        """Express this intent as an exact or FCC-mapped numeric budget."""

        if self.control is ReasoningControl.OFF:
            return None
        if self.budget_tokens is not None:
            return self.budget_tokens
        if self.effort is None:
            return None
        return self.effort.budget_tokens


DEFAULT_REASONING_POLICY = ReasoningPolicy.provider_default()
