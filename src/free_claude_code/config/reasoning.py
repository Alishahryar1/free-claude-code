"""User-configurable reasoning policy values."""

from enum import StrEnum


class ReasoningPreference(StrEnum):
    """Configuration choice applied before provider translation."""

    INHERIT = "inherit"
    OFF = "off"
    CLIENT = "client"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


ROOT_REASONING_PREFERENCES = tuple(
    preference
    for preference in ReasoningPreference
    if preference is not ReasoningPreference.INHERIT
)
ROUTE_REASONING_PREFERENCES = tuple(ReasoningPreference)
