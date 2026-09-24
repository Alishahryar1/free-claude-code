"""Native OpenCode version contract shared by launch and integration setup."""

import re

STABLE_VERSION_PATTERN = re.compile(
    r"(?m)^\s*(?:opencode(?:\s+version)?\s+)?v?"
    r"2\.\d+\.\d+(?:\+[0-9A-Za-z.-]+)?\s*$"
)
