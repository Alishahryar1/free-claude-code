"""Configuration errors exposed across storage and application boundaries."""


class ConfigConflictError(ValueError):
    """The saved configuration changed after an editor or transaction read it."""
