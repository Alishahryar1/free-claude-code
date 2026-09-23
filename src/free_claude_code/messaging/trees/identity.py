"""Stable and runtime identities for messaging conversation trees."""

from dataclasses import dataclass

from ..models import MessageScope


@dataclass(frozen=True, slots=True)
class TreeIdentity:
    """Customer conversation identity, independent of one runtime generation."""

    scope: MessageScope
    root_id: str


__all__ = ["TreeIdentity"]
