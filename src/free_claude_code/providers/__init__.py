"""Shared provider lifecycle contracts.

Ordinary OpenAI-compatible vendors are immutable profiles. Concrete adapter
classes exist only for providers with stateful or algorithmic behavior.
"""

from .base import BaseProvider, ProviderConfig

__all__ = [
    "BaseProvider",
    "ProviderConfig",
]
