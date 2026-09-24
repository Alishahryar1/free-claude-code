"""Configuration management."""

from . import provider_catalog as _provider_catalog
from .antigravity_catalog import register_antigravity_catalog
from .loader import clear_settings_cache, get_settings
from .settings import Settings

register_antigravity_catalog(_provider_catalog)

__all__ = [
    "Settings",
    "clear_settings_cache",
    "get_settings",
]
