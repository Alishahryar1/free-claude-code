"""Ordered FCC database history. Published migrations are immutable.

Add the next numbered module and append its upgrade function here. The runner
owns transactions and user_version. Migrations use explicit execute statements,
never executescript, and must not import mutable domain models or serializers.
"""

import sqlite3
from collections.abc import Callable

from . import (
    v0001_code_schema,
    v0002_code_permissions,
    v0003_code_context_usage,
    v0004_messaging_schema,
    v0005_active_code_prompts,
)

MIGRATIONS: tuple[tuple[int, Callable[[sqlite3.Connection], None]], ...] = (
    (1, v0001_code_schema.upgrade),
    (2, v0002_code_permissions.upgrade),
    (3, v0003_code_context_usage.upgrade),
    (4, v0004_messaging_schema.upgrade),
    (5, v0005_active_code_prompts.upgrade),
)
