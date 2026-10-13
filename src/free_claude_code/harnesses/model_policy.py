"""Shared effort choices for FCC-generated native client catalogs."""

from free_claude_code.application.model_catalog import CatalogModel

SUPPORTED_REASONING_LEVELS = {
    "none": "Turn reasoning off",
    "low": "Fast responses with lighter reasoning",
    "medium": "Balances speed and reasoning depth for everyday tasks",
    "high": "Greater reasoning depth for complex problems",
    "xhigh": "Extra high reasoning depth for complex problems",
    "max": "Maximum reasoning effort",
}
DEFAULT_REASONING_LEVEL = "medium"


def reasoning_levels(model: CatalogModel) -> tuple[str, ...]:
    return (
        tuple(SUPPORTED_REASONING_LEVELS)
        if model.supports_reasoning is not False
        else ()
    )
