"""Build Codex model catalogs from the FCC model-list route."""

from collections.abc import Sequence

from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality

from .model_catalog import ClientModel

SUPPORTED_REASONING_LEVELS: list[JsonObject] = [
    {"effort": "none", "description": "Turn reasoning off"},
    {"effort": "low", "description": "Fast responses with lighter reasoning"},
    {
        "effort": "medium",
        "description": "Balances speed and reasoning depth for everyday tasks",
    },
    {"effort": "high", "description": "Greater reasoning depth for complex problems"},
    {
        "effort": "xhigh",
        "description": "Extra high reasoning depth for complex problems",
    },
    {"effort": "max", "description": "Maximum reasoning effort"},
]

CODEX_BASE_INSTRUCTIONS = (
    "You are Codex, a coding agent. Help the user understand, modify, test, "
    "and review code in their workspace. Follow the user's instructions, use "
    "tools when needed, and communicate concise progress and verification."
)


def build_codex_model_catalog(models: Sequence[ClientModel]) -> JsonObject:
    """Convert FCC `/v1/models` data into Codex `model_catalog_json` payload."""

    return {
        "models": [
            _codex_catalog_entry(model, priority=priority)
            for priority, model in enumerate(models)
        ]
    }


def _codex_catalog_entry(candidate: ClientModel, *, priority: int) -> JsonObject:
    supports_reasoning = candidate.supports_reasoning is not False
    input_modalities = candidate.input_modalities or frozenset(
        {ModelInputModality.TEXT}
    )
    context_window = (
        candidate.context_window_tokens
        if candidate.context_window_tokens is not None
        else 200000
    )
    entry: JsonObject = {
        "slug": candidate.wire_slug,
        "display_name": candidate.display_name,
        "description": "Free Claude Code provider model",
        "supported_reasoning_levels": (
            SUPPORTED_REASONING_LEVELS if supports_reasoning else []
        ),
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": priority,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "base_instructions": CODEX_BASE_INSTRUCTIONS,
        "supports_reasoning_summaries": supports_reasoning,
        "support_verbosity": True,
        "default_verbosity": "low",
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text_and_image",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_parallel_tool_calls": True,
        "supports_image_detail_original": True,
        "context_window": context_window,
        "max_context_window": context_window,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": [
            modality.value
            for modality in ModelInputModality
            if modality in input_modalities
        ],
        "supports_search_tool": True,
        "use_responses_lite": False,
    }
    if supports_reasoning:
        entry["default_reasoning_level"] = "medium"
        entry["default_reasoning_summary"] = "none"
    return entry
