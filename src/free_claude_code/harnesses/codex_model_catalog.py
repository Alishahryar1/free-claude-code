"""Codex-specific model policy and native catalog serialization."""

from collections.abc import Sequence
from dataclasses import dataclass

from free_claude_code.application.model_catalog import (
    CatalogModel,
    context_window_for_client,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses.model_policy import (
    DEFAULT_REASONING_LEVEL,
    SUPPORTED_REASONING_LEVELS,
    reasoning_levels,
)

CODEX_BASE_INSTRUCTIONS = (
    "You are Codex, a coding agent. Help the user understand, modify, test, "
    "and review code in their workspace. Follow the user's instructions, use "
    "tools when needed, and communicate concise progress and verification."
)


@dataclass(frozen=True, slots=True)
class CodexModel:
    model: CatalogModel
    priority: int
    reasoning_levels: tuple[str, ...]
    default_reasoning_level: str | None


def project_codex_models(models: Sequence[CatalogModel]) -> tuple[CodexModel, ...]:
    return tuple(
        CodexModel(
            model=model,
            priority=priority,
            reasoning_levels=reasoning_levels(model),
            default_reasoning_level=(
                DEFAULT_REASONING_LEVEL
                if model.supports_reasoning is not False
                else None
            ),
        )
        for priority, model in enumerate(models)
    )


def build_codex_model_catalog(models: Sequence[CatalogModel]) -> JsonObject:
    """Serialize already ordered application records for Codex."""
    return {
        "models": [codex_model_entry(model) for model in project_codex_models(models)]
    }


def codex_model_entry(model: CodexModel) -> JsonObject:
    candidate = model.model
    supports_reasoning = bool(model.reasoning_levels)
    input_modalities = candidate.input_modalities
    if input_modalities is None or ModelInputModality.TEXT not in input_modalities:
        input_modalities = frozenset({ModelInputModality.TEXT})
    context_window = context_window_for_client(candidate)
    entry: JsonObject = {
        "slug": candidate.wire_slug,
        "display_name": candidate.display_name,
        "description": "Free Claude Code provider model",
        "supported_reasoning_levels": [
            {"effort": effort, "description": SUPPORTED_REASONING_LEVELS[effort]}
            for effort in model.reasoning_levels
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": model.priority,
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
        entry["default_reasoning_level"] = model.default_reasoning_level
        entry["default_reasoning_summary"] = "none"
    return entry
