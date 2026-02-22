"""Model name normalization utilities.

Centralizes model name mapping logic to avoid duplication across the codebase.
"""

import os

# Provider prefixes to strip from model names
_PROVIDER_PREFIXES = ["anthropic/", "openai/", "gemini/", "nvidia_nim/"]

# Claude model identifiers
_CLAUDE_IDENTIFIERS = ["haiku", "sonnet", "opus", "claude"]


def strip_provider_prefixes(model: str) -> str:
    """
    Strip provider prefixes from model name.

    Args:
        model: The model name, possibly with prefix

    Returns:
        Model name without provider prefix
    """
    for prefix in _PROVIDER_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def is_claude_model(model: str) -> bool:
    """
    Check if a model name identifies as a Claude model.

    Args:
        model: The (prefix-stripped) model name

    Returns:
        True if this is a Claude model
    """
    model_lower = model.lower()
    return any(name in model_lower for name in _CLAUDE_IDENTIFIERS)


def normalize_model_name(model: str, default_model: str | None = None) -> str:
    """
    Normalize a model name by stripping prefixes and mapping to default if needed.

    This is the central function for model name normalization across the API.
    It strips provider prefixes and maps Claude model names to the configured model.

    Args:
        model: The model name (may include provider prefix)
        default_model: The default model to use for Claude models.
                       If None, uses settings.model from config.

    Returns:
        Normalized model name (original if not a Claude model, mapped if Claude)
    """
    # Strip provider prefixes
    clean = strip_provider_prefixes(model)

    # Map Claude models to default
    if is_claude_model(clean):
        if default_model is None:
            # Use environment/config default
            default_model = os.getenv("MODEL", "moonshotai/kimi-k2-thinking")
        return default_model

    return clean


def get_original_model(model: str) -> str:
    """
    Get the original model name, storing it before normalization.

    Convenience function that returns the input unchanged, intended to be
    called alongside normalize_model_name to capture the original.

    Args:
        model: The model name

    Returns:
        The model name unchanged (for documentation purposes)
    """
    return model
