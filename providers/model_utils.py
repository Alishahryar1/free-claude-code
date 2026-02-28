"""Model name normalization utilities.

Centralizes model name mapping logic to avoid duplication across the codebase.
"""

# Provider prefixes to strip from model names
_PROVIDER_PREFIXES = ["anthropic/", "openai/", "gemini/"]

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


def normalize_model_name(model: str, settings) -> str:
    """
    Normalize a model name by stripping prefixes and mapping to default if needed.

    This is the central function for model name normalization across the API.
    It strips provider prefixes and maps Claude model names to the configured model.

    Supports per-Claude-family mapping:
    - haiku_model -> for Haiku models (claude-3-haiku, etc.)
    - sonnet_model -> for Sonnet models (claude-3-sonnet, etc.)
    - opus_model -> for Opus models (claude-3-opus, etc.)
    - Falls back to model_name (default MODEL) for other Claude models

    Args:
        model: The model name (may include provider prefix)
        settings: Settings object with model configuration

    Returns:
        Normalized model name (original if not a Claude model, mapped if Claude)
    """
    # Strip provider prefixes
    clean = strip_provider_prefixes(model)

    # Map Claude models based on family
    if is_claude_model(clean):
        clean_lower = clean.lower()

        # Check for specific Claude family mappings
        if "haiku" in clean_lower and settings.haiku_model:
            return settings.haiku_model
        if "sonnet" in clean_lower and settings.sonnet_model:
            return settings.sonnet_model
        if "opus" in clean_lower and settings.opus_model:
            return settings.opus_model

        # Fall back to default model for other Claude models
        return settings.model_name

    return model


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
