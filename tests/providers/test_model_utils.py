from unittest.mock import Mock

import pytest

from providers.model_utils import (
    get_original_model,
    is_claude_model,
    normalize_model_name,
    strip_provider_prefixes,
)


def _make_mock_settings(
    model_name: str = "default-model",
    haiku_model: str | None = None,
    sonnet_model: str | None = None,
    opus_model: str | None = None,
):
    """Create a mock Settings object for testing."""
    settings = Mock()
    settings.model_name = model_name
    settings.haiku_model = haiku_model
    settings.sonnet_model = sonnet_model
    settings.opus_model = opus_model
    return settings


def test_strip_provider_prefixes():
    assert strip_provider_prefixes("anthropic/claude-3") == "claude-3"
    assert strip_provider_prefixes("openai/gpt-4") == "gpt-4"
    assert strip_provider_prefixes("gemini/gemini-pro") == "gemini-pro"
    assert strip_provider_prefixes("no-prefix") == "no-prefix"


def test_is_claude_model():
    assert is_claude_model("claude-3-sonnet") is True
    assert is_claude_model("claude-3-opus") is True
    assert is_claude_model("claude-3-haiku") is True
    assert is_claude_model("claude-2.1") is True
    assert is_claude_model("gpt-4") is False
    assert is_claude_model("gemini-pro") is False


def test_normalize_model_name_claude_maps_to_default():
    settings = _make_mock_settings(model_name="target-model")
    # Strips prefix AND maps to default
    assert normalize_model_name("anthropic/claude-3-sonnet", settings) == "target-model"
    assert normalize_model_name("claude-3-opus", settings) == "target-model"


def test_normalize_model_name_non_claude_unchanged():
    settings = _make_mock_settings(model_name="target-model")
    assert normalize_model_name("gpt-4", settings) == "gpt-4"
    assert (
        normalize_model_name("openai/gpt-3.5-turbo", settings) == "openai/gpt-3.5-turbo"
    )


def test_get_original_model():
    assert get_original_model("any-model") == "any-model"


def test_normalize_model_name_haiku_mapping():
    """Test Haiku models map to haiku_model when set."""
    settings = _make_mock_settings(
        model_name="default-model",
        haiku_model="haiku-target",
    )
    assert normalize_model_name("claude-3-haiku", settings) == "haiku-target"
    assert normalize_model_name("claude-3.5-haiku", settings) == "haiku-target"


def test_normalize_model_name_sonnet_mapping():
    """Test Sonnet models map to sonnet_model when set."""
    settings = _make_mock_settings(
        model_name="default-model",
        sonnet_model="sonnet-target",
    )
    assert normalize_model_name("claude-3-sonnet", settings) == "sonnet-target"
    assert normalize_model_name("claude-3.5-sonnet", settings) == "sonnet-target"


def test_normalize_model_name_opus_mapping():
    """Test Opus models map to opus_model when set."""
    settings = _make_mock_settings(
        model_name="default-model",
        opus_model="opus-target",
    )
    assert normalize_model_name("claude-3-opus", settings) == "opus-target"


def test_normalize_model_name_fallback_to_default():
    """Test that Claude models without specific mapping fall back to default."""
    settings = _make_mock_settings(
        model_name="default-model",
        haiku_model="haiku-target",
    )
    # Sonnet and Opus should fall back to default when not set
    assert normalize_model_name("claude-3-sonnet", settings) == "default-model"
    assert normalize_model_name("claude-3-opus", settings) == "default-model"


def test_normalize_model_name_all_three_set():
    """Test with all three Claude model mappings set."""
    settings = _make_mock_settings(
        model_name="default-model",
        haiku_model="haiku-target",
        sonnet_model="sonnet-target",
        opus_model="opus-target",
    )
    assert normalize_model_name("claude-3-haiku", settings) == "haiku-target"
    assert normalize_model_name("claude-3-sonnet", settings) == "sonnet-target"
    assert normalize_model_name("claude-3-opus", settings) == "opus-target"
    # Other Claude models fall back to default
    assert normalize_model_name("claude-2.1", settings) == "default-model"


def test_normalize_model_name_none_specific_mappings():
    """Test with all specific mappings set to None."""
    settings = _make_mock_settings(
        model_name="default-model",
        haiku_model=None,
        sonnet_model=None,
        opus_model=None,
    )
    # All Claude models should map to default
    assert normalize_model_name("claude-3-haiku", settings) == "default-model"
    assert normalize_model_name("claude-3-sonnet", settings) == "default-model"
    assert normalize_model_name("claude-3-opus", settings) == "default-model"


# --- Parametrized Edge Case Tests ---


@pytest.mark.parametrize(
    "model,expected",
    [
        ("anthropic/claude-3", "claude-3"),
        ("openai/gpt-4", "gpt-4"),
        ("gemini/gemini-pro", "gemini-pro"),
        ("no-prefix", "no-prefix"),
        ("", ""),
        ("anthropic/", ""),
        ("anthropic/openai/nested", "openai/nested"),
    ],
    ids=[
        "anthropic",
        "openai",
        "gemini",
        "no_prefix",
        "empty_string",
        "prefix_only",
        "nested_prefix",
    ],
)
def test_strip_provider_prefixes_parametrized(model, expected):
    """Parametrized prefix stripping with edge cases."""
    assert strip_provider_prefixes(model) == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-3-sonnet", True),
        ("claude-3-opus", True),
        ("claude-3-haiku", True),
        ("claude-2.1", True),
        ("gpt-4", False),
        ("gemini-pro", False),
        ("", False),
        ("my-claude-wrapper", True),  # "claude" as substring
        ("CLAUDE-3-SONNET", True),  # case insensitive
        ("sonnet-v2", True),  # "sonnet" identifier without "claude"
        ("haiku-model", True),  # "haiku" identifier
    ],
    ids=[
        "sonnet",
        "opus",
        "haiku",
        "claude2",
        "gpt4",
        "gemini",
        "empty",
        "claude_substring",
        "uppercase",
        "sonnet_standalone",
        "haiku_standalone",
    ],
)
def test_is_claude_model_parametrized(model, expected):
    """Parametrized Claude model detection with edge cases."""
    assert is_claude_model(model) is expected


@pytest.mark.parametrize(
    "model,settings_config,expected",
    [
        # Claude models with default mapping
        ("claude-3-sonnet", {"model_name": "target"}, "target"),
        ("anthropic/claude-3-opus", {"model_name": "target"}, "target"),
        # Non-Claude models unchanged
        ("gpt-4", {"model_name": "target"}, "gpt-4"),
        ("openai/gpt-3.5-turbo", {"model_name": "target"}, "openai/gpt-3.5-turbo"),
        # Empty string edge case
        ("", {"model_name": "target"}, ""),
        # Specific Haiku mapping
        (
            "claude-3-haiku",
            {"model_name": "default", "haiku_model": "haiku-target"},
            "haiku-target",
        ),
        # Specific Sonnet mapping
        (
            "claude-3-sonnet",
            {"model_name": "default", "sonnet_model": "sonnet-target"},
            "sonnet-target",
        ),
        # Specific Opus mapping
        (
            "claude-3-opus",
            {"model_name": "default", "opus_model": "opus-target"},
            "opus-target",
        ),
    ],
    ids=[
        "claude_mapped",
        "prefixed_claude",
        "non_claude",
        "prefixed_non_claude",
        "empty",
        "haiku_specific",
        "sonnet_specific",
        "opus_specific",
    ],
)
def test_normalize_model_name_parametrized(model, settings_config, expected):
    """Parametrized model normalization with various settings."""
    settings = _make_mock_settings(**settings_config)
    assert normalize_model_name(model, settings) == expected
