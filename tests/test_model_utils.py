import pytest

from providers.model_utils import (
    strip_provider_prefixes,
    is_claude_model,
    normalize_model_name,
    get_original_model,
)


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
    default = "target-model"
    # Strips prefix AND maps to default
    assert normalize_model_name("anthropic/claude-3-sonnet", default) == default
    assert normalize_model_name("claude-3-opus", default) == default


def test_normalize_model_name_non_claude_unchanged():
    default = "target-model"
    assert normalize_model_name("gpt-4", default) == "gpt-4"
    assert (
        normalize_model_name("openai/gpt-3.5-turbo", default) == "openai/gpt-3.5-turbo"
    )


def test_get_original_model():
    assert get_original_model("any-model") == "any-model"


def test_normalize_model_name_without_default(monkeypatch):
    monkeypatch.setenv("MODEL", "env-default-model")
    assert normalize_model_name("claude-3") == "env-default-model"


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
    "model,default,expected",
    [
        ("claude-3-sonnet", "target", "target"),
        ("anthropic/claude-3-opus", "target", "target"),
        ("gpt-4", "target", "gpt-4"),
        ("openai/gpt-3.5-turbo", "target", "openai/gpt-3.5-turbo"),
        ("", "target", ""),  # empty string is not a claude model
    ],
    ids=[
        "claude_mapped",
        "prefixed_claude",
        "non_claude",
        "prefixed_non_claude",
        "empty",
    ],
)
def test_normalize_model_name_parametrized(model, default, expected):
    """Parametrized model normalization."""
    assert normalize_model_name(model, default) == expected
