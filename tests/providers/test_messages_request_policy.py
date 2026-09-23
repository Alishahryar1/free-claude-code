"""Native controls stay metadata-driven and preserve exact caller budgets."""

import pytest

from free_claude_code.core.anthropic.models import ThinkingConfig
from free_claude_code.core.anthropic.native import NativeMessagesError
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.anthropic_messages.request_policy import (
    MessagesModelCapabilities,
    resolve_messages_options,
)


def test_unknown_defaults_do_not_invent_thinking_or_output_cap() -> None:
    options = resolve_messages_options(
        model="anything", max_tokens=None, reasoning=ReasoningPolicy()
    )
    assert options.max_tokens == 8192
    assert options.thinking is None
    assert options.output_effort is None


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_output_limit_rejected(limit: int) -> None:
    with pytest.raises(NativeMessagesError, match="positive integer"):
        resolve_messages_options(
            model="m", max_tokens=limit, reasoning=ReasoningPolicy()
        )


def test_output_cap_clamps_generation_but_never_an_exact_thinking_budget() -> None:
    caps = MessagesModelCapabilities(
        max_output_tokens=2048, adaptive_thinking="optional"
    )
    with pytest.raises(NativeMessagesError, match="below max_tokens"):
        resolve_messages_options(
            model="m",
            max_tokens=4096,
            capabilities=caps,
            reasoning=ReasoningPolicy.on(budget_tokens=2048),
        )
    options = resolve_messages_options(
        model="m",
        max_tokens=4096,
        capabilities=caps,
        reasoning=ReasoningPolicy.on(budget_tokens=1024),
    )
    assert options.max_tokens == 2048
    assert options.thinking == {"type": "enabled", "budget_tokens": 1024}


def test_adaptive_only_rejects_exact_budget_but_allows_off() -> None:
    caps = MessagesModelCapabilities(adaptive_thinking="required")
    with pytest.raises(NativeMessagesError, match="requires adaptive"):
        resolve_messages_options(
            model="m",
            max_tokens=None,
            capabilities=caps,
            reasoning=ReasoningPolicy.on(budget_tokens=1024),
        )
    options = resolve_messages_options(
        model="m", max_tokens=None, capabilities=caps, reasoning=ReasoningPolicy.off()
    )
    assert options.thinking == {"type": "disabled"}


def test_explicit_native_mode_works_without_model_heuristics() -> None:
    options = resolve_messages_options(
        model="unfamiliar-name",
        max_tokens=4096,
        reasoning=ReasoningPolicy.on(budget_tokens=1536),
        thinking=ThinkingConfig(type="enabled", budget_tokens=1536, display="omitted"),
    )
    assert options.thinking == {
        "type": "enabled",
        "budget_tokens": 1536,
        "display": "omitted",
    }
    options = resolve_messages_options(
        model="another-name",
        max_tokens=4096,
        reasoning=ReasoningPolicy.on(),
        thinking=ThinkingConfig(type="adaptive"),
    )
    assert options.thinking == {"type": "adaptive"}


def test_unknown_mode_cannot_be_guessed_from_effort_support() -> None:
    with pytest.raises(NativeMessagesError, match="thinking mode"):
        resolve_messages_options(
            model="claude-something",
            max_tokens=None,
            capabilities=MessagesModelCapabilities(supports_output_effort=True),
            reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
        )


def test_advertised_effort_without_forced_thinking_works() -> None:
    options = resolve_messages_options(
        model="m",
        max_tokens=None,
        capabilities=MessagesModelCapabilities(supported_efforts=("low", "high")),
        reasoning=ReasoningPolicy(effort=ReasoningEffort.MINIMAL),
    )
    assert options.output_effort == "low"
    assert options.thinking is None
    with pytest.raises(NativeMessagesError, match="does not support effort"):
        resolve_messages_options(
            model="m",
            max_tokens=None,
            capabilities=MessagesModelCapabilities(supported_efforts=("low", "high")),
            reasoning=ReasoningPolicy(effort=ReasoningEffort.MAX),
        )


def test_manual_effort_uses_budget_and_minimum() -> None:
    options = resolve_messages_options(
        model="m",
        max_tokens=1500,
        capabilities=MessagesModelCapabilities(
            adaptive_thinking="unsupported", supports_output_effort=False
        ),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.LOW),
    )
    assert options.thinking == {"type": "enabled", "budget_tokens": 1024}
    assert options.output_effort is None


@pytest.mark.parametrize("raw", ["bogus", 1, {"mode": "high"}])
def test_client_on_cannot_hide_invalid_output_effort(raw: object) -> None:
    with pytest.raises(NativeMessagesError, match="client output effort"):
        resolve_messages_options(
            model="m",
            max_tokens=None,
            output_effort=raw,
            reasoning=ReasoningPolicy.on(budget_tokens=1536),
            thinking=ThinkingConfig(type="enabled", budget_tokens=1536),
        )


def test_off_overrides_native_hint_without_dropping_independent_effort() -> None:
    options = resolve_messages_options(
        model="m",
        max_tokens=None,
        reasoning=ReasoningPolicy(
            control=ReasoningControl.OFF, effort=ReasoningEffort.LOW
        ),
        capabilities=MessagesModelCapabilities(supported_efforts=("low",)),
        thinking=ThinkingConfig(type="enabled", budget_tokens=1024, display="omitted"),
    )
    assert options.thinking == {"type": "disabled"}
    assert options.output_effort == "low"


def test_unknown_thinking_options_and_adaptive_budget_rejected() -> None:
    with pytest.raises(NativeMessagesError, match="Unsupported native thinking option"):
        resolve_messages_options(
            model="m",
            max_tokens=None,
            reasoning=ReasoningPolicy(),
            thinking=ThinkingConfig.model_validate({"type": "adaptive", "mystery": 1}),
        )
    with pytest.raises(NativeMessagesError, match="exact token budget"):
        resolve_messages_options(
            model="m",
            max_tokens=None,
            reasoning=ReasoningPolicy(),
            thinking=ThinkingConfig(type="adaptive", budget_tokens=1024),
        )


def test_configured_off_does_not_resurrect_raw_client_effort() -> None:
    options = resolve_messages_options(
        model="m",
        max_tokens=None,
        reasoning=ReasoningPolicy.off(),
        output_effort="high",
        capabilities=MessagesModelCapabilities(supported_efforts=("high",)),
    )
    assert options.thinking == {"type": "disabled"}
    assert options.output_effort is None


@pytest.mark.parametrize("effort", ["minimal", "bogus"])
def test_effort_support_boolean_does_not_authorize_unknown_native_values(
    effort: str,
) -> None:
    with pytest.raises(NativeMessagesError, match="effort"):
        resolve_messages_options(
            model="m",
            max_tokens=None,
            reasoning=ReasoningPolicy(),
            output_effort=effort,
            capabilities=MessagesModelCapabilities(supports_output_effort=True),
        )


def test_manual_minimal_effort_is_represented_only_by_numeric_budget() -> None:
    options = resolve_messages_options(
        model="m",
        max_tokens=None,
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL),
        capabilities=MessagesModelCapabilities(
            adaptive_thinking="unsupported", supports_output_effort=False
        ),
    )
    assert options.thinking == {"type": "enabled", "budget_tokens": 1024}
    assert options.output_effort is None
