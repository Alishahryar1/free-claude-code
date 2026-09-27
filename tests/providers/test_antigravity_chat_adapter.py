"""Request-shape normalization for the Antigravity chat adapter."""

from copy import deepcopy

from free_claude_code.providers.antigravity.chat_adapter import (
    _strip_anthropic_billing_header,
)


def _envelope(text: str, *extra_parts: dict[str, str]) -> dict[str, object]:
    return {
        "model": "gemini-test",
        "project": "project-1",
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": text}, *extra_parts],
            },
        },
    }


def test_strips_leading_anthropic_billing_metadata_without_mutating_input() -> None:
    envelope = _envelope(
        "x-anthropic-billing-header: cc_version=2.1.272.8e8; "
        "cc_entrypoint=cli;\n\n"
        "You are Claude Code, Anthropic's official CLI for Claude.",
        {"text": "Follow project rules."},
    )
    original = deepcopy(envelope)

    sanitized = _strip_anthropic_billing_header(envelope)

    assert envelope == original
    request = sanitized["request"]
    assert isinstance(request, dict)
    system = request["systemInstruction"]
    assert isinstance(system, dict)
    assert system["role"] == "system"
    assert system["parts"] == [
        {"text": "You are Claude Code, Anthropic's official CLI for Claude."},
        {"text": "Follow project rules."},
    ]


def test_strips_billing_metadata_with_crlf_separator() -> None:
    envelope = _envelope(
        "x-anthropic-billing-header: cc_version=2.1.272;\r\n\r\n"
        "Keep this system prompt."
    )

    sanitized = _strip_anthropic_billing_header(envelope)

    request = sanitized["request"]
    assert isinstance(request, dict)
    system = request["systemInstruction"]
    assert isinstance(system, dict)
    assert system["parts"] == [{"text": "Keep this system prompt."}]


def test_non_billing_system_prompt_is_left_unchanged() -> None:
    envelope = _envelope("You are a coding assistant.")

    sanitized = _strip_anthropic_billing_header(envelope)

    assert sanitized is envelope


def test_malformed_billing_header_without_prompt_separator_is_left_unchanged() -> None:
    envelope = _envelope("x-anthropic-billing-header: cc_version=2.1.272")

    sanitized = _strip_anthropic_billing_header(envelope)

    assert sanitized is envelope
