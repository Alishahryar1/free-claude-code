"""Edge case tests for api/detection.py."""

import pytest

from free_claude_code.api.detection import (
    detect_safety_classifier_stop_sequence,
)
from free_claude_code.core.anthropic.models import Message, MessagesRequest


def _make_request(
    content: str, *, inline_system: str | None = None, **kwargs
) -> MessagesRequest:
    messages = []
    if inline_system is not None:
        messages.append(Message(role="system", content=inline_system))
    messages.append(Message(role="user", content=content))
    return MessagesRequest(
        model="claude-3-sonnet",
        max_tokens=kwargs.pop("max_tokens", 100),
        messages=messages,
        **kwargs,
    )


class TestDetectSafetyClassifierStopSequence:
    _LEGACY_SYSTEM = (
        "You are a security monitor. Respond with <block>yes</block> "
        "or <block>no</block>."
    )
    _CURRENT_SYSTEM = (
        "Classify the command's safety. Output <severity>N</severity> where N "
        "is the numeric severity."
    )
    _TRANSCRIPT = (
        "<transcript>\nUser: review the repo\n"
        "Bash: inspect the requested file\n</transcript>"
    )

    @pytest.mark.parametrize(
        ("system", "expected"),
        [
            (_CURRENT_SYSTEM, "</severity>"),
            (_LEGACY_SYSTEM, "</block>"),
        ],
    )
    def test_known_classifier_request_returns_owned_stop_hint(
        self, system: str, expected: str
    ) -> None:
        req = _make_request(self._TRANSCRIPT, system=system)

        assert detect_safety_classifier_stop_sequence(req) == expected

    def test_inline_system_context_uses_same_detection_path(self) -> None:
        req = _make_request(
            self._TRANSCRIPT,
            inline_system=self._CURRENT_SYSTEM,
        )

        assert detect_safety_classifier_stop_sequence(req) == "</severity>"

    def test_request_with_tools_is_not_classifier(self) -> None:
        req = _make_request(
            self._TRANSCRIPT,
            system=self._CURRENT_SYSTEM,
            tools=[{"name": "search"}],
        )

        assert detect_safety_classifier_stop_sequence(req) is None

    def test_missing_transcript_marker(self) -> None:
        req = _make_request("Classify this command", system=self._CURRENT_SYSTEM)

        assert detect_safety_classifier_stop_sequence(req) is None

    def test_missing_verdict_instruction(self) -> None:
        req = _make_request(self._TRANSCRIPT, system="Just classify this command.")

        assert detect_safety_classifier_stop_sequence(req) is None

    def test_user_content_cannot_select_classifier_format(self) -> None:
        req = _make_request(
            (
                "<transcript>\nQuoted docs say Output <severity>N</severity> and "
                "<block>yes</block>.\n</transcript>"
            ),
            system="Explain the quoted text.",
        )

        assert detect_safety_classifier_stop_sequence(req) is None

    def test_near_miss_verdict_instruction_is_not_classifier(self) -> None:
        req = _make_request(
            self._TRANSCRIPT,
            system="Output <severity>low</severity>.",
        )

        assert detect_safety_classifier_stop_sequence(req) is None

    def test_current_schema_wins_if_both_known_instructions_are_present(
        self,
    ) -> None:
        req = _make_request(
            self._TRANSCRIPT,
            system=f"{self._LEGACY_SYSTEM}\n{self._CURRENT_SYSTEM}",
        )

        assert detect_safety_classifier_stop_sequence(req) == "</severity>"
