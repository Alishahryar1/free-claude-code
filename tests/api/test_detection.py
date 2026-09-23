"""Edge case tests for api/detection.py."""

from unittest.mock import patch

import pytest

from free_claude_code.api.detection import (
    detect_safety_classifier_stop_sequence,
    is_filepath_extraction_request,
    is_prefix_detection_request,
    is_quota_check_request,
    is_title_generation_request,
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


def test_quota_detection_ignores_inline_system_context() -> None:
    request = _make_request(
        "Check my quota", inline_system="Current request context", max_tokens=1
    )

    assert is_quota_check_request(request) is True


def test_title_detection_reads_inline_system_context() -> None:
    request = _make_request(
        "Summarize this session",
        inline_system=(
            "Generate a concise, sentence-case title for this coding session. "
            'Return JSON with a single "title" field.'
        ),
    )

    assert is_title_generation_request(request) is True


class TestIsPrefixDetectionRequest:
    def test_inline_system_context_does_not_hide_single_user_turn(self):
        req = _make_request(
            "<policy_spec> Command: git status",
            inline_system="Current request context",
        )

        assert is_prefix_detection_request(req) == (True, "git status")

    def test_output_marker_handling(self):
        """Content with Command: but Output: after cmd_start; output has < or \\n\\n."""
        content = "<policy_spec> Command:\nls -la\nOutput:\na.txt\nb.txt\n\nmore"
        req = _make_request(content)
        is_req, cmd = is_prefix_detection_request(req)
        assert is_req is True
        assert "ls -la" in cmd

    def test_prefix_detection_with_empty_command_section(self):
        """Command: at end with no content returns empty command."""
        req = _make_request("<policy_spec> Command: ")
        is_req, cmd = is_prefix_detection_request(req)
        assert is_req is True
        assert cmd == ""

    def test_exception_in_try_returns_false(self):
        """Exception in try block (e.g. content slice) returns False, ''."""
        req = _make_request("<policy_spec> Command: x")

        # Return object that raises when sliced - triggers except in is_prefix_detection_request
        class BadStr(str):
            def __getitem__(self, key):
                raise TypeError("bad slice")

        with patch(
            "free_claude_code.api.detection.extract_text_from_content",
            return_value=BadStr("<policy_spec> Command: x"),
        ):
            is_req, cmd = is_prefix_detection_request(req)
        assert is_req is False
        assert cmd == ""


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


class TestIsFilepathExtractionRequest:
    def test_inline_system_context_preserves_filepath_detection(self):
        req = _make_request(
            "Command:\nls\nOutput:\na.txt",
            inline_system="Extract any file paths that this command reads or modifies.",
        )

        assert is_filepath_extraction_request(req) == (True, "ls", "a.txt")

    def test_output_marker_minus_one_returns_false(self):
        """Output: not found after Command: returns False."""
        content = "Command:\nls\nfilepaths"
        req = _make_request(content)
        is_fp, cmd, out = is_filepath_extraction_request(req)
        assert is_fp is False
        assert cmd == ""
        assert out == ""

    def test_output_has_angle_bracket_splits(self):
        """Output containing < is split and first part used."""
        content = "Command:\nls\nOutput:\na.txt b.txt <extra>\nfilepaths"
        req = _make_request(content)
        is_fp, _cmd, out = is_filepath_extraction_request(req)
        assert is_fp is True
        assert "<" not in out
        assert out == "a.txt b.txt"

    def test_output_has_double_newline_splits(self):
        """Output containing \\n\\n is split and first part used."""
        content = "Command:\nls\nOutput:\na.txt\nb.txt\n\nmore text\nfilepaths"
        req = _make_request(content)
        is_fp, _cmd, out = is_filepath_extraction_request(req)
        assert is_fp is True
        assert "more" not in out
