from unittest.mock import MagicMock

import pytest

from free_claude_code.api.command_utils import extract_filepaths_from_command
from free_claude_code.api.detection import (
    is_filepath_extraction_request,
    is_suggestion_mode_request,
)
from free_claude_code.core.anthropic.models import Message, MessagesRequest


def _mk_req(messages, tools=None, system=None):
    req = MagicMock(spec=MessagesRequest)
    req.messages = messages
    req.tools = tools
    req.system = system
    return req


def _mk_msg(role: str, content):
    msg = MagicMock(spec=Message)
    msg.role = role
    msg.content = content
    return msg


class TestSuggestionMode:
    def test_detects_suggestion_mode_in_any_user_message(self):
        req = _mk_req(
            [
                _mk_msg("assistant", "ignore"),
                _mk_msg("user", "Hello\n[SUGGESTION MODE: on]\nworld"),
            ]
        )
        assert is_suggestion_mode_request(req) is True

    def test_suggestion_mode_ignores_non_user_messages(self):
        req = _mk_req([_mk_msg("assistant", "[SUGGESTION MODE: on]")])
        assert is_suggestion_mode_request(req) is False


class TestFilepathExtractionDetection:
    def test_rejects_when_tools_present(self):
        msg = _mk_msg(
            "user",
            "Command: cat foo.txt\nOutput: hi\n\nPlease extract <filepaths>.",
        )
        req = _mk_req([msg], tools=[{"name": "search"}])
        ok, cmd, out = is_filepath_extraction_request(req)
        assert (ok, cmd, out) == (False, "", "")

    def test_rejects_when_missing_output_marker(self):
        msg = _mk_msg(
            "user",
            "Command: cat foo.txt\n(no output marker)\n<filepaths>",
        )
        req = _mk_req([msg], tools=None)
        ok, cmd, out = is_filepath_extraction_request(req)
        assert (ok, cmd, out) == (False, "", "")

    def test_rejects_when_not_asking_for_filepaths(self):
        msg = _mk_msg("user", "Command: cat foo.txt\nOutput: hi")
        req = _mk_req([msg], tools=None)
        ok, cmd, out = is_filepath_extraction_request(req)
        assert (ok, cmd, out) == (False, "", "")

    def test_detects_filepath_extraction_via_system_block(self):
        """Command: + Output: in user, no filepaths in user; system has extract instructions."""
        msg = _mk_msg("user", "Command: ls\nOutput: avazu-ctr\nfree-claude-code")
        req = _mk_req(
            [msg],
            tools=None,
            system="Extract any file paths that this command reads or modifies.",
        )
        ok, cmd, out = is_filepath_extraction_request(req)
        assert ok is True
        assert cmd == "ls"
        assert "avazu-ctr" in out
        assert "free-claude-code" in out

    def test_extracts_command_and_output_and_cleans_output(self):
        msg = _mk_msg(
            "user",
            "Command: cat foo.txt\n"
            "Output: line1\nline2\n\n"
            "Please extract <filepaths>.\n"
            "<next_section>ignore me</next_section>",
        )
        req = _mk_req([msg], tools=None)
        ok, cmd, out = is_filepath_extraction_request(req)
        assert ok is True
        assert cmd == "cat foo.txt"
        assert out == "line1\nline2"


class TestExtractFilepathsFromCommand:
    @pytest.mark.parametrize(
        "command,expected_paths",
        [
            ("ls -la", []),
            ("dir .", []),
            ("cat foo.txt", ["foo.txt"]),
            ("cat -n foo.txt bar.md", ["foo.txt", "bar.md"]),
            ("type C:\\tmp\\a.txt", ["C:\\tmp\\a.txt"]),
            ("grep pattern file1.txt file2.txt", ["file1.txt", "file2.txt"]),
            ("grep -n pattern file.txt", ["file.txt"]),
            ("grep -e pattern file.txt", ["file.txt"]),
            ("unknowncmd arg1 arg2", []),
            ("", []),
        ],
        ids=[
            "listing_ls",
            "listing_dir",
            "read_cat",
            "read_cat_flags",
            "read_type_windows_path",
            "grep_simple",
            "grep_with_flag",
            "grep_with_e",
            "unknown",
            "empty",
        ],
    )
    def test_extracts_expected_paths(self, command, expected_paths):
        result = extract_filepaths_from_command(command, output="(ignored)")
        for p in expected_paths:
            assert p in result
        if not expected_paths:
            assert result.strip() == "<filepaths>\n</filepaths>"
