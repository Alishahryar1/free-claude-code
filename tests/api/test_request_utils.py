"""Tests for api/request_utils.py module."""

from unittest.mock import MagicMock

import pytest

from api.command_utils import extract_command_prefix
from api.detection import (
    is_prefix_detection_request,
    is_quota_check_request,
    is_title_generation_request,
)
from api.models.anthropic import Message, MessagesRequest
from api.request_utils import get_token_count


class TestQuotaCheckRequest:
    """Tests for is_quota_check_request function."""

    def test_quota_check_simple_string(self):
        """Test quota check with simple string content."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "Check my quota"

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 1
        req.messages = [msg]

        assert is_quota_check_request(req) is True

    def test_quota_check_case_insensitive(self):
        """Test quota check is case insensitive."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "Check my QUOTA"

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 1
        req.messages = [msg]

        assert is_quota_check_request(req) is True

    def test_quota_check_list_content(self):
        """Test quota check with list content blocks."""
        block = MagicMock()
        block.text = "Check my quota"

        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = [block]

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 1
        req.messages = [msg]

        assert is_quota_check_request(req) is True

    def test_not_quota_check_wrong_max_tokens(self):
        """Test not quota check when max_tokens != 1."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "Check my quota"

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 100
        req.messages = [msg]

        assert is_quota_check_request(req) is False

    def test_not_quota_check_multiple_messages(self):
        """Test not quota check when multiple messages."""
        msg1 = MagicMock(spec=Message)
        msg1.role = "user"
        msg1.content = "Check my quota"

        msg2 = MagicMock(spec=Message)
        msg2.role = "assistant"
        msg2.content = "Hello"

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 1
        req.messages = [msg1, msg2]

        assert is_quota_check_request(req) is False

    def test_not_quota_check_wrong_role(self):
        """Test not quota check when role is not user."""
        msg = MagicMock(spec=Message)
        msg.role = "assistant"
        msg.content = "Check my quota"

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 1
        req.messages = [msg]

        assert is_quota_check_request(req) is False

    def test_not_quota_check_no_quota_keyword(self):
        """Test not quota check when content doesn't contain quota."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "Hello world"

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 1
        req.messages = [msg]

        assert is_quota_check_request(req) is False


class TestTitleGenerationRequest:
    """Tests for is_title_generation_request function."""

    def _title_gen_system(self) -> list[MagicMock]:
        block = MagicMock()
        block.text = "Analyze if this message indicates a new conversation topic. If it does, extract a 2-3 word title."
        return [block]

    def test_title_generation_detected_via_system(self):
        """Title gen detected by system prompt containing topic/title keywords."""
        req = MagicMock(spec=MessagesRequest)
        req.system = self._title_gen_system()
        req.tools = None

        assert is_title_generation_request(req) is True

    def test_title_generation_not_detected_with_tools(self):
        """Not detected when tools are present (main conversation, not title gen)."""
        req = MagicMock(spec=MessagesRequest)
        req.system = self._title_gen_system()
        req.tools = [MagicMock()]

        assert is_title_generation_request(req) is False

    def test_title_generation_not_detected_no_system(self):
        """Not detected when system is absent."""
        req = MagicMock(spec=MessagesRequest)
        req.system = None
        req.tools = None

        assert is_title_generation_request(req) is False

    def test_title_generation_not_detected_unrelated_system(self):
        """Not detected when system prompt has no topic/title keywords."""
        block = MagicMock()
        block.text = "You are a helpful assistant."
        req = MagicMock(spec=MessagesRequest)
        req.system = [block]
        req.tools = None

        assert is_title_generation_request(req) is False


class TestExtractCommandPrefix:
    """Tests for extract_command_prefix function."""

    def test_simple_command(self):
        """Test extraction of simple command."""
        assert extract_command_prefix("git status") == "git status"
        assert extract_command_prefix("ls -la") == "ls"

    def test_two_word_commands(self):
        """Test extraction of two-word commands."""
        assert extract_command_prefix("git commit -m 'message'") == "git commit"
        assert extract_command_prefix("npm install package") == "npm install"
        assert extract_command_prefix("docker run image") == "docker run"
        assert extract_command_prefix("kubectl get pods") == "kubectl get"

    def test_two_word_command_with_options(self):
        """Test two-word command with options only returns first word."""
        assert extract_command_prefix("git -v") == "git"
        assert extract_command_prefix("npm --version") == "npm"

    def test_with_env_vars(self):
        """Test command with environment variables."""
        assert extract_command_prefix("DEBUG=1 python script.py") == "DEBUG=1 python"
        assert (
            extract_command_prefix("API_KEY=secret node app.js")
            == "API_KEY=secret node"
        )

    def test_single_word_commands(self):
        """Test single word commands."""
        assert extract_command_prefix("ls") == "ls"
        assert extract_command_prefix("python") == "python"
        assert extract_command_prefix("make") == "make"

    def test_command_injection_detected(self):
        """Test detection of command injection attempts."""
        assert extract_command_prefix("`whoami`") == "command_injection_detected"
        assert extract_command_prefix("$(whoami)") == "command_injection_detected"
        assert (
            extract_command_prefix("echo $(cat /etc/passwd)")
            == "command_injection_detected"
        )

    def test_empty_command(self):
        """Test handling of empty commands."""
        assert extract_command_prefix("") == "none"
        assert extract_command_prefix("   ") == "none"

    def test_complex_git_command(self):
        """Test complex git command extraction."""
        assert extract_command_prefix("git log --oneline --graph") == "git log"
        assert (
            extract_command_prefix("git checkout -b feature-branch") == "git checkout"
        )

    def test_cargo_command(self):
        """Test cargo command extraction."""
        assert extract_command_prefix("cargo build") == "cargo build"
        assert extract_command_prefix("cargo test") == "cargo test"
        assert extract_command_prefix("cargo --version") == "cargo"


class TestPrefixDetectionRequest:
    """Tests for is_prefix_detection_request function."""

    def test_prefix_detection_with_policy_spec(self):
        """Test prefix detection with policy spec and command."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "<policy_spec>policy</policy_spec> Command: git status"

        req = MagicMock(spec=MessagesRequest)
        req.messages = [msg]

        is_prefix, command = is_prefix_detection_request(req)
        assert is_prefix is True
        assert command == "git status"

    def test_prefix_detection_case_sensitive(self):
        """Test prefix detection is case sensitive for Command:."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "<policy_spec>policy</policy_spec> command: git status"

        req = MagicMock(spec=MessagesRequest)
        req.messages = [msg]

        is_prefix, command = is_prefix_detection_request(req)
        assert is_prefix is False
        assert command == ""

    def test_not_prefix_detection_no_policy_spec(self):
        """Test not prefix detection without policy_spec."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "Command: git status"

        req = MagicMock(spec=MessagesRequest)
        req.messages = [msg]

        is_prefix, command = is_prefix_detection_request(req)
        assert is_prefix is False
        assert command == ""

    def test_not_prefix_detection_multiple_messages(self):
        """Test not prefix detection with multiple messages."""
        msg1 = MagicMock(spec=Message)
        msg1.role = "user"
        msg1.content = "<policy_spec>policy</policy_spec> Command: git status"

        msg2 = MagicMock(spec=Message)
        msg2.role = "assistant"
        msg2.content = "OK"

        req = MagicMock(spec=MessagesRequest)
        req.messages = [msg1, msg2]

        is_prefix, command = is_prefix_detection_request(req)
        assert is_prefix is False
        assert command == ""

    def test_not_prefix_detection_wrong_role(self):
        """Test not prefix detection when message is not from user."""
        msg = MagicMock(spec=Message)
        msg.role = "assistant"
        msg.content = "<policy_spec>policy</policy_spec> Command: git status"

        req = MagicMock(spec=MessagesRequest)
        req.messages = [msg]

        is_prefix, command = is_prefix_detection_request(req)
        assert is_prefix is False
        assert command == ""

    def test_prefix_detection_list_content(self):
        """Test prefix detection with list content blocks."""
        block = MagicMock()
        block.text = "<policy_spec>policy</policy_spec> Command: ls -la"

        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = [block]

        req = MagicMock(spec=MessagesRequest)
        req.messages = [msg]

        is_prefix, command = is_prefix_detection_request(req)
        assert is_prefix is True
        assert command == "ls -la"


class TestGetTokenCount:
    """Tests for get_token_count function."""

    def test_empty_messages(self):
        """Test token count with empty messages."""
        count = get_token_count([])
        assert count >= 1  # Returns max(1, tokens)

    def test_simple_message(self):
        """Test token count with simple text message."""
        msg = MagicMock()
        msg.content = "Hello world"

        count = get_token_count([msg])
        assert count > 0
        # "Hello world" is ~2-3 tokens plus overhead
        assert count >= 3

    def test_message_with_system_prompt(self):
        """Test token count includes system prompt."""
        msg = MagicMock()
        msg.content = "Hello"

        count = get_token_count([msg], system="You are a helpful assistant")
        assert count > 0

    def test_message_with_list_content(self):
        """Test token count with list content blocks."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello world"

        msg = MagicMock()
        msg.content = [text_block]

        count = get_token_count([msg])
        assert count > 0

    def test_message_with_thinking_block(self):
        """Test token count includes thinking blocks."""
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Let me think about this..."

        msg = MagicMock()
        msg.content = [thinking_block]

        count = get_token_count([msg])
        assert count > 0

    def test_message_with_tool_use(self):
        """Test token count includes tool use blocks."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search"
        tool_block.input = {"query": "test"}

        msg = MagicMock()
        msg.content = [tool_block]

        count = get_token_count([msg])
        assert count > 0

    def test_message_with_tool_result(self):
        """Test token count includes tool result blocks."""
        result_block = MagicMock()
        result_block.type = "tool_result"
        result_block.content = "Search results here"

        msg = MagicMock()
        msg.content = [result_block]

        count = get_token_count([msg])
        assert count > 0

    def test_message_with_tools(self):
        """Test token count includes tool definitions."""
        msg = MagicMock()
        msg.content = "Use the search tool"

        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search for information"
        tool.input_schema = {"type": "object", "properties": {}}

        count = get_token_count([msg], tools=[tool])
        assert count > 0

    def test_system_as_list(self):
        """Test token count with system as list of blocks."""
        msg = MagicMock()
        msg.content = "Hello"

        block = MagicMock()
        block.text = "System prompt"

        count = get_token_count([msg], system=[block])
        assert count > 0

    def test_tool_result_with_dict_content(self):
        """Test token count with tool result containing dict content."""
        result_block = MagicMock()
        result_block.type = "tool_result"
        result_block.content = {"result": "data"}

        msg = MagicMock()
        msg.content = [result_block]

        count = get_token_count([msg])
        assert count > 0

    def test_multiple_messages_overhead(self):
        """Test that multiple messages include overhead."""
        msg1 = MagicMock()
        msg1.content = "Hi"

        msg2 = MagicMock()
        msg2.content = "Hello"

        count_single = get_token_count([msg1])
        count_double = get_token_count([msg1, msg2])

        # Double message should have more tokens (including overhead)
        assert count_double > count_single

    def test_per_message_overhead_four_tokens(self):
        """Per-message overhead is 4 tokens (was 3)."""
        msg = MagicMock()
        msg.content = "x"  # Minimal content
        count = get_token_count([msg])
        # 1 msg * 4 overhead + content tokens
        assert count >= 5

    def test_system_overhead_added(self):
        """System prompt adds ~4 tokens overhead."""
        msg = MagicMock()
        msg.content = "Hi"
        count_no_sys = get_token_count([msg])
        count_with_sys = get_token_count([msg], system="You are helpful")
        assert count_with_sys >= count_no_sys + 4

    def test_system_as_list_of_dicts(self):
        """System blocks as dicts (not objects) are counted."""
        msg = MagicMock()
        msg.content = "Hi"
        count_no_sys = get_token_count([msg])
        system_dicts = [{"type": "text", "text": "System prompt from dict"}]
        count_with_dict_sys = get_token_count([msg], system=system_dicts)
        assert count_with_dict_sys > count_no_sys

    def test_tool_use_includes_id(self):
        """Tool use blocks count id field."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search"
        tool_block.input = {"q": "test"}
        tool_block.id = "call_abc123"
        msg = MagicMock()
        msg.content = [tool_block]
        count = get_token_count([msg])
        assert count > 0

    def test_tool_result_includes_tool_use_id(self):
        """Tool result blocks count tool_use_id field."""
        result_block = MagicMock()
        result_block.type = "tool_result"
        result_block.content = "ok"
        result_block.tool_use_id = "call_xyz"
        msg = MagicMock()
        msg.content = [result_block]
        count = get_token_count([msg])
        assert count > 0

    def test_unrecognized_block_type_fallback(self):
        """Unrecognized block types are tokenized via json.dumps fallback."""
        unknown_block = {"type": "custom", "spec": "data"}
        msg = MagicMock()
        msg.content = [unknown_block]
        count = get_token_count([msg])
        assert count > 0

    def test_message_with_image_block(self):
        """Test token count includes image blocks."""
        image_block = MagicMock()
        image_block.type = "image"
        image_block.source = {
            "type": "base64",
            "media_type": "image/png",
            "data": "x" * 3000,
        }
        msg = MagicMock()
        msg.content = [image_block]
        count = get_token_count([msg])
        assert count >= 85

    def test_image_block_with_dict_source(self):
        """Image block with dict-style source is counted."""
        image_block = {"type": "image", "source": {"data": "a" * 10000}}
        msg = MagicMock()
        msg.content = [image_block]
        count = get_token_count([msg])
        assert count >= 85

    def test_known_payload_estimate_range(self):
        """Known payload produces estimate within expected range (validation harness)."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        system_text = "You are a helpful assistant."
        user_text = "Hello, how are you?"
        sys_tokens = len(enc.encode(system_text))
        user_tokens = len(enc.encode(user_text))
        # Min: content tokens + system overhead (4) + per-msg overhead (4)
        expected_min = sys_tokens + user_tokens + 4 + 4
        msg = MagicMock()
        msg.content = user_text
        count = get_token_count([msg], system=system_text)
        assert count >= expected_min, f"count={count} < expected_min={expected_min}"


# --- Parametrized Edge Case Tests ---


@pytest.mark.parametrize(
    "command,expected",
    [
        ("git status", "git status"),
        ("ls -la", "ls"),
        ("git commit -m 'msg'", "git commit"),
        ("npm install pkg", "npm install"),
        ("ls", "ls"),
        ("python", "python"),
        ("", "none"),
        ("   ", "none"),
        ("`whoami`", "command_injection_detected"),
        ("$(whoami)", "command_injection_detected"),
        ("echo $(cat /etc/passwd)", "command_injection_detected"),
        ("git -v", "git"),
        ("DEBUG=1 python script.py", "DEBUG=1 python"),
        ("cargo build", "cargo build"),
        ("cargo --version", "cargo"),
    ],
    ids=[
        "git_status",
        "ls_with_flag",
        "git_commit",
        "npm_install",
        "bare_ls",
        "bare_python",
        "empty",
        "whitespace",
        "injection_backtick",
        "injection_dollar",
        "injection_echo",
        "git_flag",
        "env_var",
        "cargo_build",
        "cargo_flag",
    ],
)
def test_extract_command_prefix_parametrized(command, expected):
    """Parametrized command prefix extraction."""
    assert extract_command_prefix(command) == expected


def test_extract_command_prefix_unterminated_quote():
    """Unterminated quote falls back to simple split (shlex.split ValueError)."""
    result = extract_command_prefix("git commit -m 'unterminated")
    # Should fall back to command.split()[0] = "git"
    assert result == "git"


def test_extract_command_prefix_pipe():
    """Piped commands - shlex handles pipe character."""
    result = extract_command_prefix("cat file.txt | grep pattern")
    assert result in ("cat", "cat file.txt")


@pytest.mark.parametrize(
    "content,max_tokens,role,expected",
    [
        ("Check my quota", 1, "user", True),
        ("Check my QUOTA", 1, "user", True),
        ("Hello world", 1, "user", False),
        ("Check my quota", 100, "user", False),
        ("Check my quota", 1, "assistant", False),
    ],
    ids=["basic", "case_insensitive", "no_keyword", "wrong_max_tokens", "wrong_role"],
)
def test_quota_check_parametrized(content, max_tokens, role, expected):
    """Parametrized quota check request detection."""
    msg = MagicMock(spec=Message)
    msg.role = role
    msg.content = content

    req = MagicMock(spec=MessagesRequest)
    req.max_tokens = max_tokens
    req.messages = [msg]

    assert is_quota_check_request(req) is expected


def test_quota_check_empty_messages():
    """Quota check with empty message list should not crash."""
    req = MagicMock(spec=MessagesRequest)
    req.max_tokens = 1
    req.messages = []
    assert is_quota_check_request(req) is False
