import os

from free_claude_code.cli.managed.claude import (
    MANAGED_CLAUDE_MODEL_TIER,
    ManagedClaudeConfig,
    ManagedClaudeParseState,
    ManagedClaudeTaskRequest,
    build_managed_claude_env,
    build_managed_claude_invocation,
    extract_managed_claude_session_id,
    parse_managed_claude_stdout_line,
)
from free_claude_code.cli.managed.diagnostics import classify_managed_claude_stderr
from free_claude_code.harnesses.claude import build_claude_proxy_env


def _config(**overrides: object) -> ManagedClaudeConfig:
    workspace_path = overrides.get("workspace_path", os.path.normpath("/tmp/workspace"))
    proxy_root_url = overrides.get("proxy_root_url", "http://localhost:8082")
    raw_allowed_dirs = overrides.get("allowed_dirs")
    allowed_dirs: list[str] = []
    if raw_allowed_dirs is not None:
        assert isinstance(raw_allowed_dirs, list)
        for directory in raw_allowed_dirs:
            assert isinstance(directory, str)
            allowed_dirs.append(directory)
    claude_bin = overrides.get("claude_bin", "claude")
    auth_token = overrides.get("auth_token", "proxy-token")

    assert isinstance(workspace_path, str)
    assert isinstance(proxy_root_url, str)
    assert isinstance(claude_bin, str)
    assert isinstance(auth_token, str)
    return ManagedClaudeConfig(
        workspace_path=workspace_path,
        proxy_root_url=proxy_root_url,
        allowed_dirs=allowed_dirs,
        claude_bin=claude_bin,
        auth_token=auth_token,
    )


def test_managed_claude_defaults_to_fable() -> None:
    assert MANAGED_CLAUDE_MODEL_TIER == "fable"


def test_managed_claude_builds_new_task_command_and_env() -> None:
    invocation = build_managed_claude_invocation(
        config=_config(allowed_dirs=[os.path.normpath("/tmp/extra")]),
        request=ManagedClaudeTaskRequest(prompt="hello"),
        base_env={"PATH": "keep", "ANTHROPIC_API_KEY": "official"},
    )

    assert invocation.argv[:4] == (
        "claude",
        "--model",
        MANAGED_CLAUDE_MODEL_TIER,
        "-p",
    )
    assert "hello" in invocation.argv
    assert "--output-format" in invocation.argv
    assert "stream-json" in invocation.argv
    assert "--add-dir" in invocation.argv
    assert os.path.normpath("/tmp/extra") in invocation.argv
    assert "--settings" not in invocation.argv
    assert invocation.env["PATH"] == "keep"
    assert invocation.env["ANTHROPIC_BASE_URL"] == "http://localhost:8082"
    assert invocation.env["ANTHROPIC_AUTH_TOKEN"] == "proxy-token"
    assert invocation.env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert invocation.env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "190000"
    assert invocation.env["DISABLE_AUTOUPDATER"] == "1"
    assert invocation.env["DISABLE_FEEDBACK_COMMAND"] == "1"
    assert invocation.env["DISABLE_ERROR_REPORTING"] == "1"
    assert invocation.env["DISABLE_TELEMETRY"] == "1"
    assert invocation.env["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert invocation.env["no_proxy"] == invocation.env["NO_PROXY"]
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" not in invocation.env
    assert "ANTHROPIC_API_URL" not in invocation.env
    assert "ANTHROPIC_API_KEY" not in invocation.env
    assert invocation.trace_metadata["client_cli_id"] == "claude"
    assert invocation.trace_metadata["claude_binary"] == "claude"
    assert invocation.trace_metadata["managed_model_tier"] == MANAGED_CLAUDE_MODEL_TIER


def test_managed_claude_builds_resume_and_fork_commands() -> None:
    resume = build_managed_claude_invocation(
        config=_config(),
        request=ManagedClaudeTaskRequest(prompt="again", session_id="sess_1"),
        base_env={},
    )
    fork = build_managed_claude_invocation(
        config=_config(),
        request=ManagedClaudeTaskRequest(
            prompt="branch", session_id="sess_1", fork_session=True
        ),
        base_env={},
    )

    assert resume.argv[:6] == (
        "claude",
        "--resume",
        "sess_1",
        "--model",
        MANAGED_CLAUDE_MODEL_TIER,
        "-p",
    )
    assert "--fork-session" not in resume.argv
    assert fork.argv[:7] == (
        "claude",
        "--resume",
        "sess_1",
        "--fork-session",
        "--model",
        MANAGED_CLAUDE_MODEL_TIER,
        "-p",
    )
    assert "--fork-session" in fork.argv


def test_managed_claude_uses_native_plan_storage() -> None:
    invocation = build_managed_claude_invocation(
        config=_config(),
        request=ManagedClaudeTaskRequest(prompt="hello"),
        base_env={},
    )

    assert "--settings" not in invocation.argv


def test_managed_claude_env_forwards_retained_proxy_auth_token() -> None:
    env = build_managed_claude_env(
        proxy_root_url="http://localhost:8082",
        auth_token="freecc",
        base_env={"ANTHROPIC_AUTH_TOKEN": "stale"},
    )

    assert env["ANTHROPIC_AUTH_TOKEN"] == "freecc"


def test_managed_claude_env_adds_noninteractive_process_policy() -> None:
    base_env = {
        "PATH": "keep",
        "ANTHROPIC_API_URL": "https://api.anthropic.com/v1",
        "ANTHROPIC_API_KEY": "official-key",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "0",
    }
    proxy_env = build_claude_proxy_env(
        proxy_root_url="http://localhost:8082",
        auth_token="proxy-token",
        base_env=base_env,
    )

    managed_env = build_managed_claude_env(
        proxy_root_url="http://localhost:8082",
        auth_token="proxy-token",
        base_env=base_env,
    )

    assert managed_env == {
        **proxy_env,
        "DISABLE_TELEMETRY": "1",
        "TERM": "dumb",
        "PYTHONIOENCODING": "utf-8",
    }


def test_managed_claude_stderr_classifier_filters_known_benign_notice() -> None:
    diagnostics = classify_managed_claude_stderr(
        "claude.ai connectors are disabled in this environment"
    )

    assert diagnostics.has_benign
    assert diagnostics.benign_lines == (
        "claude.ai connectors are disabled in this environment",
    )
    assert diagnostics.fatal_text is None


def test_managed_claude_stderr_classifier_preserves_unknown_lines() -> None:
    diagnostics = classify_managed_claude_stderr(
        "claude.ai connectors are disabled in this environment\nFatal error"
    )

    assert diagnostics.has_benign
    assert diagnostics.fatal_text == "Fatal error"


def test_managed_claude_extracts_session_ids() -> None:
    assert extract_managed_claude_session_id({"session_id": "direct"}) == "direct"
    assert extract_managed_claude_session_id({"sessionId": "camel"}) == "camel"
    assert (
        extract_managed_claude_session_id({"init": {"session_id": "nested"}})
        == "nested"
    )
    assert (
        extract_managed_claude_session_id({"result": {"sessionId": "result"}})
        == "result"
    )
    assert extract_managed_claude_session_id({"conversation": {"id": "conv"}}) == "conv"
    assert extract_managed_claude_session_id({"type": "message"}) is None
    assert extract_managed_claude_session_id("not a dict") is None


def test_managed_claude_parser_emits_session_info_once() -> None:
    state = ManagedClaudeParseState()

    first = list(parse_managed_claude_stdout_line('{"session_id": "sess_1"}', state))
    second = list(parse_managed_claude_stdout_line('{"session_id": "sess_2"}', state))

    assert first == [
        {"type": "session_info", "session_id": "sess_1"},
        {"session_id": "sess_1"},
    ]
    assert second == [{"session_id": "sess_2"}]


def test_managed_claude_parser_returns_raw_for_non_json() -> None:
    events = list(
        parse_managed_claude_stdout_line(
            "not json", ManagedClaudeParseState(log_raw_cli_diagnostics=False)
        )
    )

    assert events == [{"type": "raw", "content": "not json"}]


def test_managed_claude_parser_returns_raw_for_non_object_json() -> None:
    events = list(
        parse_managed_claude_stdout_line(
            '"not an event"', ManagedClaudeParseState(log_raw_cli_diagnostics=False)
        )
    )

    assert events == [{"type": "raw", "content": '"not an event"'}]
