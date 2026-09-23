from pathlib import Path

from free_claude_code.cli.managed.session import ManagedClaudeSession


def test_cli_session_owns_typed_runner_config(tmp_path: Path) -> None:
    session = ManagedClaudeSession(
        workspace_path=str(tmp_path),
        proxy_root_url="http://127.0.0.1:8082",
        allowed_dirs=[str(tmp_path)],
        claude_bin="claude-test",
        auth_token="freecc",
    )

    assert session.config.workspace_path == str(tmp_path)
    assert session.config.proxy_root_url == "http://127.0.0.1:8082"
    assert session.config.allowed_dirs == [str(tmp_path)]
    assert session.config.claude_bin == "claude-test"
    assert session.config.auth_token == "freecc"
