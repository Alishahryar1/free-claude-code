import pytest

from free_claude_code.cli import configure_headroom as cli
from free_claude_code.harnesses.headroom_mcp import McpSetupError, json_registration


@pytest.fixture
def setup(tmp_path, monkeypatch):
    binary = tmp_path / "headroom"
    binary.touch()
    events = []
    monkeypatch.setattr(cli.NativeClients, "run", lambda self, args: "headroom 0.40.0")

    async def verify(command):
        events.append("verify")

    def prepare(self, agent):
        assert self.cwd != tmp_path and self.cwd.is_dir()
        events.append(agent)
        return json_registration(
            agent, tmp_path / f"{agent}.json", ("mcpServers",), binary
        )

    monkeypatch.setattr(cli, "verify_mcp", verify)
    monkeypatch.setattr(cli.NativeClients, "prepare", prepare)
    return binary, events


def test_check_then_apply_verifies_first_and_deduplicates_agents(
    setup, tmp_path, capsys
):
    binary, events = setup
    args = ["--headroom", str(binary), "--agent", "claude", "--agent", "claude"]
    cli.main([*args, "--check"])
    assert events == ["verify", "claude"]
    assert not (tmp_path / "claude.json").exists()
    cli.main(args)
    assert (tmp_path / "claude.json").exists()
    assert "claude: configured" in capsys.readouterr().out


def test_failed_mcp_check_prevents_registration(setup, monkeypatch, tmp_path):
    binary, events = setup

    async def broken(command):
        raise McpSetupError("required MCP tools are missing")

    monkeypatch.setattr(cli, "verify_mcp", broken)
    with pytest.raises(SystemExit, match="1"):
        cli.main(["--headroom", str(binary), "--agent", "codex"])
    assert not events
    assert not (tmp_path / "codex.json").exists()


def test_invalid_config_is_reported_without_quoting_credentials(
    setup, tmp_path, capsys
):
    binary, _ = setup
    path = tmp_path / "codex.json"
    path.write_text('{"secret":do-not-print-this}')
    with pytest.raises(SystemExit, match="1"):
        cli.main(["--headroom", str(binary), "--agent", "claude", "--agent", "codex"])
    assert not (tmp_path / "claude.json").exists()
    assert "do-not-print-this" not in capsys.readouterr().err


def test_desktop_and_cli_sharing_native_scope_write_once(setup, monkeypatch, tmp_path):
    binary, _ = setup
    monkeypatch.setattr(cli, "managed_env_path", lambda: tmp_path / "missing.env")
    monkeypatch.setattr(
        cli, "prepare_desktops", lambda clients, settings: [clients.prepare("codex")]
    )
    cli.main(
        ["--headroom", str(binary), "--agent", "codex", "--include-connected-desktops"]
    )
    before = (tmp_path / "codex.json").read_bytes()
    cli.main(
        ["--headroom", str(binary), "--agent", "codex", "--include-connected-desktops"]
    )
    assert (tmp_path / "codex.json").read_bytes() == before


def test_rejects_relative_executable_before_launch(monkeypatch):
    monkeypatch.setattr(cli.NativeClients, "run", lambda *args: pytest.fail("launched"))
    with pytest.raises(SystemExit, match="1"):
        cli.main(["--headroom", "headroom"])
