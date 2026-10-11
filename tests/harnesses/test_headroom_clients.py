import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from free_claude_code.harnesses.headroom_clients import NativeClients
from free_claude_code.harnesses.headroom_mcp import (
    McpSetupError,
    configure_registrations,
)


@pytest.fixture
def clients(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("shutil.which", lambda name, **kwargs: str(tmp_path / name))
    owner = NativeClients(tmp_path / "headroom", {}, tmp_path)
    calls = []

    def run(args, **kwargs):
        calls.append(tuple(args))
        if "--version" in args:
            return "1.4.4"
        return "--scope --transport --global --yes --json --command COMMAND"

    monkeypatch.setattr(owner, "run", run)
    return owner, calls


@pytest.mark.parametrize(
    ("agent", "tail", "relative_path"),
    [
        (
            "claude",
            ("mcp", "add", "--scope", "user", "--transport", "stdio", "headroom", "--"),
            ".claude.json",
        ),
        ("codex", ("mcp", "add", "headroom", "--"), ".codex/config.toml"),
        ("pi", ("mcp", "add", "headroom", "--"), ".pi/agent/mcp.json"),
        (
            "opencode",
            ("mcp", "add", "headroom", "--global", "--"),
            ".config/opencode/opencode.json",
        ),
        (
            "cline",
            ("mcp", "install", "headroom", "--yes", "--json", "--"),
            ".cline/data/settings/cline_mcp_settings.json",
        ),
        (
            "grok",
            ("mcp", "add", "--scope", "user", "headroom", "--"),
            ".grok/config.toml",
        ),
    ],
)
def test_uses_native_user_registration_contract(
    clients, tmp_path, agent, tail, relative_path
):
    owner, calls = clients
    target = owner.prepare(agent)
    assert target is not None
    assert target.path == tmp_path / relative_path
    target.write()
    assert calls[-1] == (
        str(tmp_path / agent),
        *tail,
        str(tmp_path / "headroom"),
        "mcp",
        "serve",
    )


def test_cline_honors_mcp_override_independently_of_provider_file(clients, tmp_path):
    owner, _ = clients
    owner.env.update(
        CLINE_MCP_SETTINGS_PATH=str(tmp_path / "custom-mcp.json"),
        CLINE_PROVIDER_SETTINGS_PATH=str(tmp_path / "provider.json"),
    )
    target = owner.prepare("cline")
    assert target is not None
    assert target.path == tmp_path / "custom-mcp.json"
    assert target.expected == {
        "transport": {
            "type": "stdio",
            "command": str(tmp_path / "headroom"),
            "args": ["mcp", "serve"],
        }
    }


def test_old_pi_is_skipped_without_upgrade(clients, monkeypatch):
    owner, _ = clients
    monkeypatch.setattr(owner, "run", lambda args, **kwargs: "0.85.1")
    assert owner.prepare("pi") is None
    assert "0.99.0" in owner.outcomes[-1]


def test_opencode_keeps_existing_jsonc_destination(clients, tmp_path):
    owner, _ = clients
    root = tmp_path / ".config/opencode"
    root.mkdir(parents=True)
    (root / "opencode.jsonc").write_text('{/* native comment */"mcp":{"servers":{}}}')
    target = owner.prepare("opencode")
    assert target is not None and target.path == root / "opencode.jsonc"
    assert target.expected["type"] == "local"


def test_muse_preserves_legacy_schema(clients, tmp_path):
    owner, _ = clients
    path = tmp_path / ".config/muse/settings.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"mcp_servers":{"other":{"command":"other"}},"theme":"dark"}')
    target = owner.prepare("muse")
    assert target is not None
    configure_registrations([target])
    result = json.loads(path.read_text())
    assert "mcpServers" not in result
    assert result["mcp_servers"]["headroom"]["transport"] == "stdio"
    assert result["theme"] == "dark"


def test_muse_conflicting_aliases_are_not_rewritten(clients, tmp_path):
    owner, _ = clients
    path = tmp_path / ".config/muse/settings.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"mcp_servers":{},"mcpServers":{}}')
    with pytest.raises(McpSetupError, match=r"Muse.*aliases"):
        owner.prepare("muse")


def test_aider_reports_no_native_mcp(clients):
    owner, _ = clients
    assert owner.prepare("aider") is None
    assert "native MCP" in owner.outcomes[-1]


def test_grok_native_policy_block_is_preserved(clients, monkeypatch, tmp_path):
    owner, calls = clients
    original_run = owner.run

    def run(args, **kwargs):
        if args[1:] == ["mcp", "list", "--json"]:
            return json.dumps(
                [
                    {
                        "name": "headroom",
                        "command": str(owner.headroom),
                        "args": ["mcp", "serve"],
                        "enabled": True,
                        "blocked_reason": "managed allowlist",
                        "scope": "user",
                    }
                ]
            )
        return original_run(args, **kwargs)

    monkeypatch.setattr(owner, "run", run)
    target = owner.prepare("grok")
    assert target is not None
    results = configure_registrations([target])
    assert "skipped" in results[0]
    assert not any("headroom" in call for call in calls)


@pytest.mark.skipif(os.name != "nt", reason="Windows native command wrappers")
def test_timed_out_native_command_reaps_its_child(tmp_path):
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "native.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "Path(sys.argv[1]).write_text(str(child.pid))\n"
        "time.sleep(30)\n"
    )
    owner = NativeClients(tmp_path / "headroom", os.environ, tmp_path)
    try:
        with pytest.raises(McpSetupError, match="Could not run"):
            owner.run([sys.executable, str(script), str(pid_file)], timeout=0.5)
        pid = pid_file.read_text()
        listed = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], text=True
        )
        assert f'"{pid}"' not in listed
    finally:
        if pid_file.exists():
            subprocess.run(
                ["taskkill", "/PID", pid_file.read_text(), "/T", "/F"],
                capture_output=True,
            )
