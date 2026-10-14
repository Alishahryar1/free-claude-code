"""Smoke server subprocesses isolate runtime state and retain credential ownership."""

import json
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from free_claude_code.config import paths
from free_claude_code.config.env_migrations import (
    CONFIG_SCHEMA_VERSION,
    atomic_write_managed_config,
)
from free_claude_code.config.loader import ManagedConfigStore, get_settings
from free_claude_code.core.interprocess_lock import InterprocessFileLock
from smoke.lib.child_process import cmd_fcc_server
from smoke.lib.config import SmokeConfig
from smoke.lib.server import start_server


@pytest.fixture
def smoke_config(tmp_path, monkeypatch):
    home = tmp_path / "user-home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(paths, "config_dir_path", lambda: home / ".fcc")
    store = ManagedConfigStore()
    store.initialize({})
    values = dict(store.read({}).managed)
    values.update(
        MODEL="nvidia_nim/managed-model",
        NVIDIA_NIM_API_KEY='["managed-one", "managed-two"]',
        ANTHROPIC_AUTH_TOKEN="managed-proxy-token",
    )
    store.commit(values)
    monkeypatch.setenv("GROQ_API_KEY", "process-only-key")
    return SmokeConfig(
        root=Path(__file__).resolve().parents[2],
        results_dir=tmp_path / "results",
        live=True,
        interactive=False,
        targets=frozenset(),
        provider_matrix=frozenset(),
        timeout_s=10,
        prompt="",
        claude_bin="claude",
        worker_id="test",
        settings=get_settings(),
    )


def probe_command():
    # Execute the normal smoke command against a local owner/HTTP probe. This
    # exercises its actual child bootstrap without contacting model providers.
    probe = textwrap.dedent("""
        import json, os
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from free_claude_code.cli import entrypoints
        from free_claude_code.config import paths
        from free_claude_code.config.api_keys import api_key_values
        from free_claude_code.config.loader import get_settings
        from free_claude_code.core.interprocess_lock import InterprocessFileLock

        def serve():
            settings = get_settings()
            print(json.dumps({
                'config': str(paths.config_dir_path()),
                'database': str(paths.fcc_database_path()),
                'owner': str(paths.server_owner_lock_path()),
                'auth': str(paths.openai_auth_path()),
                'auth_lock': str(paths.openai_auth_lock_path()),
                'copilot': str(paths.github_copilot_auth_path()),
                'model': settings.model,
                'keys': api_key_values(settings.nvidia_nim_api_keys),
                'token': settings.proxy_auth_token,
                'removed': os.environ.get('GROQ_API_KEY'),
                'groq': api_key_values(settings.groq_api_keys),
                'home': os.environ.get('USERPROFILE'),
            }), flush=True)
            class Health(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                def log_message(self, *args):
                    pass
            with InterprocessFileLock(paths.server_owner_lock_path()):
                with HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Health) as server:
                    server.serve_forever()
        entrypoints.serve = serve
    """)
    return [sys.executable, "-c", probe + "\n" + cmd_fcc_server()[-1]]


def snapshot(server):
    return json.loads(server.log_path.read_text().splitlines()[0])


def test_smoke_servers_isolate_runtime_and_share_original_auth(smoke_config):
    source = paths.config_dir_path()
    original = paths.managed_env_path().read_bytes()
    overrides = {"MODEL": "nvidia_nim/override"}
    unset = {"NVIDIA_NIM_API_KEY", "GROQ_API_KEY", "ANTHROPIC_AUTH_TOKEN"}
    with InterprocessFileLock(paths.server_owner_lock_path()):
        with start_server(
            smoke_config,
            command=probe_command(),
            env_overrides=overrides,
            env_unset=unset,
        ) as first:
            one = snapshot(first)
            with start_server(
                smoke_config,
                command=probe_command(),
                env_unset=unset,
                env_overrides={**overrides, "NVIDIA_NIM_API_KEY": "explicit-key"},
            ) as second:
                two = snapshot(second)
                assert one["config"] != two["config"] != str(source)
                assert one["database"] != two["database"]
                assert (
                    one["owner"] != two["owner"] != str(paths.server_owner_lock_path())
                )
                assert one["keys"] == ["managed-one", "managed-two"]
                assert two["keys"] == ["explicit-key"]
                for state in (one, two):
                    assert state["model"] == "nvidia_nim/override"
                    assert state["token"] == "managed-proxy-token"
                    assert state["removed"] is None
                    assert state["groq"] == []
                    assert state["auth"] == str(paths.openai_auth_path())
                    assert state["auth_lock"] == str(paths.openai_auth_lock_path())
                    assert state["copilot"] == str(paths.github_copilot_auth_path())
                    assert state["home"] == str(source.parent)
        assert not Path(one["config"]).exists()
        assert not Path(two["config"]).exists()
    assert paths.managed_env_path().read_bytes() == original
    assert first.log_path.is_file() and second.log_path.is_file()


def test_explicit_smoke_home_remains_authoritative(smoke_config, tmp_path):
    home = tmp_path / "caller-home"
    atomic_write_managed_config(
        {"FCC_CONFIG_SCHEMA": CONFIG_SCHEMA_VERSION, "ANTHROPIC_AUTH_TOKEN": "caller"},
        path=home / ".fcc" / ".env",
    )
    with start_server(
        smoke_config,
        command=probe_command(),
        env_overrides={"HOME": str(home), "USERPROFILE": str(home)},
        env_unset={"NVIDIA_NIM_API_KEY", "GROQ_API_KEY", "ANTHROPIC_AUTH_TOKEN"},
    ) as server:
        state = snapshot(server)
    assert state["config"] == str(home / ".fcc")
    assert state["keys"] == []
    assert state["token"] != "managed-proxy-token"
    assert state["auth"] == str(home / ".fcc" / "auth" / "openai.json")
    assert (home / ".fcc").is_dir()


@pytest.mark.parametrize("failure", ["health", "spawn"])
def test_failed_smoke_start_cleans_private_state(smoke_config, monkeypatch, failure):
    from smoke.lib import server as smoke_server

    captured = {}
    command = ["custom-command", "--custom-arg"]

    def popen(args, **kwargs):
        assert args == command
        captured.update(kwargs["env"])
        if failure == "spawn":
            raise OSError("spawn failed")
        return object()

    def unhealthy(_server, **_kwargs):
        raise OSError("health failed")

    monkeypatch.setattr(smoke_server.subprocess, "Popen", popen)
    monkeypatch.setattr(smoke_server, "_wait_for_health", unhealthy)
    monkeypatch.setattr(smoke_server, "_stop_process", lambda _process: None)
    with (
        pytest.raises(OSError, match=f"{failure} failed"),
        start_server(replace(smoke_config, timeout_s=0.1), command=command),
    ):
        pytest.fail("startup should fail")
    assert not Path(captured["FCC_SMOKE_CONFIG_DIR"]).exists()
