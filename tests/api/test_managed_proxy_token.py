"""Independent Codex credential commands agree with the server's managed token."""

import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.loader import ManagedConfigStore
from tests.api.support import create_test_app


@pytest.mark.parametrize("managed_token", [None, "admin-token"])
@pytest.mark.parametrize("client_token", [None, "different-terminal-token"])
def test_independent_codex_token_authenticates_with_server(
    monkeypatch, managed_token, client_token
):
    store = ManagedConfigStore()
    store.initialize({})
    values = dict(store.read({}).managed)
    values["PROXY_AUTH_ENABLED"] = "true"
    if managed_token is not None:
        values["ANTHROPIC_AUTH_TOKEN"] = managed_token
    store.commit(values)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "server-terminal-token")
    settings = store.read().settings
    child_env = dict(os.environ)
    child_env.pop("ANTHROPIC_AUTH_TOKEN", None)
    if client_token is not None:
        child_env["ANTHROPIC_AUTH_TOKEN"] = client_token
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from free_claude_code.config import paths; "
            "paths.config_dir_path = lambda: Path(sys.argv[1]); "
            "from free_claude_code.cli.launchers.codex import launch; "
            "launch(['--print-proxy-auth-token'])",
            str(store.path.parent),
        ],
        env=child_env,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    with TestClient(create_test_app(settings)) as client:
        assert client.get("/v1/models").status_code == 401
        response = client.get(
            "/v1/models", headers={"Authorization": f"Bearer {result.stdout.strip()}"}
        )
        assert response.status_code == 200
    assert result.stdout == (managed_token or "freecc") + "\n"
    assert settings.proxy_auth_token == (managed_token or "freecc")
