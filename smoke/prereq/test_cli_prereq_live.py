import os
import shutil
from pathlib import Path

import pytest

from free_claude_code.harnesses.claude import build_claude_proxy_env
from smoke.lib.child_process import (
    cmd_fcc_server,
    run_captured_text,
)
from smoke.lib.config import SmokeConfig
from smoke.lib.server import start_server
from smoke.lib.skips import skip_upstream_unavailable

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("cli")]


def test_fcc_server_entrypoint_starts_server(smoke_config: SmokeConfig) -> None:
    with start_server(
        smoke_config,
        command=cmd_fcc_server(),
        env_overrides={"MESSAGING_PLATFORM": "none"},
        name="entrypoint",
    ) as server:
        assert server.process.poll() is None


def test_claude_cli_prompt_when_available(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    claude_bin = shutil.which(smoke_config.claude_bin)
    if not claude_bin:
        pytest.skip(f"Claude CLI not found: {smoke_config.claude_bin}")
    models = smoke_config.provider_models()
    if not models:
        pytest.skip("no configured provider model available for Claude CLI smoke")

    with start_server(
        smoke_config,
        env_overrides={"MODEL": models[0].full_model, "MESSAGING_PLATFORM": "none"},
        name="claude-cli",
    ) as server:
        env = build_claude_proxy_env(
            proxy_root_url=server.base_url,
            auth_token=smoke_config.settings.proxy_auth_token,
            base_env=os.environ,
        )
        result = run_captured_text(
            [claude_bin, "-p", "Reply with exactly FCC_SMOKE_PONG"],
            cwd=tmp_path,
            env=env,
            timeout=smoke_config.timeout_s,
            check=False,
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "GET /v1/models" in server_log, (
        "Claude CLI did not discover models from the local gateway"
    )
    assert "POST /v1/messages" in server_log, (
        "Claude CLI did not call the local Anthropic-compatible endpoint"
    )
    if "FCC_SMOKE_PONG" not in result.stdout:
        skip_upstream_unavailable(
            "Claude CLI reached the local proxy but returned no smoke token"
        )
