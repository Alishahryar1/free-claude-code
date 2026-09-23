import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from free_claude_code.config.paths import managed_env_path
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.core.json_types import JsonObject, JsonValue
from smoke.lib.child_process import cmd_python_c
from smoke.lib.claude_cli_matrix import run_claude_cli
from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import (
    ClientProtocolDriver,
    ConversationDriver,
    ProviderMatrixDriver,
    SmokeServerDriver,
    assert_product_stream,
)
from smoke.lib.server import find_free_port

pytestmark = [pytest.mark.live]


def _json_object_lines(text: str) -> list[JsonObject]:
    objects: list[JsonObject] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            value: JsonValue = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


@pytest.mark.smoke_target("clients")
def test_vscode_protocol_e2e(smoke_config: SmokeConfig) -> None:
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    with SmokeServerDriver(
        smoke_config,
        name="product-vscode",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        turn = ConversationDriver(server, smoke_config).stream(
            ClientProtocolDriver.adaptive_thinking_payload(),
            headers=ClientProtocolDriver.vscode_headers(),
        )

    assert_product_stream(turn.events)


@pytest.mark.smoke_target("clients")
def test_jetbrains_protocol_e2e(smoke_config: SmokeConfig) -> None:
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    with SmokeServerDriver(
        smoke_config,
        name="product-jetbrains",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        driver = ConversationDriver(server, smoke_config)
        first = driver.stream(
            ClientProtocolDriver.tool_result_payload(),
            headers=ClientProtocolDriver.jetbrains_headers(),
        )

    assert_product_stream(first.events)


@pytest.mark.smoke_target("clients")
def test_pi_cli_prompt_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    if not shutil.which("pi"):
        pytest.skip("missing_env: Pi CLI not found")
    uv_bin = shutil.which("uv")
    if not uv_bin:
        pytest.skip("missing_env: uv not found")
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    auth_token = smoke_config.settings.proxy_auth_token

    with SmokeServerDriver(
        smoke_config,
        name="product-pi-cli",
        env_overrides={
            "MODEL": provider_model.full_model,
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        env = os.environ.copy()
        env.update(
            {
                "HOST": "127.0.0.1",
                "PORT": str(server.port),
                "FCC_OPEN_BROWSER": "0",
                "ANTHROPIC_AUTH_TOKEN": auth_token,
                "PI_CODING_AGENT_DIR": str(tmp_path / "pi-agent"),
            }
        )
        result = subprocess.run(
            [
                uv_bin,
                "run",
                "--project",
                str(smoke_config.root),
                "--no-sync",
                "fcc-pi",
                "--no-session",
                "--no-approve",
                "--print",
                "Reply with exactly FCC_SMOKE_PI",
            ],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=smoke_config.timeout_s + 15,
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    assert result.returncode == 0, result.stderr or result.stdout
    assert "FCC_SMOKE_PI" in result.stdout
    assert "POST /v1/messages" in server_log


@pytest.mark.smoke_target("clients")
def test_opencode_cli_prompt_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    if not shutil.which("opencode"):
        pytest.skip("missing_env: OpenCode CLI not found")
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    auth_token = smoke_config.settings.proxy_auth_token
    isolated_home = tmp_path / "opencode-home"
    isolated_config = tmp_path / "opencode-config"
    for path in (isolated_home, isolated_config):
        path.mkdir()
    isolated_fcc = isolated_home / ".fcc"
    isolated_fcc.mkdir(mode=0o700)
    shutil.copyfile(managed_env_path(), isolated_fcc / ".env")
    isolated_env = {
        "HOME": str(isolated_home),
        "USERPROFILE": str(isolated_home),
        "XDG_CONFIG_HOME": str(isolated_home / "config"),
        "XDG_DATA_HOME": str(isolated_home / "data"),
        "XDG_CACHE_HOME": str(isolated_home / "cache"),
        "XDG_STATE_HOME": str(isolated_home / "state"),
        "OPENCODE_CONFIG_DIR": str(isolated_config),
    }

    with SmokeServerDriver(
        smoke_config,
        name="product-opencode-cli",
        env_overrides={
            **isolated_env,
            "MODEL": provider_model.full_model,
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        env = os.environ.copy()
        env.update(
            {
                "HOST": "127.0.0.1",
                "PORT": str(server.port),
                "FCC_OPEN_BROWSER": "0",
                "ANTHROPIC_AUTH_TOKEN": auth_token,
                **isolated_env,
            }
        )
        env.pop("OPENCODE_CONFIG", None)
        env.pop("OPENCODE_CONFIG_CONTENT", None)
        command = cmd_python_c(
            "from free_claude_code.cli.launchers.opencode import launch; launch()"
        )

        def run(*args: str) -> str:
            process = _start_attached_process([*command, *args], cwd=tmp_path, env=env)
            try:
                stdout, stderr = process.communicate(
                    timeout=smoke_config.timeout_s + 15
                )
                assert process.returncode == 0, stderr or stdout
                assert stdout.strip(), stderr or "OpenCode returned no output"
                return stdout
            finally:
                _stop_attached_process(process)

        marker = f"FCC_SMOKE_OPENCODE_{uuid.uuid4().hex}"
        (tmp_path / "fcc-smoke-marker.txt").write_text(marker, encoding="utf-8")
        output = run(
            "run",
            "--format",
            "json",
            "--model",
            f"free-claude-code/{provider_model.full_model}",
            "--auto",
            "Use the file-reading tool to read fcc-smoke-marker.txt. "
            "Reply with exactly its contents. Do not modify any files.",
        )
        events = _json_object_lines(output)
        assert any(
            event.get("type") == "text"
            and isinstance(part := event.get("part"), dict)
            and marker in str(part.get("text", ""))
            for event in events
        ), output
        assert any(
            event.get("type") == "tool_use"
            and isinstance(part := event.get("part"), dict)
            and isinstance(state := part.get("state"), dict)
            and state.get("status") == "completed"
            and marker in str(state.get("output", ""))
            for event in events
        ), output
        session_id = events[0]["sessionID"]
        assert isinstance(session_id, str) and session_id
        resumed = run(
            "run",
            "--format",
            "json",
            "--session",
            session_id,
            "Reply with that marker again from our conversation. Do not use tools.",
        )
        assert marker in resumed
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    assert "POST /v1/responses" in server_log
    assert "POST /v1/chat/completions" not in server_log
    assert not list((isolated_fcc / "tmp" / "launchers").iterdir())


@pytest.mark.smoke_target("clients")
def test_aider_cli_prompt_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    if not shutil.which("aider"):
        pytest.skip("missing_env: Aider CLI not found")
    uv_bin = shutil.which("uv")
    if not uv_bin:
        pytest.skip("missing_env: uv not found")
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    auth_token = smoke_config.settings.proxy_auth_token
    isolated_home = tmp_path / "aider-home"
    isolated_home.mkdir()

    with SmokeServerDriver(
        smoke_config,
        name="product-aider-cli",
        env_overrides={
            "MODEL": provider_model.full_model,
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        env = os.environ.copy()
        env.update(
            {
                "HOST": "127.0.0.1",
                "PORT": str(server.port),
                "FCC_OPEN_BROWSER": "0",
                "ANTHROPIC_AUTH_TOKEN": auth_token,
                "HOME": str(isolated_home),
                "USERPROFILE": str(isolated_home),
                "PYTHONUTF8": "1",
            }
        )
        env.pop("AIDER_CONFIG_FILE", None)
        result = subprocess.run(
            [
                uv_bin,
                "run",
                "--project",
                str(smoke_config.root),
                "--no-sync",
                "fcc-aider",
                "--no-git",
                "--no-auto-commits",
                "--no-stream",
                "--no-check-update",
                "--no-analytics",
                "--yes-always",
                "--model",
                provider_model.full_model,
                "--message",
                "Reply with exactly FCC_SMOKE_AIDER",
            ],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=smoke_config.timeout_s + 15,
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    assert result.returncode == 0, result.stderr or result.stdout
    assert "FCC_SMOKE_AIDER" in result.stdout
    assert "POST /v1/messages" in server_log
    assert "POST /v1/responses" not in server_log
    assert not any((isolated_home / ".fcc" / "tmp" / "aider").iterdir())


@pytest.mark.smoke_target("clients")
def test_cline_cli_prompt_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    if not shutil.which("cline"):
        pytest.skip("missing_env: Cline CLI not found")
    uv_bin = shutil.which("uv")
    if not uv_bin:
        pytest.skip("missing_env: uv not found")
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    auth_token = smoke_config.settings.proxy_auth_token
    isolated_home = tmp_path / "cline-home"
    isolated_home.mkdir()

    with SmokeServerDriver(
        smoke_config,
        name="product-cline-cli",
        env_overrides={
            "MODEL": provider_model.full_model,
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        env = os.environ.copy()
        env.update(
            {
                "HOST": "127.0.0.1",
                "PORT": str(server.port),
                "FCC_OPEN_BROWSER": "0",
                "ANTHROPIC_AUTH_TOKEN": auth_token,
                "HOME": str(isolated_home),
                "USERPROFILE": str(isolated_home),
            }
        )
        env.pop("CLINE_PROVIDER_SETTINGS_PATH", None)
        env.pop("CLINE_SESSION_BACKEND_MODE", None)
        result = subprocess.run(
            [
                uv_bin,
                "run",
                "--project",
                str(smoke_config.root),
                "--no-sync",
                "fcc-cline",
                "--json",
                "--model",
                provider_model.full_model,
                "Reply with exactly FCC_SMOKE_CLINE",
            ],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=smoke_config.timeout_s + 15,
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    assert result.returncode == 0, result.stderr or result.stdout
    assert "FCC_SMOKE_CLINE" in result.stdout
    assert "POST /v1/responses" in server_log
    assert "POST /v1/chat/completions" not in server_log


@pytest.mark.smoke_target("clients")
def test_hermes_cli_prompt_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    if not shutil.which("hermes"):
        pytest.skip("missing_env: Hermes Agent not found")
    uv_bin = shutil.which("uv")
    if not uv_bin:
        pytest.skip("missing_env: uv not found")

    model_id = "fcc-smoke-hermes"
    full_model = f"lmstudio/{model_id}"
    marker = "FCC_SMOKE_HERMES"
    auth_token = smoke_config.settings.proxy_auth_token
    isolated_home = tmp_path / "home"
    hermes_home = tmp_path / "hermes-home"
    isolated_home.mkdir()
    hermes_home.mkdir()
    native_config = hermes_home / "config.yaml"
    native_env = hermes_home / ".env"
    native_config.write_text(
        json.dumps(
            {
                "model": {
                    "provider": "openrouter",
                    "default": "native/model",
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    native_env.write_text("OPENAI_API_KEY=native-sentinel\n", encoding="utf-8")
    config_before = native_config.read_bytes()
    env_before = native_env.read_bytes()
    credential_env_keys = {
        descriptor.credential_env
        for descriptor in PROVIDER_CATALOG.values()
        if descriptor.credential_env is not None
    }

    with (
        _successful_openai_provider(model_id=model_id, marker=marker) as (
            provider_base_url,
            provider_requests,
        ),
        SmokeServerDriver(
            smoke_config,
            name="product-hermes-cli",
            env_overrides={
                "MODEL": full_model,
                "MODEL_FABLE": full_model,
                "MODEL_OPUS": full_model,
                "MODEL_SONNET": full_model,
                "MODEL_HAIKU": full_model,
                "LM_STUDIO_BASE_URL": provider_base_url,
                "MESSAGING_PLATFORM": "none",
            },
            env_unset=credential_env_keys,
        ).run() as server,
    ):
        env = os.environ.copy()
        for key in credential_env_keys:
            env.pop(key, None)
        for key in tuple(env):
            if key.startswith("FCC_HERMES_"):
                env.pop(key)
        env.update(
            {
                "HOST": "127.0.0.1",
                "PORT": str(server.port),
                "FCC_OPEN_BROWSER": "0",
                "ANTHROPIC_AUTH_TOKEN": auth_token,
                "HOME": str(isolated_home),
                "USERPROFILE": str(isolated_home),
                "HERMES_HOME": str(hermes_home),
            }
        )
        env.pop("HERMES_MANAGED_DIR", None)
        result = subprocess.run(
            [
                uv_bin,
                "run",
                "--project",
                str(smoke_config.root),
                "--no-sync",
                "fcc-hermes",
                "--model",
                full_model,
                "-z",
                f"Reply with exactly {marker}",
            ],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=smoke_config.timeout_s + 15,
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout == f"{marker}\n"
    assert result.stderr == ""
    assert "GET /v1/models" in server_log
    assert server_log.count("POST /v1/responses") == 1
    assert "POST /v1/chat/completions" not in server_log
    assert [request["path"] for request in provider_requests] == [
        "/v1/chat/completions"
    ]
    provider_body = provider_requests[0]["body"]
    assert isinstance(provider_body, dict)
    assert provider_body["model"] == model_id
    assert native_config.read_bytes() == config_before
    assert native_env.read_bytes() == env_before


@pytest.mark.smoke_target("clients")
def test_dsh_cli_headless_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    dsh_bin = shutil.which("dsh")
    if not dsh_bin:
        pytest.skip("missing_env: DeepSeek Harness not found")
    uv_bin = shutil.which("uv")
    if not uv_bin:
        pytest.skip("missing_env: uv not found")

    model_id = "fcc-smoke-dsh"
    full_model = f"lmstudio/{model_id}"
    marker = "FCC_SMOKE_DSH"
    auth_token = smoke_config.settings.proxy_auth_token
    credential_env_keys = _provider_credential_env_keys()

    with (
        _successful_openai_provider(model_id=model_id, marker=marker) as (
            provider_base_url,
            provider_requests,
        ),
        SmokeServerDriver(
            smoke_config,
            name="product-dsh-cli-headless",
            env_overrides=_local_provider_overrides(full_model, provider_base_url),
            env_unset=credential_env_keys,
        ).run() as server,
    ):
        env = _isolated_dsh_env(
            tmp_path=tmp_path,
            server_port=server.port,
            auth_token=auth_token,
            credential_env_keys=credential_env_keys,
        )
        result = subprocess.run(
            [
                uv_bin,
                "run",
                "--project",
                str(smoke_config.root),
                "--no-sync",
                "fcc-dsh",
                "--profile",
                "headless",
                f"Reply with exactly {marker}",
            ],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=smoke_config.timeout_s + 30,
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == marker
    assert auth_token not in combined
    assert "GET /v1/models" in server_log
    assert server_log.count("POST /v1/responses") == 1
    assert "POST /v1/chat/completions" not in server_log
    assert [request["path"] for request in provider_requests] == [
        "/v1/chat/completions"
    ]
    provider_body = provider_requests[0]["body"]
    assert isinstance(provider_body, dict)
    assert provider_body["model"] == model_id


@pytest.mark.smoke_target("clients")
def test_dsh_cli_terminal_failure_e2e(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    if not shutil.which("dsh"):
        pytest.skip("missing_env: DeepSeek Harness not found")
    uv_bin = shutil.which("uv")
    if not uv_bin:
        pytest.skip("missing_env: uv not found")

    full_model = "lmstudio/fcc-smoke-failing-model"
    auth_token = smoke_config.settings.proxy_auth_token
    credential_env_keys = _provider_credential_env_keys()
    with (
        _deliberately_failing_openai_provider() as (
            provider_base_url,
            provider_requests,
        ),
        SmokeServerDriver(
            smoke_config,
            name="product-dsh-cli-provider-error",
            env_overrides=_local_provider_overrides(full_model, provider_base_url),
            env_unset=credential_env_keys,
        ).run() as server,
    ):
        env = _isolated_dsh_env(
            tmp_path=tmp_path,
            server_port=server.port,
            auth_token=auth_token,
            credential_env_keys=credential_env_keys,
        )
        result = subprocess.run(
            [
                uv_bin,
                "run",
                "--project",
                str(smoke_config.root),
                "--no-sync",
                "fcc-dsh",
                "--profile",
                "headless",
                "Reply with exactly FCC_SMOKE_UNREACHABLE",
            ],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=smoke_config.timeout_s + 30,
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert auth_token not in combined
    assert server_log.count("POST /v1/responses") == 1
    assert provider_requests == ["/v1/chat/completions"]


@pytest.mark.smoke_target("clients")
def test_dsh_cli_web_startup_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    if not shutil.which("dsh"):
        pytest.skip("missing_env: DeepSeek Harness not found")
    uv_bin = shutil.which("uv")
    if not uv_bin:
        pytest.skip("missing_env: uv not found")

    model_id = "fcc-smoke-dsh-web"
    full_model = f"lmstudio/{model_id}"
    auth_token = smoke_config.settings.proxy_auth_token
    credential_env_keys = _provider_credential_env_keys()
    web_port = find_free_port()
    with (
        _successful_openai_provider(model_id=model_id, marker="unused") as (
            provider_base_url,
            provider_requests,
        ),
        SmokeServerDriver(
            smoke_config,
            name="product-dsh-cli-web",
            env_overrides=_local_provider_overrides(full_model, provider_base_url),
            env_unset=credential_env_keys,
        ).run() as server,
    ):
        env = _isolated_dsh_env(
            tmp_path=tmp_path,
            server_port=server.port,
            auth_token=auth_token,
            credential_env_keys=credential_env_keys,
        )
        command = [
            uv_bin,
            "run",
            "--project",
            str(smoke_config.root),
            "--no-sync",
            "fcc-dsh",
            "web",
            "--no-open",
            "--port",
            str(web_port),
        ]
        process = _start_attached_process(command, cwd=tmp_path, env=env)
        try:
            _wait_for_web_root(
                process,
                url=f"http://127.0.0.1:{web_port}/",
                timeout_s=smoke_config.timeout_s + 30,
            )
            assert process.poll() is None
        finally:
            stdout, stderr = _stop_attached_process(process)
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    assert process.poll() is not None
    assert auth_token not in f"{stdout}\n{stderr}"
    assert "GET /v1/models" in server_log
    assert provider_requests == []


@pytest.mark.smoke_target("clients")
def test_grok_cli_headless_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    if not shutil.which("grok"):
        pytest.skip("missing_env: Grok Build not found")
    uv_bin = shutil.which("uv")
    if not uv_bin:
        pytest.skip("missing_env: uv not found")

    model_id = "fcc-smoke-grok"
    full_model = f"lmstudio/{model_id}"
    marker = "FCC_SMOKE_GROK"
    auth_token = smoke_config.settings.proxy_auth_token
    credential_env_keys = _provider_credential_env_keys()

    with (
        _successful_openai_provider(model_id=model_id, marker=marker) as (
            provider_base_url,
            provider_requests,
        ),
        SmokeServerDriver(
            smoke_config,
            name="product-grok-cli-headless",
            env_overrides=_local_provider_overrides(full_model, provider_base_url),
            env_unset=credential_env_keys,
        ).run() as server,
    ):
        isolated_home = tmp_path / "grok-home"
        isolated_home.mkdir()
        env = os.environ.copy()
        for key in credential_env_keys:
            env.pop(key, None)
        for key in tuple(env):
            if key.startswith("FCC_GROK_") or key.startswith("GROK_"):
                env.pop(key)
        env.pop("XAI_API_KEY", None)
        env.update(
            {
                "HOST": "127.0.0.1",
                "PORT": str(server.port),
                "FCC_OPEN_BROWSER": "0",
                "ANTHROPIC_AUTH_TOKEN": auth_token,
                "HOME": str(isolated_home),
                "USERPROFILE": str(isolated_home),
            }
        )
        result = subprocess.run(
            [
                uv_bin,
                "run",
                "--project",
                str(smoke_config.root),
                "--no-sync",
                "fcc-grok",
                "--model",
                full_model,
                "--single",
                f"Reply with exactly {marker}",
            ],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=smoke_config.timeout_s + 30,
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, result.stderr or result.stdout
    assert marker in result.stdout
    assert auth_token not in combined
    assert "GET /v1/models?view=responses" in server_log
    responses_count = server_log.count("POST /v1/responses")
    assert responses_count >= 1
    assert "POST /v1/chat/completions" not in server_log
    assert responses_count == len(provider_requests)
    assert {request["path"] for request in provider_requests} == {
        "/v1/chat/completions"
    }
    for request in provider_requests:
        provider_body = request["body"]
        assert isinstance(provider_body, dict)
        assert provider_body["model"] == model_id


@pytest.mark.smoke_target("clients")
def test_muse_cli_headless_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    if not shutil.which("muse"):
        pytest.skip("missing_env: Muse Code not found")
    uv_bin = shutil.which("uv")
    if not uv_bin:
        pytest.skip("missing_env: uv not found")

    model_id = "fcc-smoke-muse"
    full_model = f"lmstudio/{model_id}"
    marker = "FCC_SMOKE_MUSE"
    auth_token = smoke_config.settings.proxy_auth_token
    credential_env_keys = _provider_credential_env_keys()

    with (
        _successful_openai_provider(model_id=model_id, marker=marker) as (
            provider_base_url,
            provider_requests,
        ),
        SmokeServerDriver(
            smoke_config,
            name="product-muse-cli-headless",
            env_overrides=_local_provider_overrides(full_model, provider_base_url),
            env_unset=credential_env_keys,
        ).run() as server,
    ):
        isolated_home = tmp_path / "muse-home"
        isolated_home.mkdir()
        env = os.environ.copy()
        for key in credential_env_keys:
            env.pop(key, None)
        for key in tuple(env):
            if key.startswith("FCC_MUSE_"):
                env.pop(key)
        for key in (
            "META_API_KEY",
            "MUSE_MODEL",
            "MUSE_CUSTOM_HEADERS",
            "MUSE_WWW_ROUTING",
        ):
            env.pop(key, None)
        env.update(
            {
                "HOST": "127.0.0.1",
                "PORT": str(server.port),
                "FCC_OPEN_BROWSER": "0",
                "ANTHROPIC_AUTH_TOKEN": auth_token,
                "HOME": str(isolated_home),
                "USERPROFILE": str(isolated_home),
                "XDG_CONFIG_HOME": str(isolated_home / "config"),
                "XDG_DATA_HOME": str(isolated_home / "data"),
                "XDG_CACHE_HOME": str(isolated_home / "cache"),
                "XDG_STATE_HOME": str(isolated_home / "state"),
                "MUSE_NO_AUTO_UPDATE": "1",
            }
        )
        result = subprocess.run(
            [
                uv_bin,
                "run",
                "--project",
                str(smoke_config.root),
                "--no-sync",
                "fcc-muse",
                "exec",
                "--json",
                "--no-session-log",
                "--disable-web-tools",
                "--disable-write",
                "--disable-shell",
                "--max-model-steps",
                "1",
                "--model",
                full_model,
                f"Reply with exactly {marker}",
            ],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=smoke_config.timeout_s + 30,
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, result.stderr or result.stdout
    assert marker in result.stdout
    assert auth_token not in combined
    assert "GET /v1/models?view=responses" in server_log
    assert "GET /muse-code/models" in server_log
    # Muse accepts a generic OpenAI list but hides rows without its native
    # metadata. Inspect the actual CLI's parsed catalog, not just HTTP success.
    catalog_files = list(
        (isolated_home / "data" / "muse" / "model-catalog").glob("*.json")
    )
    assert catalog_files, "Muse did not persist its parsed provider catalog"
    visible_models = {
        row["model_id"]
        for catalog_file in catalog_files
        for row in json.loads(catalog_file.read_text(encoding="utf-8"))["rows"]
        if row["visibility"] == "visible"
    }
    assert full_model in visible_models, "Muse hides the configured FCC model"
    responses_count = server_log.count("POST /v1/responses")
    assert responses_count >= 1
    assert responses_count == len(provider_requests)
    assert "POST /v1/chat/completions" not in server_log
    assert all(
        request["path"] == "/v1/chat/completions" for request in provider_requests
    )
    for request in provider_requests:
        provider_body = request["body"]
        assert isinstance(provider_body, dict)
        assert provider_body["model"] == model_id


@pytest.mark.smoke_target("cli")
def test_claude_cli_adaptive_thinking_e2e(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    claude_bin = shutil.which(smoke_config.claude_bin)
    if not claude_bin:
        pytest.skip(f"missing_env: Claude CLI not found: {smoke_config.claude_bin}")
    provider_model = ProviderMatrixDriver(smoke_config).first_model()

    with SmokeServerDriver(
        smoke_config,
        name="product-claude-cli-adaptive",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        result = ClientProtocolDriver.run_claude_prompt(
            claude_bin=claude_bin,
            server=server,
            config=smoke_config,
            cwd=tmp_path,
            prompt="think hard, then reply with exactly FCC_SMOKE_CLI",
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    assert result.returncode == 0, result.stderr or result.stdout
    assert "POST /v1/messages" in server_log
    assert " 422 " not in server_log
    assert 'HTTP/1.1" 422' not in server_log
    assert "400 Bad Request" not in result.stdout
    assert "FCC_SMOKE_CLI" in result.stdout


@pytest.mark.smoke_target("cli")
def test_claude_cli_web_search_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    if os.environ.get("FCC_SMOKE_RUN_WEB_TOOLS") != "1":
        pytest.skip("missing_env: set FCC_SMOKE_RUN_WEB_TOOLS=1")
    claude_bin = shutil.which(smoke_config.claude_bin)
    if not claude_bin:
        pytest.skip(f"missing_env: Claude CLI not found: {smoke_config.claude_bin}")
    provider_model = ProviderMatrixDriver(smoke_config).first_model()

    with SmokeServerDriver(
        smoke_config,
        name="product-claude-cli-web-search",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
            "ENABLE_WEB_SERVER_TOOLS": "true",
            "LOG_LEVEL": "DEBUG",
        },
    ).run() as server:
        automatic_turn = ConversationDriver(server, smoke_config).stream(
            {
                "model": provider_model.full_model,
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "You must use the available web search tool to find the "
                            "Free Claude Code GitHub repository."
                        ),
                    }
                ],
                "tools": [
                    {
                        "name": "web_search",
                        "type": "web_search_20250305",
                    }
                ],
                "tool_choice": {"type": "auto"},
            }
        )
        run = run_claude_cli(
            claude_bin=claude_bin,
            server=server,
            config=smoke_config,
            cwd=tmp_path,
            bare=False,
            prompt=(
                "Use WebSearch exactly once to find the Free Claude Code GitHub "
                "repository. Then reply with FCC_SMOKE_WEB_SEARCH and one source URL."
            ),
            tools="WebSearch",
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    assert run.timed_out is False, run.combined_output
    assert run.returncode == 0, run.combined_output
    assert "FCC_SMOKE_WEB_SEARCH" in run.combined_output
    assert "http" in run.combined_output
    automatic_payload = json.dumps(
        [event.data for event in automatic_turn.events], sort_keys=True
    )
    assert '"type": "server_tool_use"' in automatic_payload
    assert '"type": "web_search_tool_result"' in automatic_payload
    assert "github.com" in automatic_payload
    log_rows = _json_object_lines(server_log)
    assert (
        sum(
            row.get("event") == "free_claude_code.api.web_search.automatic_recognized"
            for row in log_rows
        )
        == 1
    ), server_log
    assert any(
        row.get("event") == "free_claude_code.api.optimization.web_server_tool"
        for row in log_rows
    ), server_log
    assert (
        sum(
            row.get("event") == "free_claude_code.api.web_search.automatic_selected"
            for row in log_rows
        )
        == 1
    ), server_log
    assert (
        sum(
            row.get("event") == "free_claude_code.api.web_search.automatic_completed"
            for row in log_rows
        )
        == 1
    ), server_log


@pytest.mark.smoke_target("cli")
def test_claude_auto_mode_openai_connected_e2e(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    claude_bin = shutil.which(smoke_config.claude_bin)
    if not claude_bin:
        pytest.skip(f"missing_env: Claude CLI not found: {smoke_config.claude_bin}")

    provider_models = ProviderMatrixDriver(smoke_config).provider_smoke_models()
    provider_model = next(
        (model for model in provider_models if model.provider == "openai"),
        None,
    )
    if provider_model is None:
        pytest.skip(
            "missing_env: set FCC_SMOKE_MODEL_OPENAI to run the connected-account "
            "Auto-mode smoke"
        )

    marker = f"FCC_SMOKE_AUTO_MODE_{uuid.uuid4().hex}"
    marker_path = tmp_path / "auto-mode-marker.txt"
    marker_path.write_text(marker, encoding="utf-8")
    workspace = tmp_path / "workspace"
    routed_model = provider_model.full_model

    with SmokeServerDriver(
        smoke_config,
        name="product-claude-auto-mode-openai",
        env_overrides={
            "MODEL": routed_model,
            "MODEL_FABLE": routed_model,
            "MODEL_OPUS": routed_model,
            "MODEL_SONNET": routed_model,
            "MODEL_HAIKU": routed_model,
            "MESSAGING_PLATFORM": "none",
            "LOG_LEVEL": "DEBUG",
            "LOG_RAW_API_PAYLOADS": "false",
            "LOG_RAW_SSE_EVENTS": "false",
        },
    ).run() as server:
        run = run_claude_cli(
            claude_bin=claude_bin,
            server=server,
            config=smoke_config,
            cwd=workspace,
            prompt=(
                "Use Bash exactly once to run `cat "
                f'"{marker_path.as_posix()}"`. After the tool succeeds, reply '
                "with exactly the file contents."
            ),
            tools="Bash",
            auto_mode=True,
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    assert run.timed_out is False, run.combined_output
    assert run.returncode == 0, run.combined_output
    cli_events = _json_object_lines(run.stdout)
    assert cli_events, run.stdout
    encoded_events = json.dumps(cli_events, sort_keys=True)
    assert '"type": "tool_use"' in encoded_events
    assert '"name": "Bash"' in encoded_events
    assert '"type": "tool_result"' in encoded_events
    assert marker in encoded_events

    combined_lower = run.combined_output.lower()
    for unexpected in (
        "temporarily unavailable",
        "cannot determine the safety",
        "automode-unavailable",
        "openai responses cannot represent",
    ):
        assert unexpected not in combined_lower

    log_rows = _json_object_lines(server_log)
    policy_rows = [
        row
        for row in log_rows
        if row.get("event") == "free_claude_code.api.route.safety_classifier_policy"
    ]
    assert any(
        row.get("classifier_stop_sequence") == "</severity>"
        and row.get("stop_sequence_removed") is True
        for row in policy_rows
    ), server_log
    message_requests = [
        line for line in server_log.splitlines() if "POST /v1/messages" in line
    ]
    assert len(message_requests) >= 2, server_log
    assert all(" 400 " not in message for message in message_requests), server_log


@pytest.mark.smoke_target("cli")
def test_claude_cli_provider_error_e2e(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    claude_bin = shutil.which(smoke_config.claude_bin)
    if not claude_bin:
        pytest.skip(f"missing_env: Claude CLI not found: {smoke_config.claude_bin}")
    broken_model = "lmstudio/fcc-smoke-failing-model"

    with (
        _deliberately_failing_openai_provider() as (
            provider_base_url,
            provider_requests,
        ),
        SmokeServerDriver(
            smoke_config,
            name="product-claude-cli-provider-error",
            env_overrides={
                "MODEL": broken_model,
                "MODEL_FABLE": broken_model,
                "MODEL_OPUS": broken_model,
                "MODEL_SONNET": broken_model,
                "MODEL_HAIKU": broken_model,
                "LM_STUDIO_BASE_URL": provider_base_url,
                "MESSAGING_PLATFORM": "none",
            },
        ).run() as server,
    ):
        result = ClientProtocolDriver.run_claude_prompt(
            claude_bin=claude_bin,
            server=server,
            config=smoke_config,
            cwd=tmp_path,
            prompt="Reply with exactly FCC_SMOKE_UNREACHABLE.",
            model="claude-sonnet-4-5-20250929",
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    combined = f"{result.stdout}\n{result.stderr}"
    lower = combined.lower()
    failed_downstream_requests = sum(
        "POST /v1/messages" in line and "400 Bad Request" in line
        for line in server_log.splitlines()
    )

    assert result.returncode != 0
    assert "empty or malformed" not in lower
    assert "proxy or gateway intercepting" not in lower
    assert "api error" in lower or "selected model" in lower
    assert "fcc smoke provider rejected the request deliberately" in lower
    assert failed_downstream_requests == 1, server_log
    assert provider_requests == ["/v1/chat/completions"]


def _provider_credential_env_keys() -> set[str]:
    return {
        descriptor.credential_env
        for descriptor in PROVIDER_CATALOG.values()
        if descriptor.credential_env is not None
    }


def _local_provider_overrides(
    full_model: str, provider_base_url: str
) -> dict[str, str]:
    return {
        "MODEL": full_model,
        "MODEL_FABLE": full_model,
        "MODEL_OPUS": full_model,
        "MODEL_SONNET": full_model,
        "MODEL_HAIKU": full_model,
        "LM_STUDIO_BASE_URL": provider_base_url,
        "MESSAGING_PLATFORM": "none",
    }


def _isolated_dsh_env(
    *,
    tmp_path: Path,
    server_port: int,
    auth_token: str,
    credential_env_keys: set[str],
) -> dict[str, str]:
    isolated_home = tmp_path / "home"
    dsh_home = tmp_path / "dsh-home"
    isolated_home.mkdir(exist_ok=True)
    dsh_home.mkdir(exist_ok=True)

    env = os.environ.copy()
    for key in credential_env_keys:
        env.pop(key, None)
    for key in tuple(env):
        if key.startswith("FCC_DSH_"):
            env.pop(key)
    env.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(server_port),
            "FCC_OPEN_BROWSER": "0",
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "HOME": str(isolated_home),
            "USERPROFILE": str(isolated_home),
            "DSH_HOME": str(dsh_home),
        }
    )
    return env


def _start_attached_process(
    command: list[str], *, cwd: Path, env: dict[str, str]
) -> subprocess.Popen[str]:
    if os.name == "nt":
        return subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _wait_for_web_root(
    process: subprocess.Popen[str], *, url: str, timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            response = httpx.get(url, timeout=2.0, follow_redirects=True)
            if response.status_code == HTTPStatus.OK:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.25)
    raise AssertionError(
        "DeepSeek Harness Web did not become ready. "
        f"exit={process.poll()!r} last_error={last_error!r}"
    )


def _stop_attached_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    try:
        return process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
        return process.communicate(timeout=5)


@contextmanager
def _deliberately_failing_openai_provider() -> Iterator[tuple[str, list[str]]]:
    provider_requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "fcc-smoke-failing-model",
                                "object": "model",
                            }
                        ],
                    },
                )
                return
            if self.path == "/api/v0/models":
                self._write_json(HTTPStatus.OK, {"data": []})
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            self.rfile.read(length)
            provider_requests.append(self.path)
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "fcc_smoke_failure",
                        "message": "FCC smoke provider rejected the request deliberately.",
                    }
                },
            )

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write_json(self, status: HTTPStatus, payload: object) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        yield f"http://127.0.0.1:{port}/v1", provider_requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _successful_openai_provider(
    *, model_id: str, marker: str
) -> Iterator[tuple[str, list[JsonObject]]]:
    provider_requests: list[JsonObject] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [{"id": model_id, "object": "model"}],
                    },
                )
                return
            if self.path == "/api/v0/models":
                self._write_json(HTTPStatus.OK, {"data": []})
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            raw_body = self.rfile.read(length)
            parsed: JsonValue = json.loads(raw_body) if raw_body else {}
            if not isinstance(parsed, dict):
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid body"})
                return
            provider_requests.append({"path": self.path, "body": parsed})
            if self.path != "/v1/chat/completions":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "close")
            self.end_headers()
            self._write_chunk(content=marker, finish_reason=None)
            self._write_chunk(content=None, finish_reason="stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write_json(self, status: HTTPStatus, payload: object) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def _write_chunk(
            self, *, content: str | None, finish_reason: str | None
        ) -> None:
            delta: JsonObject = {}
            if content is not None:
                delta = {"role": "assistant", "content": content}
            payload = {
                "id": "chatcmpl-hermes-smoke",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        yield f"http://127.0.0.1:{port}/v1", provider_requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.smoke_target("cli")
def test_claude_cli_multiturn_tool_protocol_e2e(smoke_config: SmokeConfig) -> None:
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    with SmokeServerDriver(
        smoke_config,
        name="product-claude-cli-protocol",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        turn = ConversationDriver(server, smoke_config).stream(
            ClientProtocolDriver.tool_result_payload()
        )

    assert_product_stream(turn.events)
