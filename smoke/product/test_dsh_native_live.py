"""Native DSH operations against local, isolated FCC providers."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from smoke.lib.config import SmokeConfig
from smoke.lib.dsh_provider import DshProvider, dsh_provider
from smoke.lib.e2e import SmokeServerDriver
from smoke.product.test_client_product_live import (
    _isolated_dsh_env,
    _json_object_lines,
    _local_provider_overrides,
    _provider_credential_env_keys,
)

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("clients")]


def command(config: SmokeConfig) -> list[str]:
    if not shutil.which("dsh") or not (uv := shutil.which("uv")):
        pytest.skip("missing_env: DeepSeek Harness and uv are required")
    return [uv, "run", "--project", str(config.root), "--no-sync", "fcc-dsh"]


@pytest.mark.parametrize(
    "operation", ["subagent", "idle_tool", "native_responses", "vision_no_thinking"]
)
def test_dsh_native_operations(
    smoke_config: SmokeConfig, tmp_path: Path, operation: str
) -> None:
    native = command(smoke_config)
    model = "fcc-dsh-operations"
    full_model = f"lmstudio/{model}"
    scenario = DshProvider(model, "FCC_NATIVE_DONE")
    if operation == "subagent":
        scenario.tool_call = (
            "subagent",
            {
                "description": "Read a fixture instruction",
                "prompt": "FCC_CHILD_TASK: Reply FCC_CHILD_DONE",
                "run_in_background": False,
            },
        )
    elif operation == "idle_tool":
        scenario.tool_call = (
            "pwsh" if os.name == "nt" else "bash",
            {
                "command": "Start-Sleep -Seconds 3; Write-Output 'FCC_IDLE_TOOL_DONE'"
                if os.name == "nt"
                else "sleep 3; printf FCC_IDLE_TOOL_DONE",
                "description": "Wait briefly and print a fixture marker",
                "timeoutMs": 10000,
            },
        )
    elif operation == "vision_no_thinking":
        full_model = f"alibaba_cloud/{model}"
        image = tmp_path / "vision.png"
        shutil.copyfile(smoke_config.root / "smoke/assets/vision-fcc.png", image)
        scenario.tool_call = ("read_image", {"file_path": str(image)})
    else:
        scenario.protocol = "responses"
        full_model = "custom_" + "d" * 32 + "/" + model
    credentials = _provider_credential_env_keys()
    with dsh_provider(scenario) as upstream:
        overrides = _local_provider_overrides(full_model, upstream)
        if operation == "vision_no_thinking":
            overrides.update(
                ALIBABA_CLOUD_BASE_URL=upstream,
                ALIBABA_CLOUD_API_KEY="local-fixture-only",
            )
        if operation == "native_responses":
            overrides["FCC_CUSTOM_PROVIDERS"] = json.dumps(
                [
                    {
                        "provider_id": full_model.split("/")[0],
                        "display_name": "Native Responses fixture",
                        "base_url": upstream,
                        "api_format": "openai_responses",
                        "model_ids": [model],
                    }
                ]
            )
        with SmokeServerDriver(
            smoke_config,
            name=f"dsh-native-{operation}",
            env_overrides=overrides,
            env_unset=credentials,
        ).run() as server:
            env = _isolated_dsh_env(
                tmp_path=tmp_path,
                server_port=server.port,
                auth_token=smoke_config.settings.proxy_auth_token,
                credential_env_keys=credentials,
            )
            result = subprocess.run(
                [*native, "headless", "--json", "Complete the fixture task."],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
    (tmp_path / "native-requests.json").write_text(
        json.dumps(scenario.requests, indent=2), encoding="utf-8"
    )
    (tmp_path / "native-output.log").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = _json_object_lines(result.stdout)
    assert any(
        item.get("type") == "final" and item.get("text") == "FCC_NATIVE_DONE"
        for item in events
    )
    if operation in ("subagent", "idle_tool"):
        witness = "FCC_CHILD_DONE" if operation == "subagent" else "FCC_IDLE_TOOL_DONE"
        assert any(
            item.get("type") == "tool_result"
            and item.get("status") == "completed"
            and witness in str(item.get("result"))
            for item in events
        )
        assert sum(item["purpose"] == "main" for item in scenario.requests) == 2
    if operation == "subagent":
        assert sum(item["purpose"] == "subagent" for item in scenario.requests) == 1
    if operation == "native_responses":
        assert all(item["path"] == "/v1/responses" for item in scenario.requests)
        assert sum(item["purpose"] == "main" for item in scenario.requests) == 1
    if operation == "vision_no_thinking":
        assert any(
            item.get("type") == "tool_result" and item.get("status") == "completed"
            for item in events
        )
        assert not any(item.get("type") == "thinking" for item in events)
        main = [item["body"] for item in scenario.requests if item["purpose"] == "main"]
        assert len(main) == 2
        assert all(isinstance(body, dict) and body["model"] == model for body in main)
        assert "data:image/png;base64," in json.dumps(main[-1])


def test_dsh_native_settings_migration_and_dump(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    native = command(smoke_config)
    scenario = DshProvider("fcc-dsh-migration", "FCC_MIGRATED")
    full_model = "lmstudio/fcc-dsh-migration"
    credentials = _provider_credential_env_keys()
    with (
        dsh_provider(scenario) as upstream,
        SmokeServerDriver(
            smoke_config,
            name="dsh-native-migration",
            env_overrides=_local_provider_overrides(full_model, upstream),
            env_unset=credentials,
        ).run() as server,
    ):
        env = _isolated_dsh_env(
            tmp_path=tmp_path,
            server_port=server.port,
            auth_token=smoke_config.settings.proxy_auth_token,
            credential_env_keys=credentials,
        )
        home = Path(env["DSH_HOME"])
        legacy = home / "settings.yaml"
        legacy.write_text("agent-loop:\n  maxParallelToolCalls: 3\n", encoding="utf-8")
        result = subprocess.run(
            [*native, "headless", "Reply FCC_MIGRATED"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert not legacy.exists()
        assert legacy.with_suffix(".yaml.imported").exists()
        persisted = (home / "profiles/headless/cordis.patch.yml").read_text(
            encoding="utf-8"
        )
        assert "maxParallelToolCalls: 3" in persisted
        assert (
            "free-claude-code" not in persisted and "FCC_DSH_API_KEY" not in persisted
        )
        assert smoke_config.settings.proxy_auth_token not in persisted
        user_patch = tmp_path / "caller.patch.yml"
        user_patch.write_text(
            "- id: agent-default-model\n  config:\n    provider: free-claude-code\n    model: caller-precedence\n",
            encoding="utf-8",
        )
        for selection in (
            ["headless"],
            ["--profile=headless"],
            ["--profile", "headless"],
        ):
            dumped = subprocess.run(
                [*native, *selection, "--patch", str(user_patch), "--dump-config"],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
            )
            assert dumped.returncode == 0, dumped.stdout + dumped.stderr
            parsed = YAML(typ="rt").load(dumped.stdout)
            assert "caller-precedence" in str(parsed)
            assert "openai-responses" in str(parsed)
        for invalid in (
            ["headless", "--profile", "web"],
            ["desktop"],
            ["headless", "--dump-config", "--dump-config-schema"],
        ):
            error = subprocess.run(
                [*native, *invalid],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
            )
            assert error.returncode != 0
    assert sum(item["purpose"] == "main" for item in scenario.requests) == 1
