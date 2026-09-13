import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from free_claude_code.harnesses.codex import codex_config_args
from free_claude_code.harnesses.codex_model_catalog import (
    build_codex_model_catalog,
)
from free_claude_code.harnesses.model_catalog import client_models_from_response
from free_claude_code.runtime.codex_catalog import write_codex_model_catalog
from tests.harnesses.test_codex_model_catalog import _models_payload


def test_launcher_config_composes_with_persistent_codex_config(
    tmp_path: Path,
) -> None:
    codex_binary = shutil.which("codex")
    if codex_binary is None:
        pytest.skip("Codex CLI is not installed")

    catalog_path = tmp_path / "codex-model-catalog.json"
    write_codex_model_catalog(
        catalog_path,
        build_codex_model_catalog(
            client_models_from_response(_models_payload("nvidia_nim/test-model"))
        ),
    )
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        "\n".join(
            (
                'model_provider = "fcc"',
                'model = "nvidia_nim/test-model"',
                f"model_catalog_json = {json.dumps(str(catalog_path))}",
                "",
                "[model_providers.fcc]",
                'name = "Free Claude Code"',
                'base_url = "http://127.0.0.1:8082/v1"',
                'wire_api = "responses"',
                "",
                "[model_providers.fcc.auth]",
                'command = "fcc-codex"',
                'args = ["--print-proxy-auth-token"]',
                "",
            )
        ),
        encoding="utf-8",
    )
    codex_env = os.environ.copy()
    for key in (
        "CODEX_THREAD_ID",
        "CODEX_INTERNAL_ORIGINATOR_OVERRIDE",
        "CODEX_SHELL",
        "CODEX_PERMISSION_PROFILE",
    ):
        codex_env.pop(key, None)
    codex_env["CODEX_HOME"] = str(codex_home)

    result = subprocess.run(
        [
            codex_binary,
            *codex_config_args(api_url="http://127.0.0.1:8082/v1"),
            "debug",
            "models",
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
        env=codex_env,
        errors="replace",
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "nvidia_nim/test-model" in result.stdout
