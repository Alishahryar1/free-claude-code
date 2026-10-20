"""Smoke boundaries consume and protect ordered provider credentials."""

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from smoke.lib.config import SmokeConfig, redacted
from smoke.product import test_codex_modes_product_live as codex_smoke


@pytest.mark.parametrize(
    "key",
    sorted(
        {
            descriptor.credential_env
            for descriptor in PROVIDER_CATALOG.values()
            if descriptor.credential_env
        }
    ),
)
def test_smoke_redaction_expands_every_catalog_credential_collection(key):
    first, second = "private-primary-key", "private-primary-key-with-suffix"
    encoded = json.dumps([first, second])
    message = f"{first}; {second}; {encoded}"
    assert redacted(message, {key: encoded}) == (
        f"<redacted:{key}>; <redacted:{key}>; <redacted:{key}>"
    )


def test_smoke_redaction_masks_decoded_escaped_and_unicode_keys():
    encoded = json.dumps(['quoted-"secret"-key', "秘密-secret-key"])
    result = redacted(
        'Credentials quoted-"secret"-key and 秘密-secret-key',
        {"OPENAI_API_KEY": encoded},
    )
    assert result == (
        "Credentials <redacted:OPENAI_API_KEY> and <redacted:OPENAI_API_KEY>"
    )


def test_smoke_redaction_preserves_scalar_secret_handling():
    assert (
        redacted(
            "Bearer scalar-provider-secret, bot scalar-bot-secret",
            {
                "OPENAI_API_KEY": "scalar-provider-secret",
                "TELEGRAM_BOT_TOKEN": "scalar-bot-secret",
            },
        )
        == "Bearer <redacted:OPENAI_API_KEY>, bot <redacted:TELEGRAM_BOT_TOKEN>"
    )


def test_smoke_redaction_does_not_fail_while_reporting_malformed_array_input():
    malformed = '["unfinished-private-key"'
    assert (
        redacted(f"Failed to parse {malformed}", {"OPENAI_API_KEY": malformed})
        == "Failed to parse <redacted:OPENAI_API_KEY>"
    )


@pytest.mark.parametrize(
    "credential", ["synthetic-first", '["synthetic-first","synthetic-tail"]']
)
def test_live_codex_smoke_exports_first_key_as_subprocess_text(
    monkeypatch, tmp_path, credential
):
    monkeypatch.setenv("FCC_SMOKE_CODEX_FREE_MODEL", "open_router/synthetic/free:free")
    monkeypatch.setattr(
        codex_smoke.shutil, "which", lambda name: str(tmp_path / "bin" / "codex")
    )
    monkeypatch.setattr(
        codex_smoke,
        "run_captured_text",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            ["codex", "--version"], 0, stdout="codex synthetic\n", stderr=""
        ),
    )
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://openrouter.ai/api/v1/models"),
        json={
            "data": [
                {
                    "id": "synthetic/free:free",
                    "pricing": {"prompt": "0", "completion": "0"},
                }
            ]
        },
    )
    monkeypatch.setattr(codex_smoke.httpx, "get", lambda *args, **kwargs: response)

    class CapturedLaunch(Exception):
        pass

    def capture(command, **kwargs):
        env = kwargs["env"]
        assert env["OPENROUTER_API_KEY"] == "synthetic-first"
        assert all(isinstance(value, str) for value in env.values())
        raise CapturedLaunch

    monkeypatch.setattr("smoke.lib.server.subprocess.Popen", capture)
    config = SmokeConfig(
        root=Path.cwd(),
        results_dir=tmp_path / "results",
        live=False,
        interactive=False,
        targets=frozenset(),
        provider_matrix=frozenset(),
        timeout_s=1,
        prompt="unused",
        claude_bin="unused",
        worker_id="test",
        settings=Settings(OPENROUTER_API_KEY=credential),
    )
    with pytest.raises(CapturedLaunch):
        codex_smoke.test_codex_modes_free_provider_e2e(config, tmp_path)
