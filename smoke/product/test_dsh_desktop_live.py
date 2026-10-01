"""Real installed DSH Desktop, isolated profiles, and scripted local inference."""

import json
import os
import queue
import shutil
import threading
from pathlib import Path

import playwright
import pytest

from free_claude_code.application.model_catalog import CatalogModel, ModelCatalog
from free_claude_code.harnesses import dsh_desktop_integration as desktop
from smoke.lib.config import SmokeConfig
from smoke.lib.dsh_provider import DshProvider, dsh_provider
from smoke.lib.e2e import SmokeServerDriver
from smoke.product.test_client_product_live import (
    _isolated_dsh_env,
    _local_provider_overrides,
    _provider_credential_env_keys,
    _start_attached_process,
    _stop_attached_process,
)

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("clients")]


def test_dsh_desktop_live_catalog_credentials_and_restart(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    executable = os.environ.get("FCC_SMOKE_DSH_DESKTOP_BIN")
    if not executable and os.name == "nt":
        executable = str(
            Path(os.environ["LOCALAPPDATA"])
            / "Programs/DeepSeek Harness/DeepSeek Harness.exe"
        )
    if (
        not executable
        or not Path(executable).is_file()
        or not (node := shutil.which("node"))
    ):
        pytest.skip(
            "missing_env: installed DSH Desktop and Node required; set FCC_SMOKE_DSH_DESKTOP_BIN for a custom installation"
        )
    credentials = _provider_credential_env_keys()
    env = _isolated_dsh_env(
        tmp_path=tmp_path,
        server_port=0,
        auth_token="desktop-local-only",
        credential_env_keys=credentials,
    )
    env["DSH_TELEMETRY_DISABLED"] = "1"
    home = Path(env["DSH_HOME"])
    state = tmp_path / "ownership.json"
    options = {
        "executable": executable,
        "userData": str(tmp_path / "electron"),
        "artifacts": str(tmp_path),
        "phase": "initialize",
    }
    config_file = tmp_path / "driver.json"
    driver = [
        node,
        str(smoke_config.root / "smoke/lib/dsh_desktop.cjs"),
        str(Path(playwright.__file__).parent / "driver/package"),
        str(config_file),
    ]
    config_file.write_text(json.dumps(options), encoding="utf-8")
    process = _start_attached_process(driver, cwd=tmp_path, env=env)
    try:
        stdout, stderr = process.communicate(timeout=90)
        assert process.returncode == 0, stdout + stderr
    finally:
        _stop_attached_process(process)
    witness = tmp_path / "witness.txt"
    witness.write_text("DESKTOP_TOOL_WITNESS", encoding="utf-8")
    model = "fcc-desktop-native"
    full_model = f"lmstudio/{model}"
    first = DshProvider(model, "FCC_DESKTOP_DONE", read_path=str(witness))
    second = DshProvider(model, "FCC_DESKTOP_UPDATED", read_path=str(witness))

    def server_env(upstream: str, token: str, name: str) -> dict[str, str]:
        root = tmp_path / name
        (root / ".fcc").mkdir(parents=True)
        (root / ".fcc/.env").write_text(
            f"FCC_CONFIG_SCHEMA=1\nANTHROPIC_AUTH_TOKEN={token}\nPROXY_AUTH_ENABLED=true\n",
            encoding="utf-8",
        )
        return _local_provider_overrides(full_model, upstream) | {
            "HOME": str(root),
            "USERPROFILE": str(root),
            "DSH_HOME": str(home),
        }

    with (
        dsh_provider(first) as upstream,
        dsh_provider(second) as updated,
        SmokeServerDriver(
            smoke_config,
            name="dsh-desktop-native",
            env_overrides=server_env(upstream, "desktop-first", "fcc-first"),
            env_unset=credentials,
        ).run() as server,
        SmokeServerDriver(
            smoke_config,
            name="dsh-desktop-rotated",
            env_overrides=server_env(updated, "desktop-rotated", "fcc-second"),
            env_unset=credentials,
        ).run() as rotated,
    ):

        def configure(url: str, token: str, label: str) -> None:
            catalog = ModelCatalog(
                (CatalogModel(full_model, full_model, label, True),), full_model
            )
            desktop.configure(
                home,
                url,
                token,
                catalog,
                state_path=state,
                provider_progress_timeout=600,
            )

        configure(server.base_url, "desktop-first", "FCC Desktop Fixture")
        config_file.write_text(
            json.dumps(options | {"phase": "exercise"}), encoding="utf-8"
        )
        # The controller exchanges one acknowledgement for each native lifecycle action.
        import subprocess

        process = subprocess.Popen(
            driver,
            cwd=tmp_path,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        lines: queue.Queue[str | None] = queue.Queue()
        assert process.stdout is not None and process.stdin is not None
        output = process.stdout

        def collect() -> None:
            for line in output:
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        transcript = []
        completed = False
        try:
            while (line := lines.get(timeout=60)) is not None:
                transcript.append(line)
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("action") == "refresh":
                    configure(
                        rotated.base_url, "desktop-rotated", "FCC Updated Fixture"
                    )
                elif event.get("action") == "disconnect":
                    desktop.disconnect(home, state_path=state)
                else:
                    completed |= event.get("completed") is True
                    continue
                process.stdin.write("{}\n")
                process.stdin.flush()
            process.wait(timeout=15)
            assert process.returncode == 0 and completed, "".join(transcript)
        finally:
            _stop_attached_process(process)
            reader.join(timeout=5)
            (tmp_path / "desktop-driver.log").write_text(
                "".join(transcript), encoding="utf-8"
            )
            (tmp_path / "desktop-requests.json").write_text(
                json.dumps(
                    {"first": first.requests, "rotated": second.requests}, indent=2
                ),
                encoding="utf-8",
            )
    assert sum(item["purpose"] == "main" for item in first.requests) == 2
    assert sum(item["purpose"] == "main" for item in second.requests) == 4
    for requests in (first.requests, second.requests):
        assert any(
            "DESKTOP_TOOL_WITNESS" in json.dumps(item["body"]) for item in requests
        )
    assert not state.exists()
    saved = (home / "profiles/desktop/cordis.patch.yml").read_text(encoding="utf-8")
    assert "free-claude-code" not in saved
    assert (
        "ui-settings-account" in saved
    )  # Native onboarding writes survived all FCC edits.
