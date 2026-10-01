"""Real installed DSH Desktop, isolated profiles, and scripted local inference."""

import copy
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import playwright
import pytest

from free_claude_code.application.model_catalog import CatalogModel, ModelCatalog
from free_claude_code.harnesses import dsh_desktop_integration as desktop
from free_claude_code.harnesses import dsh_files
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


@pytest.mark.parametrize("copied_config", [False, True])
def test_dsh_desktop_native_composition_e2e(
    tmp_path: Path, copied_config: bool
) -> None:
    node, npm = shutil.which("node"), shutil.which("npm")
    if not node or not npm or not shutil.which("dsh"):
        pytest.skip(
            "missing_env: installed DSH CLI, Node, and npm required for native composition"
        )
    result = subprocess.run(
        [npm, "root", "-g"], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    native_package = Path(result.stdout.strip()) / "@deepseek-ai/dsh"
    app_boot = native_package / "node_modules/@deepseek-ai/dsh-app-boot/lib/index.js"
    if not app_boot.is_file():
        pytest.skip("missing_env: DSH native app-boot package unavailable")
    profile = tmp_path / "home/profiles/desktop"
    profile.mkdir(parents=True)
    (profile / "package.json").write_text(
        json.dumps(
            {
                "dsh": {
                    "profile": {
                        "bundles": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    path = profile / "cordis.patch.yml"
    path.write_text(
        "- id: llm-pi-ai\n  config:\n    providers:\n      native-user-provider:\n        api: openai-responses\n        baseURL: http://127.0.0.1:9999/v1\n        models: [{id: native-model}]\n",
        encoding="utf-8",
    )
    rows = dsh_files.read_yaml(path, sequence=True)
    rows.append(copy.deepcopy(rows[0]) if copied_config else {"id": "llm-pi-ai"})
    dsh_files.write_yaml(path, rows)
    script = (
        "import { loadProfileDirectory, composeEntries } from "
        + json.dumps(app_boot.as_uri())
        + ";\n"
        "const loaded = loadProfileDirectory('dsh', process.argv[1], process.argv[2]);\n"
        "const entries = composeEntries([...loaded.layers.map(layer => layer.patches), loaded.patches]);\n"
        "const row = entries.find(entry => entry.id === 'llm-pi-ai');\n"
        "console.log(JSON.stringify(Object.keys(row.config?.providers ?? {}).sort()));\n"
    )

    def composed() -> list[str]:
        result = subprocess.run(
            [
                node,
                "--input-type=module",
                "-e",
                script,
                str(profile),
                str(native_package / "package.json"),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout.splitlines()[-1])

    assert composed() == ["native-user-provider"]
    model = CatalogModel("fixture/model", "fixture/model", "Fixture", True)
    state = tmp_path / "ownership.json"
    desktop.configure(
        tmp_path / "home",
        "http://127.0.0.1:8182",
        "fixture-only",
        ModelCatalog((model,), model.wire_slug),
        state_path=state,
        provider_progress_timeout=600,
    )
    assert composed() == ["free-claude-code", "native-user-provider"]
    rows = dsh_files.read_yaml(path, sequence=True)
    rows.append(copy.deepcopy(rows[-1]))
    dsh_files.write_yaml(path, rows)
    desktop.disconnect(tmp_path / "home", state_path=state)
    assert composed() == ["native-user-provider"]

    atomic_write = (
        native_package / "node_modules/@deepseek-ai/dsh-atomic-write/lib/index.js"
    )
    import_lock = (
        "import { withFileLock } from " + json.dumps(atomic_write.as_uri()) + ";\n"
    )
    contender = (
        import_lock
        + """
try {
  await withFileLock(process.argv[1], async () => {}, { waitMs: 0 });
  process.exitCode = 1;
} catch (error) {
  if (!error.message.includes('timed out waiting for the writer lock')) throw error;
}
"""
    )
    lock = profile / "package.json.lock"
    with dsh_files.file_lock(lock, wait=0):
        result = subprocess.run(
            [
                node,
                "--input-type=module",
                "-e",
                contender,
                str(profile / "package.json"),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        assert lock.exists()
    python_contender = """
import sys
from pathlib import Path
from free_claude_code.harnesses.dsh_files import file_lock
try:
    with file_lock(Path(sys.argv[1]), wait=0):
        sys.exit(1)
except TimeoutError:
    pass
"""
    holder = (
        import_lock
        + """
import { spawnSync } from 'node:child_process';
await withFileLock(process.argv[1], async () => {
  const child = spawnSync(process.argv[2], ['-c', process.argv[3], process.argv[1] + '.lock'], { encoding: 'utf8' });
  if (child.status !== 0) throw new Error(child.stderr || 'FCC did not honor the native lock');
});
"""
    )
    result = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            holder,
            str(profile / "package.json"),
            sys.executable,
            python_contender,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert not lock.exists()


def test_dsh_desktop_live_catalog_credentials_and_restart_e2e(
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
                    peer = home / "profiles/web"
                    peer.mkdir(exist_ok=True)
                    (peer / "package.json").write_bytes(
                        (home / "profiles/desktop/package.json").read_bytes()
                    )
                    (peer / "cordis.patch.yml").write_text(
                        "- id: llm-deepseek\n  config:\n    apiKeyEnv: FCC_DSH_DESKTOP_API_KEY\n",
                        encoding="utf-8",
                    )
                    result = desktop.disconnect(home, state_path=state)
                    assert result["credential_retained"] and not result["configured"]
                elif event.get("action") == "reconnect":
                    configure(
                        rotated.base_url, "desktop-rotated", "FCC Updated Fixture"
                    )
                elif event.get("action") == "final_disconnect":
                    (home / "profiles/web/cordis.patch.yml").write_text(
                        "[]\n", encoding="utf-8"
                    )
                    assert not desktop.disconnect(home, state_path=state)[
                        "credential_retained"
                    ]
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
    assert sum(item["purpose"] == "main" for item in second.requests) == 6
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
