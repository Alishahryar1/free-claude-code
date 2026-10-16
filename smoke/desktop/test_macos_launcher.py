"""Native application identity must survive the installer-created launcher."""

import json
import os
import shlex
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS")


def run_probe(command, output, env):
    try:
        result = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert output.exists(), "The launched app did not write its identity"
        report = json.loads(output.read_text(encoding="utf-8"))
        print(json.dumps(report, indent=2))
        return report
    finally:
        if output.exists():
            pid = json.loads(output.read_text(encoding="utf-8"))["pid"]
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)


@pytest.fixture(scope="module")
def launched_app(tmp_path_factory):
    home = tmp_path_factory.mktemp("fcc native home")
    tool_bin = home / "tool's bin"
    tool_bin.mkdir()
    command = tool_bin / "fcc-desktop"
    probe = Path(__file__).with_name("identity_probe.py")
    command.write_text(
        f"#!{sys.executable}\n"
        "import runpy\n"
        f"runpy.run_path({str(probe)!r}, run_name='__main__')\n",
        encoding="utf-8",
    )
    command.chmod(0o755)
    env = os.environ | {"HOME": str(home)}
    source = (ROOT / "scripts/install.sh").read_text(encoding="utf-8")
    scenario = home / "install-app.sh"
    scenario.write_text(
        source.split('\nparse_args "$@"\n', 1)[0]
        + f"\ntool_bin={shlex.quote(str(tool_bin))}\ninstall_macos_desktop_app\n",
        encoding="utf-8",
    )
    subprocess.run(["/bin/sh", str(scenario)], env=env, check=True, timeout=30)
    output_dir = ROOT / "test-results/native"
    output_dir.mkdir(parents=True, exist_ok=True)
    terminal_output = output_dir / "terminal.json"
    terminal = run_probe(
        [str(command)],
        terminal_output,
        env | {"FCC_MACOS_PROBE_OUTPUT": str(terminal_output)},
    )
    app = home / "Applications/Free Claude Code.app"
    output = output_dir / "app.json"
    actual = run_probe(
        [
            "/usr/bin/open",
            "-n",
            "-W",
            "--env",
            f"HOME={home}",
            "--env",
            f"FCC_MACOS_PROBE_OUTPUT={output}",
            str(app),
        ],
        output,
        env,
    )
    return app, terminal, actual


def test_finder_launch_keeps_fcc_bundle_identity(launched_app):
    app, _terminal, actual = launched_app
    assert actual["bundle_id"] == "io.github.alishahryar1.free-claude-code"
    assert Path(actual["bundle_path"]) == app
    assert actual["main_thread"]


def test_finder_status_item_is_placed_on_screen(launched_app):
    _app, terminal, actual = launched_app
    assert terminal["intersects_screen"], terminal
    assert actual["visible"] and not actual["button_hidden"], actual
    assert actual["intersects_screen"], actual
