"""Native application identity must survive the installer-created launcher."""

import json
import os
import platform
import plistlib
import shlex
import shutil
import signal
import site
import subprocess
import sys
import sysconfig
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
    baseline = run_probe(
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
    native = home / "Applications/FCC Native Probe.app"
    shutil.copytree(app, native)
    # Signed bundles keep non-code files in Resources.
    (native / "Contents/.free-claude-code-owner").rename(
        native / "Contents/Resources/.free-claude-code-owner"
    )
    executable = native / "Contents/MacOS/fcc-desktop"
    shutil.copyfile(
        ROOT
        / "src/free_claude_code/assets/macos"
        / f"fcc-desktop-{platform.machine()}",
        executable,
    )
    executable.chmod(0o755)
    with (native / "Contents/Resources/runtime.plist").open("wb") as stream:
        plistlib.dump(
            {
                "PythonExecutable": sys.executable,
                "PythonLibrary": str(
                    Path(sys.base_prefix)
                    / "lib"
                    / sysconfig.get_config_var("LDLIBRARY")
                ),
                "PythonVersion": platform.python_version(),
            },
            stream,
        )
    subprocess.run(
        ["/usr/bin/codesign", "--force", "--sign", "-", str(native)], check=True
    )
    subprocess.run(
        ["/usr/bin/codesign", "--verify", "--strict", str(native)], check=True
    )
    packages = Path(site.getsitepackages()[0])
    bootstrap = packages / "zzz_fcc_native_probe.py"
    pth = packages / "zzz_fcc_native_probe.pth"
    bootstrap.write_text(
        "import os\n"
        "if 'FCC_MACOS_PROBE_OUTPUT' in os.environ:\n"
        "    import runpy\n"
        "    from free_claude_code.cli import desktop_entrypoint\n"
        f"    desktop_entrypoint.launch = runpy.run_path({str(probe)!r})['main']\n",
        encoding="utf-8",
    )
    pth.write_text("import zzz_fcc_native_probe\n", encoding="utf-8")
    output = output_dir / "native.json"
    try:
        actual = run_probe(
            [
                "/usr/bin/open",
                "-n",
                "-W",
                "--env",
                f"HOME={home}",
                "--env",
                f"FCC_MACOS_PROBE_OUTPUT={output}",
                "--stderr",
                str(output_dir / "native-stderr.log"),
                str(native),
            ],
            output,
            env,
        )
    finally:
        pth.unlink()
        bootstrap.unlink()
    return native, terminal, baseline, actual


def test_finder_launch_keeps_fcc_bundle_identity(launched_app):
    app, _terminal, baseline, actual = launched_app
    assert baseline["bundle_id"] != "io.github.alishahryar1.free-claude-code"
    assert actual["bundle_id"] == "io.github.alishahryar1.free-claude-code"
    assert Path(actual["bundle_path"]) == app
    assert actual["main_thread"]


def test_finder_status_item_is_placed_on_screen(launched_app):
    _app, terminal, _baseline, actual = launched_app
    assert terminal["intersects_screen"], terminal
    assert actual["visible"] and not actual["button_hidden"], actual
    assert actual["intersects_screen"], actual
