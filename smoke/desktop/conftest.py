"""Native desktop checks use a private installed environment and application."""

import os
import plistlib
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def installed_desktop(tmp_path_factory):
    if sys.platform != "darwin":
        pytest.skip("requires a macOS GUI session")
    root = tmp_path_factory.mktemp("FCC's 日本語 desktop")
    home = root / "home"
    home.mkdir()
    env = os.environ | {"HOME": str(home)}
    # CI's legacy preference option conflicts with the explicit managed-only flag.
    env.pop("UV_PYTHON_PREFERENCE", None)
    output = ROOT / "test-results/native"
    output.mkdir(parents=True, exist_ok=True)
    dist = output / "dist"
    subprocess.run(
        ["uv", "build", "--out-dir", str(dist)],
        cwd=ROOT,
        env=env | {"SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0"},
        check=True,
        timeout=120,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_release.py"),
            str(dist),
            "--version",
            "0.0.0",
        ],
        cwd=ROOT,
        check=True,
        timeout=30,
    )
    requirements = root / "requirements.txt"
    subprocess.run(
        [
            "uv",
            "export",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--output-file",
            str(requirements),
        ],
        cwd=ROOT,
        check=True,
        timeout=60,
        stdout=subprocess.DEVNULL,
    )
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "requires-python"
    ].removeprefix("==")

    def create_venv(venv):
        subprocess.run(
            ["uv", "venv", "--managed-python", "--python", f"{version}+gil", str(venv)],
            env=env,
            check=True,
            timeout=60,
        )
        python = venv / "bin/python"
        subprocess.run(
            ["uv", "pip", "sync", "--python", str(python), str(requirements)],
            env=env,
            check=True,
            timeout=180,
        )
        wheel = next(dist.glob("*.whl"))
        subprocess.run(
            ["uv", "pip", "install", "--no-deps", "--python", str(python), str(wheel)],
            env=env,
            check=True,
            timeout=60,
        )
        packages = Path(
            subprocess.check_output(
                [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
                env=env,
                text=True,
            ).strip()
        )
        for source, target in (
            ("identity_probe.py", "fcc_identity_probe.py"),
            ("lifecycle_probe.py", "fcc_native_probe.py"),
        ):
            shutil.copyfile(Path(__file__).with_name(source), packages / target)
        (packages / "zzz_fcc_native_probe.pth").write_text(
            "import fcc_native_probe; fcc_native_probe.install()\n"
        )

    venv = root / "installed venv"
    python = venv / "bin/python"
    create_venv(venv)
    subprocess.run(
        [str(venv / "bin/fcc-desktop"), "--install-macos-app"],
        env=env,
        check=True,
        timeout=30,
    )
    app = home / "Applications/Free Claude Code.app"
    with (app / "Contents/Resources/runtime.plist").open("rb") as stream:
        runtime = plistlib.load(stream)
    # Launch Services receives explicit private settings; no launchd-wide changes.
    launch_env = {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "FCC_OPEN_BROWSER": "false",
        "HOST": "127.0.0.1",
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
    }
    yield {
        "root": root,
        "home": home,
        "venv": venv,
        "python": python,
        "app": app,
        "output": output,
        "env": env,
        "launch_env": launch_env,
        "runtime": runtime,
        "create_venv": create_venv,
    }
