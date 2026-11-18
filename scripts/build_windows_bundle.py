"""Stage the exact FCC wheel, locked dependencies, and native Windows launchers."""

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path
from urllib.request import urlopen


def runtime_download(version: str, manifest: dict) -> tuple[str, str]:
    url = f"https://www.python.org/ftp/python/{version}/python-{version}-amd64.zip"
    matches = [item for item in manifest["versions"] if item.get("url") == url]
    if len(matches) != 1:
        raise ValueError(
            "Expected one pinned x64 Python runtime in the release manifest"
        )
    digest = matches[0]["hash"]["sha256"]
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Invalid x64 Python runtime digest")
    return url, digest


def download(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def extract_runtime(data: bytes, digest: str, destination: Path) -> None:
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Python runtime digest mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(destination)


def install_packages(source: Path, runtime: Path, wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="fcc-windows-build-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        requirements = root / "requirements.txt"
        python = environment / "Scripts/python.exe"
        subprocess.run(
            [
                "uv",
                "export",
                "--locked",
                "--no-default-groups",
                "--no-emit-project",
                "--output-file",
                str(requirements),
            ],
            cwd=source,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "uv",
                "venv",
                "--no-project",
                "--python",
                str(runtime / "python.exe"),
                str(environment),
            ],
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "sync",
                "--python",
                str(python),
                "--require-hashes",
                "--only-binary",
                ":all:",
                str(requirements),
            ],
            check=True,
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
            check=True,
        )
        shutil.copytree(
            environment / "Lib/site-packages",
            runtime / "Lib/site-packages",
            dirs_exist_ok=True,
        )


def build(source: Path, wheel: Path, version: str, destination: Path) -> None:
    if sys.platform != "win32":
        raise ValueError("The Windows release must be built on Windows")
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}", version):
        raise ValueError("Expected a canonical FCC release version")
    if wheel.name != f"free_claude_code-{version}-py3-none-any.whl":
        raise ValueError("Expected the exact FCC release wheel")
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    python_version = project["requires-python"].removeprefix("==")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", python_version):
        raise ValueError("Expected an exact Python requirement")
    manifest_url = f"https://www.python.org/ftp/python/{python_version}/windows-{python_version}.json"
    url, digest = runtime_download(python_version, json.loads(download(manifest_url)))

    destination.mkdir()
    runtime = destination / "runtime"
    commands = destination / "bin"
    commands.mkdir()
    extract_runtime(download(url), digest, runtime)
    install_packages(source, runtime, wheel)
    # Keep Python's DLLs, stdlib, and installed packages self-contained after relocation.
    python_abi = "".join(python_version.split(".")[:2])
    (runtime / f"python{python_abi}._pth").write_text(
        ".\nDLLs\nLib\nLib\\site-packages\nimport site\n", encoding="ascii"
    )
    launcher_source = source / "scripts/windows/launcher.c"
    with tempfile.TemporaryDirectory(prefix="fcc-windows-launcher-") as temporary:
        for group, subsystem in (("scripts", "CONSOLE"), ("gui-scripts", "WINDOWS")):
            launcher = Path(temporary) / f"launcher-{subsystem}.exe"
            subprocess.run(
                [
                    "cl.exe",
                    "/nologo",
                    "/W4",
                    "/WX",
                    "/O2",
                    "/MT",
                    str(launcher_source),
                    f"/Fe{launcher}",
                    "/link",
                    "/MACHINE:X64",
                    f"/SUBSYSTEM:{subsystem}",
                ],
                cwd=temporary,
                check=True,
            )
            for command in project[group]:
                shutil.copy2(launcher, commands / f"{command}.exe")
    shutil.copy2(source / "scripts/update/fcc-update.cmd", commands)
    (destination / "version.txt").write_text(version + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build(
        Path(__file__).resolve().parents[1],
        arguments.wheel.resolve(strict=True),
        arguments.version,
        arguments.output.resolve(),
    )


if __name__ == "__main__":
    main()
