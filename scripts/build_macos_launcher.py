"""Build the two macOS launcher templates from this exact source checkout."""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"arm64": "11.0", "x86_64": "10.15"}


def build() -> None:
    if sys.platform != "darwin":
        raise RuntimeError(
            "Building macOS launcher templates requires macOS and Xcode tools"
        )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirement = project["project"]["requires-python"]
    match = re.fullmatch(r">=([0-9]+)\.([0-9]+),<([0-9]+)\.([0-9]+)", requirement)
    if match is None:
        raise ValueError("Expected a bounded Python minor-series requirement")
    major, minor, upper_major, upper_minor = map(int, match.groups())
    if (upper_major, upper_minor) != (major, minor + 1):
        raise ValueError(
            "Expected requires-python to cover exactly one Python minor series"
        )
    version = f"{major}.{minor}"
    destination = ROOT / "src/free_claude_code/assets/macos"
    destination.mkdir(parents=True, exist_ok=True)
    for architecture, minimum in TARGETS.items():
        output = destination / f"fcc-desktop-{architecture}"
        subprocess.run(
            [
                "xcrun",
                "clang",
                "-arch",
                architecture,
                f"-mmacosx-version-min={minimum}",
                "-fobjc-arc",
                "-Os",
                "-Wall",
                "-Wextra",
                "-Werror",
                f'-DFCC_PYTHON_VERSION="{version}"',
                "-framework",
                "AppKit",
                "-framework",
                "Foundation",
                str(ROOT / "scripts/macos/launcher.m"),
                "-o",
                str(output),
            ],
            check=True,
        )
        subprocess.run(
            ["/usr/bin/lipo", str(output), "-verify_arch", architecture], check=True
        )
        subprocess.run(["/usr/bin/otool", "-L", str(output)], check=True)
        subprocess.run(["/usr/bin/otool", "-l", str(output)], check=True)


if __name__ == "__main__":
    build()
