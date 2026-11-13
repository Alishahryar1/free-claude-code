"""Verify a published-style sdist retains both launchers on a non-Mac builder."""

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def check(directory: Path) -> None:
    if sys.platform == "darwin":
        raise ValueError("Run this portability check on Windows or Linux")
    source = next(directory.glob("*.tar.gz"))
    original = next(directory.glob("*.whl"))
    with tempfile.TemporaryDirectory(prefix="fcc-sdist-rebuild-") as temporary:
        subprocess.run(
            ["uv", "build", "--wheel", str(source), "--out-dir", temporary],
            check=True,
        )
        rebuilt = next(Path(temporary).glob("*.whl"))
        assert rebuilt.name == original.name
        with zipfile.ZipFile(original) as before, zipfile.ZipFile(rebuilt) as after:
            for architecture in ("arm64", "x86_64"):
                name = f"free_claude_code/assets/macos/fcc-desktop-{architecture}"
                assert before.read(name) and before.read(name) == after.read(name)
                print(f"Preserved {name}: {len(after.read(name))} bytes")
            entrypoints = next(
                name for name in before.namelist() if name.endswith("/entry_points.txt")
            )
            assert before.read(entrypoints) == after.read(entrypoints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
