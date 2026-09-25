"""Check the exact FCC distributions that will be published."""

import argparse
import configparser
import re
import subprocess
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path


def require_files(names: set[str], expected: set[str]) -> None:
    missing = expected - names
    if missing:
        raise ValueError("Missing release files: " + ", ".join(sorted(missing)))


def validate(directory: Path, version: str) -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    name = project["name"]
    if not re.fullmatch(
        r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", version
    ):
        raise ValueError("Expected a stable MAJOR.MINOR.PATCH release version")
    if "version" in project or "version" not in project.get("dynamic", []):
        raise ValueError("Release source must use dynamic version metadata")
    stem = f"{name.replace('-', '_')}-{version}"
    wheel_path = directory / f"{stem}-py3-none-any.whl"
    sdist_path = directory / f"{stem}.tar.gz"
    # uv build creates a .gitignore alongside its distributions.
    artifacts = {path for path in directory.iterdir() if path.name != ".gitignore"}
    if artifacts != {wheel_path, sdist_path}:
        raise ValueError(
            "Release directory must contain only the current wheel and sdist"
        )
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "src/free_claude_code"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    sources = set(filter(None, tracked.split("\0")))
    if "src/free_claude_code/__init__.py" not in sources:
        raise ValueError("Missing tracked FCC package sources")
    with zipfile.ZipFile(wheel_path) as archive:
        wheel = {
            file: archive.read(file)
            for file in archive.namelist()
            if not file.endswith("/")
        }
    with tarfile.open(sdist_path) as archive:
        source = {}
        for file in archive:
            stream = archive.extractfile(file) if file.isfile() else None
            if stream is not None:
                source[file.name] = stream.read()
    wheel_info = f"{stem}.dist-info"
    updater_names = {"fcc-update", "fcc-update.cmd"}
    require_files(
        set(wheel),
        {file.removeprefix("src/") for file in sources}
        | {f"{stem}.data/scripts/{script}" for script in updater_names}
        | {f"{wheel_info}/METADATA", f"{wheel_info}/entry_points.txt"},
    )
    require_files(
        set(source),
        {f"{stem}/{file}" for file in sources}
        | {f"{stem}/scripts/update/{script}" for script in updater_names}
        | {
            f"{stem}/{file}"
            for file in ("pyproject.toml", "README.md", "LICENSE", "PKG-INFO")
        },
    )
    for data in (wheel[f"{wheel_info}/METADATA"], source[f"{stem}/PKG-INFO"]):
        metadata = BytesParser().parsebytes(data)
        if metadata["Name"] != name or metadata["Version"] != version:
            raise ValueError(
                "Release metadata does not match the project name and version"
            )
    entrypoints = configparser.ConfigParser()
    entrypoints.read_string(wheel[f"{wheel_info}/entry_points.txt"].decode())
    for project_key, section in (
        ("scripts", "console_scripts"),
        ("gui-scripts", "gui_scripts"),
    ):
        actual = dict(entrypoints[section]) if section in entrypoints else {}
        if actual != project.get(project_key, {}):
            raise ValueError(f"Release entry points do not match {project_key}")
    print(f"Validated {wheel_path.name} and {sdist_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        validate(args.directory, args.version)
    except (
        ValueError,
        KeyError,
        OSError,
        zipfile.BadZipFile,
        tarfile.TarError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Release validation failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
