"""Require one version increment exactly when a PR changes release inputs."""

import argparse
import subprocess
import tomllib

import tomlkit
from packaging.version import Version

RELEASE_FILES = {".python-version", "pyproject.toml", "uv.lock"}
RELEASE_DIRS = ("assets/", "scripts/", "src/")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def blob(revision: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout if result.returncode == 0 else None


def root_package(document: dict) -> dict:
    entries = [
        entry
        for entry in document.get("package", [])
        if entry.get("name") == "free-claude-code"
        and entry.get("source") == {"editable": "."}
    ]
    if len(entries) != 1:
        raise ValueError("uv.lock must contain exactly one editable FCC package")
    return entries[0]


def version_at(revision: str) -> str:
    project = tomllib.loads(blob(revision, "pyproject.toml") or "")["project"]
    if project["name"] != "free-claude-code":
        raise ValueError("Project name must be free-claude-code")
    value = project["version"]
    if not isinstance(value, str):
        raise ValueError("Project version must be a string")
    parsed = Version(value)
    if value != f"{parsed.major}.{parsed.minor}.{parsed.micro}":
        raise ValueError("Project version must use the format MAJOR.MINOR.PATCH")
    lock = root_package(tomllib.loads(blob(revision, "uv.lock") or ""))
    if lock["version"] != value:
        raise ValueError("pyproject.toml and uv.lock versions must agree")
    return value


def without_release_version(text: str | None, path: str) -> str | None:
    if text is None:
        return None
    document = tomlkit.parse(text)
    if path == "pyproject.toml":
        document["project"]["version"] = "FCC_VERSION"
    else:
        for package in document["package"]:
            if package["name"] == "free-claude-code" and package.get("source") == {
                "editable": "."
            }:
                package["version"] = "FCC_VERSION"
    return tomlkit.dumps(document)


def check(base: str, head: str) -> None:
    base = git("rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}")
    head = git("rev-parse", "--verify", "--end-of-options", f"{head}^{{commit}}")
    ancestor = git("merge-base", base, head)
    old, new = version_at(base), version_at(head)
    changed = git("diff", "--no-renames", "--name-only", "-z", ancestor, head).split(
        "\0"
    )
    release_paths = []
    for path in changed:
        if path not in RELEASE_FILES and not path.startswith(RELEASE_DIRS):
            continue
        if path in {"pyproject.toml", "uv.lock"} and without_release_version(
            blob(ancestor, path), path
        ) == without_release_version(blob(head, path), path):
            continue
        release_paths.append(path)
    print(f"FCC version: {old} -> {new}")
    print("Release changes: " + (", ".join(release_paths) or "none"))
    previous = Version(old)
    allowed = {
        f"{previous.major}.{previous.minor}.{previous.micro + 1}",
        f"{previous.major}.{previous.minor + 1}.0",
        f"{previous.major + 1}.0.0",
    }
    if release_paths and new not in allowed:
        raise ValueError(
            "Version must increase by exactly one patch, minor, or major increment "
            "from the target branch, resetting lower components to zero. "
            f"Allowed versions: {', '.join(sorted(allowed))}"
        )
    if not release_paths and new != old:
        raise ValueError("Version must stay unchanged without release changes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()
    try:
        check(args.base, args.head)
    except (ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"Version policy failed: {error}")
        return 1
    print("Version policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
