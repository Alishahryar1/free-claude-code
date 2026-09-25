"""Publish one main commit, resuming its tag and staged distribution files."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .check_version_policy import (
    VERSION,
    changed_paths,
    git,
    next_version,
    release_kind,
    release_paths,
)


def command(args: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def distribution_names(version: str) -> tuple[str, str]:
    stem = f"free_claude_code-{version}"
    return f"{stem}-py3-none-any.whl", f"{stem}.tar.gz"


class Publisher:
    def __init__(self, repository: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Expected owner/repository")
        self.repository = repository
        self.validator = Path(__file__).with_name("validate_release.py").resolve()

    def gh(self, *args: str) -> str:
        return command(["gh", "release", *args, "--repo", self.repository])

    def releases(self) -> dict:
        pages = json.loads(
            command(
                [
                    "gh",
                    "api",
                    "--paginate",
                    "--slurp",
                    f"repos/{self.repository}/releases?per_page=100",
                ]
            )
        )
        return {release["tag_name"]: release for page in pages for release in page}

    def publish(self, target: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", target):
            raise ValueError("Release target must be a full commit SHA")
        command(["git", "fetch", "--tags", "origin", "main"])
        history = git("rev-list", "--first-parent", "origin/main").splitlines()
        if target not in history:
            raise ValueError("Release target is not on main's first-parent history")
        title = git("show", "-s", "--format=%s", target)
        kind = release_kind(title, changed_paths(f"{target}^", target))
        if kind is None:
            print(f"No release for {target}")
            return

        tags = {}
        # Peeled annotated-tag entries follow their unpeeled entries.
        for line in git("show-ref", "--tags", "--dereference").splitlines():
            sha, ref = line.split()
            name = ref.removeprefix("refs/tags/").removesuffix("^{}")
            if name.startswith("v") and VERSION.fullmatch(name[1:]):
                tags[name] = sha
        previous = None
        for sha in history[history.index(target) + 1 :]:
            names = [name for name, commit in tags.items() if commit == sha]
            if names:
                if len(names) != 1:
                    raise ValueError(f"Ambiguous release tags at {sha}")
                previous = names[0]
                break
            if release_paths(changed_paths(f"{sha}^", sha)):
                raise ValueError(
                    f"Earlier release {sha} is unfinished; retry its post-merge run"
                )
        if previous is None:
            raise ValueError("No baseline release tag found")
        releases = self.releases()
        predecessor = releases.get(previous)
        if not predecessor or predecessor["draft"] or predecessor["prerelease"]:
            raise ValueError(
                f"Earlier release {previous} is unfinished; retry its post-merge run"
            )
        version = next_version(previous[1:], kind)
        tag = f"v{version}"
        if any(sha == target and name != tag for name, sha in tags.items()):
            raise ValueError("Target already has another version")
        if tag in tags and tags[tag] != target:
            raise ValueError(f"Tag {tag} belongs to a different commit")
        release = releases.get(tag)
        if release and tag not in tags:
            raise ValueError(f"Release {tag} has no matching remote tag")
        if release and release["prerelease"]:
            raise ValueError(f"Release {tag} unexpectedly is a prerelease")
        if release and not release["draft"]:
            print(f"{tag} already published")
            return
        if tag not in tags:
            command(["git", "tag", tag, target])
            command(["git", "push", "origin", f"refs/tags/{tag}"])
        if release is None:
            self.gh(
                "create",
                tag,
                "--draft",
                "--verify-tag",
                "--title",
                tag,
                "--generate-notes",
                "--notes-start-tag",
                previous,
            )
        print(f"Publishing {tag} from {target}")
        with tempfile.TemporaryDirectory(prefix="fcc-release-") as temporary:
            root = Path(temporary).resolve()
            source, dist = root / "source", root / "dist"
            dist.mkdir()
            command(["git", "worktree", "add", "--detach", str(source), target])
            try:
                if command(["git", "status", "--porcelain"], cwd=source).strip():
                    raise ValueError("Release source must be clean")
                self.stage(tag, version, source, dist)
                command(
                    [
                        "uv",
                        "publish",
                        "--trusted-publishing",
                        "always",
                        *(str(dist / name) for name in distribution_names(version)),
                    ]
                )
                self.gh("edit", tag, "--draft=false", "--latest", "--verify-tag")
            finally:
                if not source.is_relative_to(root):
                    raise ValueError("Release worktree escaped its temporary directory")
                command(["git", "worktree", "remove", "--force", str(source)])

    def validate(self, source: Path, dist: Path, version: str) -> None:
        command(
            [sys.executable, str(self.validator), str(dist), "--version", version],
            cwd=source,
        )

    def stage(self, tag: str, version: str, source: Path, dist: Path) -> None:
        names = set(distribution_names(version))
        assets = self.releases()[tag]["assets"]
        if any(asset["name"] not in names for asset in assets):
            raise ValueError(f"Unexpected assets on {tag}")
        complete = (
            {asset["name"] for asset in assets} == names
            and len(assets) == 2
            and all(asset["state"] == "uploaded" for asset in assets)
        )
        if complete:
            self.gh("download", tag, "--dir", str(dist))
        else:
            # Publication never begins until both uploaded assets are verified.
            for asset in assets:
                self.gh("delete-asset", tag, asset["name"], "--yes")
            command(["uv", "build", "--no-sources", "--out-dir", str(dist)], cwd=source)
            self.validate(source, dist, version)
            self.gh("upload", tag, *(str(dist / name) for name in sorted(names)))
            assets = self.releases()[tag]["assets"]
        if {asset["name"] for asset in assets} != names or len(assets) != 2:
            raise ValueError(f"Incomplete staged distributions for {tag}")
        for asset in assets:
            data = (dist / asset["name"]).read_bytes()
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            if (
                asset["state"] != "uploaded"
                or asset["size"] != len(data)
                or asset.get("digest") != digest
            ):
                raise ValueError(f"Staged asset failed verification: {asset['name']}")
        self.validate(source, dist, version)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        Publisher(args.repository).publish(args.commit)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Release failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stderr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
