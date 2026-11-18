"""Publish one main commit, resuming its tag and staged distribution files."""

import argparse
import hashlib
import json
import os
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

WINDOWS_BUILDER = "scripts/build_windows_bundle.py"


def command(args: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def distribution_names(version: str) -> tuple[str, str]:
    stem = f"free_claude_code-{version}"
    return f"{stem}-py3-none-any.whl", f"{stem}.tar.gz"


def verify_asset(file: Path, asset: dict) -> None:
    with file.open("rb") as stream:
        digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    if (
        asset["state"] != "uploaded"
        or asset["size"] != file.stat().st_size
        or asset.get("digest") != digest
    ):
        raise ValueError(f"Staged asset failed verification: {file.name}")


class Publisher:
    def __init__(self, repository: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Expected owner/repository")
        self.repository = repository

    def gh(self, *args: str) -> str:
        return command(["gh", "release", *args, "--repo", self.repository])

    def release_api(self, suffix: str, *args: str) -> dict:
        return json.loads(
            command(["gh", "api", f"repos/{self.repository}/releases{suffix}", *args])
        )

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

    def publish(
        self, target: str, *, stage_only: bool = False
    ) -> tuple[str, bool] | None:
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
            notes = self.release_api(
                "/generate-notes",
                "-f",
                f"tag_name={tag}",
                "-f",
                f"previous_tag_name={previous}",
            )
            release = self.release_api(
                "",
                "-f",
                f"tag_name={tag}",
                "-f",
                f"name={tag}",
                "-f",
                f"body={notes['body']}",
                "-F",
                "draft=true",
            )
        print(f"Preparing {tag} from {target}")
        with tempfile.TemporaryDirectory(prefix="fcc-release-") as temporary:
            root = Path(temporary).resolve()
            source, dist = root / "source", root / "dist"
            dist.mkdir()
            command(["git", "worktree", "add", "--detach", str(source), target])
            try:
                if command(["git", "status", "--porcelain"], cwd=source).strip():
                    raise ValueError("Release source must be clean")
                windows_bundle_needed = self.stage(
                    tag,
                    release["id"],
                    version,
                    source,
                    dist,
                    verify_windows=not stage_only,
                )
                if stage_only:
                    return version, windows_bundle_needed
                if windows_bundle_needed:
                    raise ValueError(f"The signed Windows bundle for {tag} is missing")
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
        return None

    def validate(self, source: Path, dist: Path, version: str) -> None:
        command(
            [
                sys.executable,
                str(source / "scripts/validate_release.py"),
                str(dist),
                "--version",
                version,
            ],
            cwd=source,
        )

    def stage(
        self,
        tag: str,
        release_id: int,
        version: str,
        source: Path,
        dist: Path,
        *,
        verify_windows: bool,
    ) -> bool:
        names = set(distribution_names(version))
        windows_name = (
            f"free_claude_code-{version}-windows-x64.zip"
            if (source / WINDOWS_BUILDER).is_file()
            else None
        )
        assets = self.release_api(f"/{release_id}")["assets"]
        asset_names = {asset["name"] for asset in assets}
        if len(asset_names) != len(assets) or asset_names - names - {windows_name}:
            raise ValueError(f"Unexpected assets on {tag}")
        windows_asset = next(
            (asset for asset in assets if asset["name"] == windows_name), None
        )
        assets = [asset for asset in assets if asset["name"] in names]
        complete = {asset["name"] for asset in assets} == names and all(
            asset["state"] == "uploaded" for asset in assets
        )
        if complete:
            self.gh(
                "download",
                tag,
                "--dir",
                str(dist),
                *(part for name in sorted(names) for part in ("--pattern", name)),
            )
        else:
            if windows_asset is not None:
                raise ValueError(
                    f"Cannot replace Python assets used by the {tag} bundle"
                )
            for asset in assets:
                self.gh("delete-asset", tag, asset["name"], "--yes")
            builder = source / "scripts/build_macos_launcher.py"
            # Historical releases predate the native templates and retain their format.
            if builder.is_file():
                command([sys.executable, str(builder)], cwd=source)
            command(["uv", "build", "--no-sources", "--out-dir", str(dist)], cwd=source)
            self.validate(source, dist, version)
            self.gh("upload", tag, *(str(dist / name) for name in sorted(names)))
            assets = self.release_api(f"/{release_id}")["assets"]
        if {asset["name"] for asset in assets} != names or len(assets) != 2:
            raise ValueError(f"Incomplete staged distributions for {tag}")
        for asset in assets:
            verify_asset(dist / asset["name"], asset)
        self.validate(source, dist, version)
        if windows_asset is None or windows_asset["state"] != "uploaded":
            return windows_name is not None
        if verify_windows:
            bundle_name = windows_asset["name"]
            self.gh("download", tag, "--dir", str(dist), "--pattern", bundle_name)
            bundle = dist / bundle_name
            verify_asset(bundle, windows_asset)
            command(
                [
                    "gh",
                    "attestation",
                    "verify",
                    str(bundle),
                    "--signer-workflow",
                    f"{self.repository}/.github/workflows/post-merge.yml",
                    "--source-ref",
                    "refs/heads/main",
                    "--deny-self-hosted-runners",
                    "--repo",
                    self.repository,
                ]
            )
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    try:
        result = Publisher(args.repository).publish(
            args.commit, stage_only=args.stage_only
        )
        if result is not None and (output := os.environ.get("GITHUB_OUTPUT")):
            version, windows_needed = result
            with Path(output).open("a", encoding="utf-8") as file:
                file.write(
                    f"version={version}\nwindows_needed={str(windows_needed).lower()}\n"
                )
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Release failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stderr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
