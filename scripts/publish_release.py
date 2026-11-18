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
BUMP_ORDER = ("patch", "minor", "major")


def command(args: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def distribution_names(version: str) -> tuple[str, str]:
    stem = f"free_claude_code-{version}"
    return f"{stem}-py3-none-any.whl", f"{stem}.tar.gz"


def windows_bundle_name(version: str) -> str:
    return f"free_claude_code-{version}-windows-x64.zip"


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
        self,
        target: str,
        *,
        stage_only: bool = False,
        accepted_windows_digest: str = "",
        supersedes_windows_candidate: str = "",
    ) -> tuple[str, bool, bool] | None:
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
        target_index = history.index(target)
        for sha in history[target_index + 1 :]:
            names = [name for name, commit in tags.items() if commit == sha]
            if names:
                if len(names) != 1:
                    raise ValueError(f"Ambiguous release tags at {sha}")
                previous = names[0]
                break
            paths = changed_paths(f"{sha}^", sha)
            if release_paths(paths):
                if not supersedes_windows_candidate:
                    raise ValueError(
                        f"Earlier release {sha} is unfinished; retry its post-merge run"
                    )
                skipped_kind = release_kind(
                    git("show", "-s", "--format=%s", sha), paths
                )
                if skipped_kind is not None and BUMP_ORDER.index(
                    skipped_kind
                ) > BUMP_ORDER.index(kind):
                    kind = skipped_kind
        if previous is None:
            raise ValueError("No baseline release tag found")
        releases = self.releases()
        predecessor = releases.get(previous)
        superseding = bool(supersedes_windows_candidate)
        if superseding:
            if (
                previous != supersedes_windows_candidate
                or not predecessor
                or not predecessor["draft"]
                or predecessor["prerelease"]
                or not git(
                    "ls-tree", "--name-only", tags[previous], "--", WINDOWS_BUILDER
                ).strip()
                or not git(
                    "ls-tree", "--name-only", target, "--", WINDOWS_BUILDER
                ).strip()
            ):
                raise ValueError(
                    "Only a preceding draft Windows candidate can be superseded"
                )
        elif not predecessor or predecessor["draft"] or predecessor["prerelease"]:
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
        newer_commits = set(history[:target_index])
        if newer_commits.intersection(tags.values()):
            raise ValueError(f"A newer release is already staged; cannot publish {tag}")
        if tag not in tags:
            command(["git", "tag", tag, target])
            command(["git", "push", "origin", f"refs/tags/{tag}"])
        if release is None:
            notes_base = previous
            if superseding:
                published_by_commit = {
                    tags[name]: name
                    for name, item in releases.items()
                    if name in tags and not item["draft"] and not item["prerelease"]
                }
                notes_base = next(
                    (
                        published_by_commit[sha]
                        for sha in history[target_index + 1 :]
                        if sha in published_by_commit
                    ),
                    None,
                )
                if notes_base is None:
                    raise ValueError(
                        "No published release for the corrected candidate's notes"
                    )
            notes = self.release_api(
                "/generate-notes",
                "-f",
                f"tag_name={tag}",
                "-f",
                f"previous_tag_name={notes_base}",
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
                bundle_required = (source / WINDOWS_BUILDER).is_file()
                acceptance_required = bundle_required and (
                    superseding
                    or not any(
                        asset["name"] == windows_bundle_name(previous[1:])
                        and asset["state"] == "uploaded"
                        for asset in predecessor["assets"]
                    )
                )
                bundle_digest = self.stage(
                    tag,
                    release["id"],
                    version,
                    source,
                    dist,
                    target=target,
                    bundle_required=bundle_required,
                    verify_windows=not stage_only,
                )
                windows_bundle_needed = bundle_required and bundle_digest is None
                if stage_only:
                    return version, windows_bundle_needed, acceptance_required
                if windows_bundle_needed:
                    raise ValueError(f"The signed Windows bundle for {tag} is missing")
                if acceptance_required and bundle_digest != accepted_windows_digest:
                    raise ValueError(
                        f"The first signed Windows bundle for {tag} requires acceptance of {bundle_digest}"
                    )
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
        target: str,
        bundle_required: bool,
        verify_windows: bool,
    ) -> str | None:
        names = set(distribution_names(version))
        windows_name = windows_bundle_name(version) if bundle_required else None
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
            return None
        if verify_windows:
            bundle_name = windows_asset["name"]
            self.gh("download", tag, "--dir", str(dist), "--pattern", bundle_name)
            bundle = dist / bundle_name
            verify_asset(bundle, windows_asset)
            wheel_name = distribution_names(version)[0]
            wheel_digest = next(
                asset["digest"] for asset in assets if asset["name"] == wheel_name
            )
            self.verify_windows_attestation(bundle, version, target, wheel_digest)
        return windows_asset["digest"]

    def verify_windows_attestation(
        self, bundle: Path, version: str, target: str, wheel_digest: str
    ) -> None:
        results = json.loads(
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
                    "--predicate-type",
                    f"https://github.com/{self.repository}/attestations/windows-release/v1",
                    "--format",
                    "json",
                    "--repo",
                    self.repository,
                ]
            )
        )
        expected = {"version": version, "commit": target, "wheel_digest": wheel_digest}
        if not any(
            result["verificationResult"]["statement"]["predicate"] == expected
            for result in results
        ):
            raise ValueError(
                f"Windows bundle provenance does not match release v{version}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--accepted-windows-digest", default="")
    parser.add_argument("--supersedes-windows-candidate", default="")
    args = parser.parse_args()
    try:
        result = Publisher(args.repository).publish(
            args.commit,
            stage_only=args.stage_only,
            accepted_windows_digest=args.accepted_windows_digest,
            supersedes_windows_candidate=args.supersedes_windows_candidate,
        )
        if result is not None and (output := os.environ.get("GITHUB_OUTPUT")):
            version, windows_needed, acceptance_required = result
            with Path(output).open("a", encoding="utf-8") as file:
                file.write(
                    f"version={version}\nwindows_needed={str(windows_needed).lower()}\n"
                    f"windows_acceptance_required={str(acceptance_required).lower()}\n"
                )
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Release failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stderr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
