"""Release ordering and interrupted publication use real Git history."""

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import TypedDict

import pytest

from scripts import publish_release as publishing
from scripts.check_version_policy import next_version
from tests.scripts.test_version_policy import commit, git, write


@pytest.mark.parametrize(
    "kind,expected", [("patch", "6.2.67"), ("minor", "6.3.0"), ("major", "7.0.0")]
)
def test_one_increment(kind, expected):
    assert next_version("6.2.66", kind) == expected


@pytest.fixture
def releases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    git(tmp_path, "init", "-b", "main")
    write(tmp_path, "src/file.py", "base")
    base = commit(tmp_path)
    git(tmp_path, "tag", "v1.2.3")
    git(tmp_path, "update-ref", "refs/remotes/origin/main", base)
    state = FakeRemote(tmp_path)
    monkeypatch.setattr(publishing, "command", state.command)
    return tmp_path, state


class Asset(TypedDict):
    name: str
    state: str
    size: int
    digest: str


class Release(TypedDict):
    id: int
    tag_name: str
    draft: bool
    prerelease: bool
    assets: list[Asset]


class FakeRemote:
    def __init__(self, repo):
        self.repo = repo
        self.releases: dict[str, Release] = {
            "v1.2.3": {
                "id": 1,
                "tag_name": "v1.2.3",
                "draft": False,
                "prerelease": False,
                "assets": [],
            }
        }
        self.files = {}
        self.calls = []
        self.fail = None
        self.builds = 0
        self.uploaded = {}
        self.real_command = publishing.command

    def command(self, args, *, cwd=None):
        if "--repo" in args:
            args = args[: args.index("--repo")]
        self.calls.append(args)
        if args[0] == "git":
            if args[1] in {"fetch", "push"}:
                return ""
            return git(Path(cwd or self.repo), *args[1:])
        if args[:2] == ["gh", "api"]:
            endpoint = next(arg for arg in args if arg.startswith("repos/"))
            fields = dict(
                field.split("=", 1)
                for flag, field in pairwise(args)
                if flag in {"-f", "-F"}
            )
            if endpoint.endswith("?per_page=100"):
                return json.dumps([list(self.releases.values())])
            if endpoint.endswith("/generate-notes"):
                assert fields["previous_tag_name"] in self.releases
                return json.dumps({"body": "Generated release notes"})
            if endpoint.endswith("/releases"):
                assert fields["draft"] == "true"
                assert fields["body"] == "Generated release notes"
                tag = fields["tag_name"]
                self.releases[tag] = {
                    "id": len(self.releases) + 1,
                    "tag_name": tag,
                    "draft": True,
                    "prerelease": False,
                    "assets": [],
                }
                return json.dumps(self.releases[tag])
            release_id = int(endpoint.rsplit("/", 1)[1])
            return json.dumps(
                next(r for r in self.releases.values() if r["id"] == release_id)
            )
        if args[:3] == ["gh", "release", "upload"]:
            tag = args[3]
            for filename in args[4:]:
                file = Path(filename)
                data = file.read_bytes()
                self.files[tag, file.name] = data
                self.releases[tag]["assets"].append(
                    {
                        "name": file.name,
                        "state": "uploaded",
                        "size": len(data),
                        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
                    }
                )
                if self.fail == "stage":
                    self.fail = None
                    raise RuntimeError("interrupted staging")
        elif args[:3] == ["gh", "release", "delete-asset"]:
            tag, name = args[3:5]
            self.releases[tag]["assets"] = [
                a for a in self.releases[tag]["assets"] if a["name"] != name
            ]
            self.files.pop((tag, name), None)
        elif args[:3] == ["gh", "release", "download"]:
            tag = args[3]
            dest = Path(args[args.index("--dir") + 1])
            for asset in self.releases[tag]["assets"]:
                (dest / asset["name"]).write_bytes(self.files[tag, asset["name"]])
        elif args[:3] == ["gh", "release", "edit"]:
            if self.fail == "finalize":
                self.fail = None
                raise RuntimeError("interrupted finalization")
            self.releases[args[3]]["draft"] = False
        elif args[:2] == ["uv", "build"]:
            self.builds += 1
            assert cwd is not None
            version = git(Path(cwd), "describe", "--tags", "--exact-match")[1:]
            out = Path(args[args.index("--out-dir") + 1])
            for name in publishing.distribution_names(version):
                (out / name).write_bytes(f"build {self.builds}: {name}".encode())
        elif args[:2] == ["uv", "publish"]:
            for file in map(Path, args[4:]):
                data = file.read_bytes()
                if file.name in self.uploaded:
                    assert self.uploaded[file.name] == data
                self.uploaded[file.name] = data
                if self.fail == "publish":
                    self.fail = None
                    raise RuntimeError("interrupted publication")
        elif args[0] == publishing.sys.executable:
            pass  # Archive validation is exercised separately using real archives.
        else:
            raise AssertionError(args)
        return ""

    def advance(self, title="patch: Change", path="src/file.py"):
        write(self.repo, path, title)
        head = commit(self.repo, title)
        git(self.repo, "update-ref", "refs/remotes/origin/main", head)
        return head

    def publisher(self):
        return publishing.Publisher("owner/repo")


def test_nonrelease_never_allocates_or_builds(releases):
    _, remote = releases
    head = remote.advance("Explain usage", "README.md")
    remote.publisher().publish(head)
    assert remote.builds == 0
    assert not any(c[:2] == ["git", "push"] for c in remote.calls)


@pytest.mark.parametrize("existing_draft", [False, True])
def test_stale_release_list_does_not_block_staging(
    releases, monkeypatch, existing_draft
):
    _, remote = releases
    head = remote.advance()
    if existing_draft:
        remote.fail = "stage"
        with pytest.raises(RuntimeError, match="interrupted staging"):
            remote.publisher().publish(head)
    listed = json.dumps([list(remote.releases.values())])

    def stale_listing(args, *, cwd=None):
        if args[:2] == ["gh", "api"] and any(
            arg.endswith("/releases?per_page=100") for arg in args
        ):
            return listed
        return remote.command(args, cwd=cwd)

    monkeypatch.setattr(publishing, "command", stale_listing)
    remote.publisher().publish(head)
    assert not remote.releases["v1.2.4"]["draft"]
    assert set(remote.uploaded) == set(publishing.distribution_names("1.2.4"))


@pytest.mark.parametrize("failure", ["stage", "publish", "finalize"])
def test_retries_keep_version_and_published_bytes(releases, failure):
    repo, remote = releases
    head = remote.advance()
    remote.fail = failure
    with pytest.raises(RuntimeError):
        remote.publisher().publish(head)
    first_uploaded = dict(remote.uploaded)
    assert remote.releases["v1.2.4"]["draft"]
    remote.publisher().publish(head)
    assert not remote.releases["v1.2.4"]["draft"]
    assert git(repo, "rev-parse", "v1.2.4") == head
    assert remote.builds == (2 if failure == "stage" else 1)
    assert all(remote.uploaded[name] == value for name, value in first_uploaded.items())
    builds = remote.builds
    remote.publisher().publish(head)
    assert remote.builds == builds


def test_later_commit_waits_even_if_previous_failed_before_tag(releases):
    _, remote = releases
    first = remote.advance()
    second = remote.advance("minor: Second")
    with pytest.raises(ValueError, match=first):
        remote.publisher().publish(second)
    remote.publisher().publish(first)
    remote.publisher().publish(second)
    assert not remote.releases["v1.3.0"]["draft"]


def test_later_commit_waits_for_draft_and_old_retry_is_noop(releases):
    _, remote = releases
    first = remote.advance()
    remote.fail = "publish"
    with pytest.raises(RuntimeError):
        remote.publisher().publish(first)
    second = remote.advance("major: Second")
    with pytest.raises(ValueError, match=r"v1\.2\.4"):
        remote.publisher().publish(second)
    remote.publisher().publish(first)
    remote.publisher().publish(second)
    calls = len(remote.calls)
    remote.publisher().publish(first)
    assert not any(c[:3] == ["gh", "release", "edit"] for c in remote.calls[calls:])


def test_tag_collision_and_extra_target_tag_are_rejected(releases):
    repo, remote = releases
    head = remote.advance()
    git(repo, "tag", "v1.2.4", "v1.2.3")
    with pytest.raises(ValueError):
        remote.publisher().publish(head)
    git(repo, "tag", "-d", "v1.2.4")
    git(repo, "tag", "v9.0.0", head)
    with pytest.raises(ValueError, match="another version"):
        remote.publisher().publish(head)


def test_target_must_be_main_ancestor(releases):
    repo, remote = releases
    git(repo, "checkout", "-b", "other")
    write(repo, "src/other.py", "other")
    head = commit(repo, "patch: Other")
    with pytest.raises(ValueError, match="main"):
        remote.publisher().publish(head)


def test_malformed_or_bypassed_title_cannot_publish(releases):
    _, remote = releases
    head = remote.advance("Oops")
    with pytest.raises(ValueError, match="require"):
        remote.publisher().publish(head)


def test_reserved_tag_and_missing_draft_can_resume(releases):
    repo, remote = releases
    head = remote.advance()
    git(repo, "tag", "v1.2.4", head)
    remote.publisher().publish(head)
    assert remote.builds == 1
    assert not remote.releases["v1.2.4"]["draft"]


def test_staged_bytes_are_verified_before_any_retry_upload(releases):
    _, remote = releases
    head = remote.advance()
    remote.fail = "publish"
    with pytest.raises(RuntimeError):
        remote.publisher().publish(head)
    name = publishing.distribution_names("1.2.4")[0]
    remote.files["v1.2.4", name] = b"corrupted"
    before = len(remote.calls)
    with pytest.raises(ValueError, match="verification"):
        remote.publisher().publish(head)
    assert not any(c[:2] == ["uv", "publish"] for c in remote.calls[before:])


def test_no_release_does_not_wait_for_an_unfinished_release(releases):
    _, remote = releases
    first = remote.advance()
    remote.fail = "publish"
    with pytest.raises(RuntimeError):
        remote.publisher().publish(first)
    docs = remote.advance("Explain the release", "README.md")
    before = remote.builds
    remote.publisher().publish(docs)
    assert remote.builds == before
