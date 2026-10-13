"""Workflow events, checkout identity and publishing privileges are contracts."""

import shlex
from pathlib import Path

import pytest
import yaml

from tests.scripts.test_version_policy import commit, git, write

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return yaml.safe_load(
        (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
    )


def test_post_merge_serializes_and_limits_publish_permissions():
    config = load("post-merge.yml")
    assert config["concurrency"] == {
        "group": "post-merge-main",
        "cancel-in-progress": False,
        "queue": "max",
    }
    assert len(config["jobs"]) == 2
    cache = config["jobs"]["warm-dependency-cache"]
    assert {e["os"] for e in cache["strategy"]["matrix"]["include"]} == {
        "Linux",
        "Windows",
        "macOS",
    }
    publish = config["jobs"]["publish"]
    assert "needs" not in publish
    assert publish["environment"] == "pypi"
    assert publish["permissions"] == {"contents": "write", "id-token": "write"}
    assert cache["permissions"] == {"contents": "read"}
    for job in (cache, publish):
        assert job["if"] == "github.ref == 'refs/heads/main'"
        checkout = next(
            s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@")
        )
        assert checkout["with"]["ref"] == "${{ github.sha }}"
        assert checkout["with"]["fetch-depth"] == 0
    step = next(s for s in publish["steps"] if s.get("name") == "Publish release")
    assert step["env"]["RELEASE_COMMIT"] == "${{ inputs.release_commit || github.sha }}"
    assert "-m scripts.publish_release" in step["run"]


def test_title_edits_only_trigger_the_separate_required_check():
    policy = load("version-policy.yml")
    tests = load("tests.yml")
    assert set(policy[True]) == {"pull_request"}
    assert set(policy[True]["pull_request"]["types"]) == {
        "opened",
        "synchronize",
        "reopened",
        "edited",
        "ready_for_review",
    }
    assert tests[True]["pull_request"] is None
    assert policy["concurrency"]["group"] != tests["concurrency"]["group"]
    assert "version-policy" not in tests["jobs"]
    job = policy["jobs"]["version-policy"]
    assert job["name"] == "Version policy"
    assert job["permissions"] == {"contents": "read", "pull-requests": "read"}
    checkout = next(
        s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha }}"
    assert checkout["with"]["fetch-depth"] == 0
    current = next(
        s for s in job["steps"] if s.get("name") == "Read current pull request"
    )
    assert 'gh api "repos/$PR_REPOSITORY/pulls/$PR_NUMBER"' in current["run"]
    check = next(s for s in job["steps"] if s.get("name") == "Check version policy")
    assert "--pr-json" in check["run"]
    assert "title" not in check["env"]


def test_all_project_setup_jobs_have_history_and_fetch_canonical_tags():
    for filename in ("tests.yml", "version-policy.yml", "post-merge.yml"):
        for job in load(filename)["jobs"].values():
            if any(
                s.get("uses") == "./.github/actions/ci-environment"
                for s in job["steps"]
            ):
                checkout = next(
                    s
                    for s in job["steps"]
                    if s.get("uses", "").startswith("actions/checkout@")
                )
                assert checkout["with"]["fetch-depth"] == 0
    setup = yaml.safe_load(
        (ROOT / ".github/actions/ci-environment/action.yml").read_text()
    )
    fetch = setup["runs"]["steps"][0]
    assert fetch["env"]["CI_REPOSITORY"] == "${{ github.repository }}"


@pytest.mark.parametrize("conflicting_tag", [False, True])
def test_ci_tag_fetch_uses_only_canonical_tags(tmp_path, conflicting_tag):
    upstream = tmp_path / "upstream.git"
    upstream.mkdir()
    git(upstream, "init", "-b", "main")
    write(upstream, "file", "release")
    released = commit(upstream)
    git(upstream, "tag", "-a", "v1.2.3", "-m", "release")
    checkout = tmp_path / "fork"
    git(tmp_path, "clone", str(upstream), str(checkout))
    write(checkout, "file", "pull request")
    head = commit(checkout)
    git(checkout, "tag", "v99.0.0")
    if conflicting_tag:
        git(checkout, "tag", "-f", "v1.2.3")
    write(upstream, "file", "new release")
    latest = commit(upstream)
    git(upstream, "tag", "v1.2.4")

    setup = yaml.safe_load(
        (ROOT / ".github/actions/ci-environment/action.yml").read_text()
    )
    command = shlex.split(setup["runs"]["steps"][0]["run"])
    command = [
        arg.replace("$CI_SERVER/$CI_REPOSITORY.git", upstream.as_posix())
        for arg in command
    ]
    assert command[0] == "git"
    for _ in range(2):
        git(checkout, *command[1:])
        assert git(checkout, "tag", "--list").splitlines() == ["v1.2.3", "v1.2.4"]
        assert git(checkout, "rev-parse", "v1.2.3^{}") == released
        assert git(checkout, "rev-parse", "v1.2.4") == latest
        assert git(checkout, "rev-parse", "HEAD") == head
        assert git(checkout, "status", "--porcelain") == ""
