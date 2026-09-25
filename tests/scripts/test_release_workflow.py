"""Workflow events, checkout identity and publishing privileges are contracts."""

from pathlib import Path

import yaml

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
    assert 'git fetch --tags "$CI_SERVER/$CI_REPOSITORY.git"' in fetch["run"]
