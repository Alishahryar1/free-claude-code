"""Release decisions and workflow permissions are explicit contracts."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.scripts.test_version_policy import commit, git, version, write

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/post-merge.yml"


def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_post_merge_serializes_runs_and_separates_publish_permissions():
    config = workflow()
    assert config["concurrency"] == {
        "group": "post-merge-main",
        "cancel-in-progress": False,
        "queue": "max",
    }
    assert len(config["jobs"]) == 2
    cache = config["jobs"]["warm-dependency-cache"]
    assert {entry["os"] for entry in cache["strategy"]["matrix"]["include"]} == {
        "Linux",
        "Windows",
        "macOS",
    }
    publish = config["jobs"]["publish"]
    assert "needs" not in publish
    assert publish["environment"] == "pypi"
    assert publish["permissions"] == {"contents": "read", "id-token": "write"}
    assert cache["permissions"] == {"contents": "read"}
    for job in (cache, publish):
        assert job["if"] == "github.ref == 'refs/heads/main'"
        checkout = next(
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        )
        assert checkout["with"]["ref"] == "${{ github.sha }}"
    for name in ("Build and validate release", "Publish release"):
        step = next(step for step in publish["steps"] if step.get("name") == name)
        assert step["if"] == "steps.release.outputs.changed == 'true'"


@pytest.mark.parametrize("bump", [False, True])
@pytest.mark.parametrize("manual", [False, True])
def test_release_detection_uses_the_triggering_revision(tmp_path, bump, manual):
    git(tmp_path, "init", "-b", "main")
    version(tmp_path, "1.2.3")
    before = commit(tmp_path)
    version(tmp_path, "1.2.4" if bump else "1.2.3")
    write(tmp_path, "README.md", "changed\n")
    head = commit(tmp_path)
    # A later checkout/commit must not change the queued run's release decision.
    version(tmp_path, "9.0.0")
    commit(tmp_path)
    steps = workflow()["jobs"]["publish"]["steps"]
    run = next(step["run"] for step in steps if step.get("id") == "release")
    source = run.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    output = tmp_path / "outputs"
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=os.environ
        | {
            "BEFORE_SHA": "" if manual else before,
            "HEAD_SHA": head,
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text().strip() == f"changed={str(bump).lower()}"


def test_version_check_receives_pr_base_and_head_without_publish_permission():
    config = yaml.safe_load(
        (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    )
    job = config["jobs"]["version-policy"]
    assert job["name"] == "Version policy"
    assert job["permissions"] == {"contents": "read"}
    step = next(
        step for step in job["steps"] if step.get("name") == "Check version policy"
    )
    assert step["env"] == {
        "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
    }
    assert '--base "$BASE_SHA" --head "$HEAD_SHA"' in step["run"]
