"""Test sharding and required summaries are wired into the CI workflow."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.ci_shards import shard_count

ROOT = Path(__file__).resolve().parents[2]


def workflow():
    return yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())


@pytest.mark.parametrize("suite", ["pytest", "playwright"])
def test_shard_jobs_use_generated_matrices_and_existing_parallelism(suite):
    config = workflow()
    shard_count(config["env"][f"CI_{suite.upper()}_SHARDS"])
    prepare = config["jobs"]["prepare-shards"]
    assert f"{suite}_matrix" in prepare["outputs"]
    generate = prepare["steps"][-1]
    assert f'--{suite}-shards "$CI_{suite.upper()}_SHARDS"' in generate["run"]
    job = config["jobs"][f"{suite}-shards"]
    assert job["needs"] == "prepare-shards"
    assert job["strategy"]["matrix"] == (
        "${{ fromJSON(needs.prepare-shards.outputs." + suite + "_matrix) }}"
    )
    assert job["strategy"]["fail-fast"] is False
    assert "max-parallel" not in job["strategy"]
    assert "continue-on-error" not in job
    assert job["runs-on"] == "${{ matrix.runner }}"
    assert "${{ matrix.shard }}/${{ matrix.shards }}" in job["name"]
    command = next(
        step["run"]
        for step in job["steps"]
        if step.get("run", "").startswith("uv run --locked pytest ")
    )
    assert "--splits ${{ matrix.shards }}" in command
    assert "--group ${{ matrix.shard }}" in command
    assert "--splitting-algorithm=least_duration" in command
    assert "-n " not in command
    assert "--dist=" not in command
    assert ("pytest e2e " in command) == (suite == "playwright")
    assert "||" not in command


def test_browser_artifacts_identify_each_shard_and_run_attempt():
    job = workflow()["jobs"]["playwright-shards"]
    upload = next(
        step
        for step in job["steps"]
        if "actions/upload-artifact@" in step.get("uses", "")
    )
    name = upload["with"]["name"]
    for field in (
        "runner.os",
        "runner.arch",
        "matrix.shard",
        "matrix.shards",
        "github.run_id",
        "github.run_attempt",
    ):
        assert "${{ " + field + " }}" in name
    assert upload["if"] == "failure()"
    assert upload["with"]["path"] == "test-results/"
    assert upload["with"]["retention-days"] == 7


@pytest.mark.parametrize(
    "suite,preparation,shards,native,rebuild",
    [
        (suite, preparation, shards, "success", "success")
        for suite in ("pytest", "playwright")
        for preparation in ("success", "failure", "cancelled", "skipped")
        for shards in ("success", "failure", "cancelled", "skipped")
    ]
    + [
        ("pytest", "success", "success", native, rebuild)
        for native, rebuild in (
            ("failure", "success"),
            ("cancelled", "success"),
            ("skipped", "success"),
            ("success", "failure"),
            ("success", "cancelled"),
            ("success", "skipped"),
        )
    ],
)
def test_summary_only_passes_when_preparation_and_all_shards_succeed(
    suite, preparation, shards, native, rebuild
):
    job = workflow()["jobs"][f"{suite}-summary"]
    assert job["name"] == suite
    expected_needs = {"prepare-shards", f"{suite}-shards"}
    if suite == "pytest":
        expected_needs |= {"macos-desktop", "sdist-rebuild"}
    assert set(job["needs"]) == expected_needs
    assert job["if"] == "${{ always() }}"
    assert "continue-on-error" not in job
    assert len(job["steps"]) == 1
    step = job["steps"][0]
    expected_env = {
        "PREPARATION_RESULT": "${{ needs.prepare-shards.result }}",
        "SHARDS_RESULT": "${{ needs." + suite + "-shards.result }}",
    }
    if suite == "pytest":
        expected_env |= {
            "NATIVE_RESULT": "${{ needs.macos-desktop.result }}",
            "REBUILD_RESULT": "${{ needs.sdist-rebuild.result }}",
        }
    assert step["env"] == expected_env
    bash = shutil.which("bash")
    if os.name == "nt":
        # WSL's bash.exe does not inherit arbitrary Windows environment variables.
        git = shutil.which("git")
        bash = (
            shutil.which("bash", path=str(Path(git).parents[1] / "bin"))
            if git
            else None
        )
    if bash is None:
        pytest.skip("bash is required to execute the GitHub summary check")
    result = subprocess.run(
        [bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
        env={
            **os.environ,
            "PREPARATION_RESULT": preparation,
            "SHARDS_RESULT": shards,
            "NATIVE_RESULT": native,
            "REBUILD_RESULT": rebuild,
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert (result.returncode == 0) == (
        preparation == shards == native == rebuild == "success"
    ), result.stderr


def test_native_checks_cover_both_macs_and_nonmac_sdist_rebuilds():
    jobs = workflow()["jobs"]
    native = jobs["macos-desktop"]
    assert native["strategy"]["matrix"]["runner"] == ["macos-26", "macos-26-intel"]
    assert any(
        step.get("run") == "sh scripts/ci.sh --only macos-desktop"
        for step in native["steps"]
    )
    rebuild = jobs["sdist-rebuild"]
    assert rebuild["needs"] == "macos-desktop"
    assert rebuild["strategy"]["matrix"]["runner"] == [
        "ubuntu-latest",
        "windows-latest",
    ]
    assert "rebuild_sdist.py" in rebuild["steps"][-1]["run"]
