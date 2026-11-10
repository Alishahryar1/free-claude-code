"""Timing history is optional, but every shard must share its frozen input."""

import io
import json
import sys
import zipfile
from copy import deepcopy

import pytest

from scripts import ci_timings as timing
from scripts.ci_shards import RUNNERS, matrix


@pytest.fixture
def run():
    return timing.Run(
        100,
        1,
        "a" * 40,
        {suite: matrix(dict.fromkeys(RUNNERS, 2)) for suite in timing.SUITES},
    )


def maps():
    return {name: {"test_example.py::test_work": 0.5} for name in timing.MAP_NAMES}


def archive(manifest, values=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as output:
        output.writestr("manifest.json", json.dumps(manifest))
        for name, data in (values or maps()).items():
            output.writestr(name, json.dumps(data))
    return buffer.getvalue()


class FakeGitHub(timing.GitHub):
    def __init__(self, runs=(), artifacts=None, blobs=None):
        self.runs = list(runs)
        self.by_run = artifacts or {}
        self.blobs = blobs or {}
        self.downloaded = []
        self.queried_history = False

    def successful_runs(self):
        self.queried_history = True
        return self.runs

    def artifacts(self, run_id):
        return self.by_run.get(run_id, [])

    def download(self, artifact_id):
        self.downloaded.append(artifact_id)
        return self.blobs[artifact_id]


def prior(run, attempt=1):
    return {
        "id": run.run_id,
        "run_attempt": attempt,
        "status": "completed",
        "conclusion": "success",
        "created_at": f"2026-11-{run.run_id:02d}T00:00:00Z",
    }


def artifact(artifact_id, name, expired=False):
    return {"id": artifact_id, "name": name, "expired": expired}


def test_preparation_uses_latest_complete_source_and_normalizes_zero(run, tmp_path):
    source = timing.Run(2, 1, "b" * 40, run.matrices)
    values = maps()
    values["pytest-Linux.json"]["test_example.py::test_work"] = 0
    github = FakeGitHub(
        runs=[prior(source)],
        artifacts={2: [artifact(20, timing.COMPLETE_PREFIX + "1")]},
        blobs={20: archive(source.manifest("complete"), values)},
    )
    assert timing.prepare(run, tmp_path, github) is None
    manifest, restored = timing.read_bundle(tmp_path)
    assert manifest["source_artifact_id"] == 20
    assert manifest["sha"] == run.sha
    assert restored["pytest-Linux.json"]["test_example.py::test_work"] > 0
    assert restored["pytest-Windows.json"] == values["pytest-Windows.json"]


def test_bad_expired_and_missing_history_fall_back_to_older_source(run, tmp_path):
    source = timing.Run(1, 1, "b" * 40, run.matrices)
    newer = timing.Run(2, 2, "c" * 40, run.matrices)
    github = FakeGitHub(
        runs=[prior(newer, 2), prior(source)],
        artifacts={
            2: [
                artifact(23, timing.COMPLETE_PREFIX + "3", True),
                artifact(22, timing.COMPLETE_PREFIX + "2"),
            ],
            1: [artifact(11, timing.COMPLETE_PREFIX + "1")],
        },
        blobs={22: b"invalid zip", 11: archive(source.manifest("complete"))},
    )
    timing.prepare(run, tmp_path, github)
    assert github.downloaded == [22, 11]
    assert timing.read_bundle(tmp_path)[0]["source_artifact_id"] == 11


@pytest.mark.parametrize("unavailable", [False, True])
def test_cold_start_is_a_valid_frozen_empty_bundle(
    run, tmp_path, unavailable, monkeypatch
):
    github = FakeGitHub()
    if unavailable:

        def failure():
            raise OSError("offline")

        monkeypatch.setattr(github, "successful_runs", failure)
    timing.prepare(run, tmp_path, github)
    manifest, restored = timing.read_bundle(tmp_path)
    assert manifest["source_artifact_id"] is None
    assert all(not values for values in restored.values())


def test_rerun_reuses_original_input_even_when_cold(run, tmp_path):
    original = run.manifest("input") | {"source_artifact_id": None}
    empty = {name: {} for name in timing.MAP_NAMES}
    github = FakeGitHub(
        artifacts={100: [artifact(42, timing.INPUT_NAME)]},
        blobs={42: archive(original, empty)},
    )
    rerun = timing.Run(100, 3, run.sha, run.matrices)
    assert timing.prepare(rerun, tmp_path, github) == 42
    assert not github.queried_history
    assert timing.read_bundle(tmp_path) == (original, empty)


@pytest.mark.parametrize("warm", [False, True])
def test_rerun_rebuilds_missing_input_for_restarted_shards(run, tmp_path, warm):
    source = timing.Run(2, 1, "b" * 40, run.matrices)
    github = FakeGitHub(
        runs=[prior(source)] if warm else [],
        artifacts={2: [artifact(20, timing.COMPLETE_PREFIX + "1")]},
        blobs={20: archive(source.manifest("complete"))},
    )
    rerun = timing.Run(run.run_id, 2, run.sha, run.matrices)
    assert timing.prepare(rerun, tmp_path, github) is None
    manifest, restored = timing.read_bundle(tmp_path)
    assert manifest == rerun.manifest("input") | {
        "source_artifact_id": 20 if warm else None
    }
    assert restored == (maps() if warm else {name: {} for name in timing.MAP_NAMES})
    assert github.queried_history
    rerun.check_input(manifest)


@pytest.mark.parametrize("problem", ["sha", "matrix", "duplicate", "download"])
def test_rerun_rejects_unusable_existing_input(run, tmp_path, problem):
    manifest = run.manifest("input")
    if problem == "sha":
        manifest["sha"] = "b" * 40
    if problem == "matrix":
        manifest["matrices"]["pytest"] = matrix(dict.fromkeys(RUNNERS, 1))
    github = FakeGitHub(
        artifacts={
            100: [artifact(42, timing.INPUT_NAME)]
            + ([artifact(43, timing.INPUT_NAME)] if problem == "duplicate" else [])
        },
        blobs={} if problem == "download" else {42: archive(manifest)},
    )
    with pytest.raises(ValueError, match="fresh CI run"):
        timing.prepare(timing.Run(100, 2, run.sha, run.matrices), tmp_path, github)
    assert not github.queried_history


@pytest.mark.parametrize("value", [-1, True, "1", float("nan"), float("inf"), 10**400])
def test_rejects_invalid_durations(value):
    with pytest.raises(ValueError):
        timing.durations({"test.py::test_one": value})


def test_json_and_archive_validation(run, monkeypatch):
    with pytest.raises(ValueError, match="Duplicate"):
        timing.load_json('{"test": 1, "test": 2}')
    with pytest.raises(ValueError):
        timing.read_archive(b"bad")
    with pytest.raises(ValueError):
        timing.read_archive(archive(run.manifest("complete"), {"../escape.json": {}}))
    monkeypatch.setattr(timing, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="size"):
        timing.read_archive(archive(run.manifest("complete")))


def records(run, baseline=42):
    for (suite, system, shard), count in timing.expected_shards(run.matrices).items():
        yield run.manifest("shard") | {
            "input_artifact_id": baseline,
            "suite": suite,
            "os": system,
            "shard": shard,
            "shards": count,
            "durations": {f"test.py::test_{shard}": float(shard)},
        }


def test_merge_requires_complete_current_attempt_and_disjoint_ids(run):
    values = list(records(run))
    merged = timing.merge(run, 42, values)
    assert merged == {
        name: {"test.py::test_1": 1.0, "test.py::test_2": 2.0}
        for name in timing.MAP_NAMES
    }
    assert timing.merge(run, 42, values[:-1]) is None
    rerun = timing.Run(run.run_id, 2, run.sha, run.matrices)
    assert timing.merge(rerun, 42, list(records(rerun))[:1]) is None


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate_shard",
        "duplicate_test",
        "baseline",
        "sha",
        "attempt",
        "count",
        "unexpected",
    ],
)
def test_merge_rejects_mixed_or_ambiguous_measurements(run, problem):
    values = deepcopy(list(records(run)))
    if problem == "duplicate_shard":
        values.append(values[0])
    elif problem == "duplicate_test":
        values[1]["durations"] = values[0]["durations"]
    elif problem == "baseline":
        values[0]["input_artifact_id"] = 99
    elif problem == "sha":
        values[0]["sha"] = "b" * 40
    elif problem == "attempt":
        values[0]["attempt"] = 2
    elif problem == "count":
        values[0]["shards"] = 3
    else:
        values[0]["os"] = "other"
    with pytest.raises(ValueError):
        timing.merge(run, 42, values)


@pytest.mark.parametrize(
    "manifest", [None, [], {}, {"schema": True}, {"schema": 1, "kind": "unknown"}]
)
def test_malformed_manifests_are_rejected(manifest):
    with pytest.raises(ValueError):
        timing.read_archive(archive(manifest))


def test_newest_run_wins_over_rerun_of_an_older_run_and_prior_counts_may_differ(
    run, tmp_path
):
    old = timing.Run(1, 5, "b" * 40, run.matrices)
    new = timing.Run(
        2,
        2,
        "c" * 40,
        {suite: matrix(dict.fromkeys(RUNNERS, 1)) for suite in timing.SUITES},
    )
    github = FakeGitHub(
        runs=[prior(old, 5), prior(new, 2)],
        artifacts={
            1: [artifact(15, timing.COMPLETE_PREFIX + "5")],
            2: [
                artifact(21, timing.COMPLETE_PREFIX + "1"),
                artifact(22, timing.COMPLETE_PREFIX + "2"),
            ],
        },
        blobs={22: archive(new.manifest("complete"))},
    )
    timing.prepare(run, tmp_path, github)
    manifest, _ = timing.read_bundle(tmp_path)
    assert github.downloaded == [22]
    assert manifest["matrices"] == run.matrices


@pytest.mark.parametrize(
    "status,conclusion",
    [("in_progress", None), ("completed", "failure"), ("completed", "cancelled")],
)
def test_failed_or_running_sources_are_not_reused(run, tmp_path, status, conclusion):
    source = timing.Run(2, 1, "b" * 40, run.matrices)
    github = FakeGitHub(
        runs=[prior(source) | {"status": status, "conclusion": conclusion}]
    )
    timing.prepare(run, tmp_path, github)
    assert github.downloaded == []
    assert timing.read_bundle(tmp_path)[0]["source_artifact_id"] is None


def test_api_artifacts_are_paginated(monkeypatch):
    calls = []

    def api(self, suffix, *args):
        calls.append((suffix, args))
        return json.dumps(
            [{"artifacts": [{"id": 1}]}, {"artifacts": [{"id": 2}]}]
        ).encode()

    monkeypatch.setattr(timing.GitHub, "api", api)
    assert timing.GitHub("owner/repo").artifacts(10) == [{"id": 1}, {"id": 2}]
    assert calls == [("runs/10/artifacts?per_page=100", ("--paginate", "--slurp"))]


def test_commands_copy_record_and_merge_real_files_with_same_input_on_partial_rerun(
    run, tmp_path, monkeypatch
):
    root = tmp_path / "fcc-ci-timings"
    baseline = root / "baseline"
    timing.write_bundle(baseline, run.manifest("input"), maps())
    env = {
        "GITHUB_RUN_ID": str(run.run_id),
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SHA": run.sha,
        "RUNNER_TEMP": str(tmp_path),
        "CI_TIMINGS_INPUT_ID": "42",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        **{
            f"CI_{suite.upper()}_MATRIX": json.dumps(value)
            for suite, value in run.matrices.items()
        },
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for (suite, system, shard), count in timing.expected_shards(run.matrices).items():
        for key, value in {
            "CI_TIMINGS_SUITE": suite,
            "RUNNER_OS": system,
            "CI_TIMINGS_SHARD": str(shard),
            "CI_TIMINGS_SHARDS": str(count),
        }.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(sys, "argv", ["ci_timings", "start"])
        timing.main()
        working = root / "working" / "durations.json"
        assert (
            timing.load_json(working.read_bytes()) == maps()[f"{suite}-{system}.json"]
        )
        timing.write_json(working, {f"test.py::test_{shard}": shard})
        monkeypatch.setattr(sys, "argv", ["ci_timings", "record"])
        timing.main()
        result = timing.load_json((root / "result" / "result.json").read_bytes())
        timing.write_json(
            root / "shards" / f"{suite}-{system}-{shard}" / "result.json", result
        )
    monkeypatch.setattr(sys, "argv", ["ci_timings", "merge"])
    timing.main()
    manifest, merged = timing.read_bundle(root / "complete")
    assert manifest["input_artifact_id"] == 42
    assert all(len(values) == 2 for values in merged.values())
    assert "complete=true" in (tmp_path / "output").read_text()
    # A retry starts from original input, not its own previous measurement.
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setattr(sys, "argv", ["ci_timings", "start"])
    timing.main()
    assert timing.load_json(working.read_bytes()) == maps()[f"{suite}-{system}.json"]
