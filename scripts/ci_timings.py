"""Freeze pytest-split inputs per CI run and publish complete fresh measurements."""

import argparse
import io
import json
import math
import os
import re
import subprocess
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .ci_shards import MAX_SHARDS, RUNNERS

SUITES = ("pytest", "playwright")
MAP_NAMES = tuple(f"{suite}-{system}.json" for suite in SUITES for system in RUNNERS)
INPUT_NAME = "ci-timings-input-v1"
COMPLETE_PREFIX = "ci-timings-complete-v1-attempt-"
MAX_BYTES = 64 * 1024 * 1024


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(raw: str | bytes):
    return json.loads(raw, object_pairs_hook=unique_object)


def durations(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("Expected a test duration map")
    result = {}
    for node, seconds in value.items():
        if not isinstance(node, str) or not node or type(seconds) not in (int, float):
            raise ValueError("Invalid test duration entry")
        try:
            seconds = float(seconds)
        except OverflowError as error:
            raise ValueError("Invalid test duration") from error
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Durations must be finite and nonnegative")
        result[node] = seconds
    if not math.isfinite(sum(result.values())):
        raise ValueError("Duration total must be finite")
    return result


def effective_durations(value: object) -> dict[str, float]:
    # Zero weights otherwise leave later least_duration groups empty.
    return {node: max(seconds, 0.000001) for node, seconds in durations(value).items()}


def expected_shards(matrices: dict) -> dict[tuple[str, str, int], int]:
    if not isinstance(matrices, dict) or set(matrices) != set(SUITES):
        raise ValueError("Expected both CI suite matrices")
    expected = {}
    for suite in SUITES:
        if not isinstance(matrices[suite], dict):
            raise ValueError("Invalid suite matrix")
        rows = matrices[suite].get("include", [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Invalid matrix rows")
        for system, runner in RUNNERS.items():
            group = [row for row in rows if row.get("os") == system]
            if not group or len(group) > MAX_SHARDS:
                raise ValueError("Invalid shard count")
            for row in group:
                shard = row.get("shard")
                count = row.get("shards")
                key = (suite, system, shard)
                if (
                    type(shard) is not int
                    or type(count) is not int
                    or count != len(group)
                    or not 1 <= shard <= count
                    or row.get("runner") != runner
                    or key in expected
                ):
                    raise ValueError("Invalid shard matrix")
                expected[key] = count
        if len(rows) != sum(key[0] == suite for key in expected):
            raise ValueError("Unexpected OS in matrix")
    return expected


@dataclass(frozen=True)
class Run:
    run_id: int
    attempt: int
    sha: str
    matrices: dict

    def __post_init__(self) -> None:
        if (
            type(self.run_id) is not int
            or self.run_id < 1
            or type(self.attempt) is not int
            or self.attempt < 1
            or not isinstance(self.sha, str)
            or not re.fullmatch(r"[0-9a-f]{40}", self.sha)
        ):
            raise ValueError("Invalid CI run identity")
        expected_shards(self.matrices)

    def manifest(self, kind: str) -> dict:
        return {
            "schema": 1,
            "kind": kind,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "sha": self.sha,
            "matrices": deepcopy(self.matrices),
        }

    def check_input(self, manifest: dict) -> None:
        source = manifest_run(manifest, "input")
        if (
            source.run_id != self.run_id
            or source.sha != self.sha
            or source.matrices != self.matrices
            or source.attempt > self.attempt
        ):
            raise ValueError("Input does not belong to this CI run and matrix")


def manifest_run(value: dict, kind: str | None = None) -> Run:
    if (
        not isinstance(value, dict)
        or type(value.get("schema")) is not int
        or value["schema"] != 1
        or value.get("kind") not in ("input", "complete", "shard")
        or (kind is not None and value.get("kind") != kind)
        or not {"run_id", "attempt", "sha", "matrices"} <= value.keys()
    ):
        raise ValueError("Incompatible timing manifest")
    return Run(
        value["run_id"],
        value["attempt"],
        value["sha"],
        value["matrices"],
    )


def read_archive(raw: bytes) -> tuple[dict, dict[str, dict[str, float]]]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            files = archive.infolist()
            names = [item.filename for item in files]
            if len(names) != len(set(names)) or set(names) != {
                "manifest.json",
                *MAP_NAMES,
            }:
                raise ValueError("Unexpected or duplicate timing archive members")
            if sum(item.file_size for item in files) > MAX_BYTES:
                raise ValueError("Timing archive exceeds size limit")
            manifest = load_json(archive.read("manifest.json"))
            maps = {
                name: durations(load_json(archive.read(name))) for name in MAP_NAMES
            }
    except (zipfile.BadZipFile, UnicodeError, RuntimeError) as error:
        raise ValueError("Invalid timing archive") from error
    manifest_run(manifest)
    return manifest, maps


def read_bundle(directory: Path) -> tuple[dict, dict[str, dict[str, float]]]:
    manifest = load_json((directory / "manifest.json").read_bytes())
    manifest_run(manifest)
    return manifest, {
        name: durations(load_json((directory / name).read_bytes()))
        for name in MAP_NAMES
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_bundle(directory: Path, manifest: dict, maps: dict) -> None:
    write_json(directory / "manifest.json", manifest)
    for name in MAP_NAMES:
        write_json(directory / name, maps[name])


class GitHub:
    def __init__(self, repository: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Expected owner/repository")
        self.repository = repository

    def api(self, suffix: str, *args: str) -> bytes:
        return subprocess.run(
            ["gh", "api", f"repos/{self.repository}/actions/{suffix}", *args],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout

    def successful_runs(self) -> list[dict]:
        return load_json(
            self.api(
                "workflows/tests.yml/runs?status=success&event=pull_request&per_page=20"
            )
        )["workflow_runs"]

    def artifacts(self, run_id: int) -> list[dict]:
        pages = load_json(
            self.api(f"runs/{run_id}/artifacts?per_page=100", "--paginate", "--slurp")
        )
        return [artifact for page in pages for artifact in page["artifacts"]]

    def download(self, artifact_id: int) -> bytes:
        return self.api(f"artifacts/{artifact_id}/zip")


HISTORY_ERRORS = (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError)


def prepare(run: Run, directory: Path, github: GitHub) -> int | None:
    if run.attempt > 1:
        try:
            originals = [
                a
                for a in github.artifacts(run.run_id)
                if a["name"] == INPUT_NAME and not a["expired"]
            ]
            if len(originals) > 1:
                raise ValueError("Multiple original input artifacts")
            if originals:
                artifact_id = originals[0]["id"]
                manifest, maps = read_archive(github.download(artifact_id))
                run.check_input(manifest)
                write_bundle(directory, manifest, maps)
                print(f"Reusing original timing input artifact {artifact_id}")
                return artifact_id
        except HISTORY_ERRORS as error:
            raise ValueError(
                "Cannot recover original timing input; start a fresh CI run"
            ) from error
        # Full reruns clear artifacts. Rerunning preparation also restarts all
        # dependent shards, so they can safely share a newly frozen input.
        print("Original timing input is absent; preparing input for restarted shards")

    maps = {name: {} for name in MAP_NAMES}
    source_id = None
    try:
        runs = sorted(
            github.successful_runs(),
            key=lambda r: (r["created_at"], r["id"]),
            reverse=True,
        )
        for source in runs[:20]:
            if (
                source["id"] == run.run_id
                or source["status"] != "completed"
                or source["conclusion"] != "success"
            ):
                continue
            try:
                candidates = []
                for artifact in github.artifacts(source["id"]):
                    match = re.fullmatch(
                        re.escape(COMPLETE_PREFIX) + r"([1-9][0-9]*)", artifact["name"]
                    )
                    if match and not artifact["expired"]:
                        candidates.append((int(match[1]), artifact["id"]))
                for attempt, artifact_id in sorted(candidates, reverse=True):
                    try:
                        manifest, candidate = read_archive(github.download(artifact_id))
                        identity = manifest_run(manifest, "complete")
                        if (
                            identity.run_id != source["id"]
                            or identity.attempt != attempt
                            or attempt > source["run_attempt"]
                        ):
                            raise ValueError(
                                "Source run does not match timing artifact"
                            )
                        maps = {
                            name: effective_durations(values)
                            for name, values in candidate.items()
                        }
                        source_id = artifact_id
                        print(
                            f"Loaded timings from run {source['id']}, artifact {artifact_id}"
                        )
                        break
                    except HISTORY_ERRORS as error:
                        print(
                            f"Skipping timing artifact {artifact_id}: {type(error).__name__}"
                        )
            except HISTORY_ERRORS as error:
                print(f"Skipping timing run {source['id']}: {type(error).__name__}")
            if source_id is not None:
                break
    except HISTORY_ERRORS as error:
        print(
            f"::warning::Timing history unavailable ({type(error).__name__}); using a cold start"
        )
    if source_id is None:
        print("No compatible timing history; freezing empty timing maps")
    write_bundle(
        directory, run.manifest("input") | {"source_artifact_id": source_id}, maps
    )
    for name, values in maps.items():
        print(f"{name}: {len(values)} timing entries")
    return None


def merge(run: Run, input_id: int, records: list[dict]) -> dict | None:
    expected = expected_shards(run.matrices)
    seen = set()
    maps = {name: {} for name in MAP_NAMES}
    for record in records:
        identity = manifest_run(record, "shard")
        if not {"suite", "os", "shard", "shards"} <= record.keys():
            raise ValueError("Missing timing shard identity")
        key = (record["suite"], record["os"], record["shard"])
        if (
            identity != run
            or record.get("input_artifact_id") != input_id
            or type(record.get("shard")) is not int
            or type(record.get("shards")) is not int
            or key not in expected
            or record.get("shards") != expected[key]
            or key in seen
        ):
            raise ValueError("Unexpected, duplicate, or mismatched timing shard")
        seen.add(key)
        values = durations(record.get("durations"))
        target = maps[f"{key[0]}-{key[1]}.json"]
        if target.keys() & values.keys():
            raise ValueError("Duplicate test node IDs across timing shards")
        target.update(values)
    if seen != expected.keys():
        print(
            f"Only {len(seen)}/{len(expected)} current-attempt shards available; skipping timing refresh"
        )
        return None
    return maps


def output(name: str, value: str | int) -> None:
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as file:
        file.write(f"{name}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "start", "record", "merge"))
    args = parser.parse_args()
    run = Run(
        int(os.environ["GITHUB_RUN_ID"]),
        int(os.environ["GITHUB_RUN_ATTEMPT"]),
        os.environ["GITHUB_SHA"],
        {
            suite: load_json(os.environ[f"CI_{suite.upper()}_MATRIX"])
            for suite in SUITES
        },
    )
    root = Path(os.environ["RUNNER_TEMP"]) / "fcc-ci-timings"
    baseline = root / "baseline"
    if args.command == "prepare":
        original = prepare(run, baseline, GitHub(os.environ["GITHUB_REPOSITORY"]))
        output("artifact_id", original or "")
        return
    input_id = int(os.environ["CI_TIMINGS_INPUT_ID"])
    manifest, maps = read_bundle(baseline)
    run.check_input(manifest)
    if args.command == "merge":
        records = [
            load_json(path.read_bytes())
            for path in sorted((root / "shards").glob("*/result.json"))
        ]
        merged = merge(run, input_id, records)
        output("complete", "true" if merged is not None else "false")
        if merged is not None:
            write_bundle(
                root / "complete",
                run.manifest("complete") | {"input_artifact_id": input_id},
                merged,
            )
        return
    suite, system = os.environ["CI_TIMINGS_SUITE"], os.environ["RUNNER_OS"]
    shard, count = (
        int(os.environ["CI_TIMINGS_SHARD"]),
        int(os.environ["CI_TIMINGS_SHARDS"]),
    )
    if expected_shards(run.matrices).get((suite, system, shard)) != count:
        raise ValueError("Shard does not belong to frozen input matrix")
    working = root / "working" / "durations.json"
    if args.command == "start":
        write_json(working, maps[f"{suite}-{system}.json"])
        print(
            f"Timing input artifact {input_id}: {suite}/{system}, shard {shard}/{count}"
        )
    else:
        write_json(
            root / "result" / "result.json",
            run.manifest("shard")
            | {
                "input_artifact_id": input_id,
                "suite": suite,
                "os": system,
                "shard": shard,
                "shards": count,
                "durations": durations(load_json(working.read_bytes())),
            },
        )


if __name__ == "__main__":
    main()
