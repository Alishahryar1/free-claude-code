"""CI matrix generation rejects invalid counts and covers every OS shard."""

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from free_claude_code.core.json_types import load_json_value
from tests.wire_payload import array_payload, object_payload

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = {
    "Linux": "ubuntu-latest",
    "Windows": "windows-latest",
    "macOS": "macos-latest",
}


def run_matrix(counts: Sequence[int | str]):
    arguments = []
    for suite, suite_counts in zip(
        ("pytest", "playwright"), (counts[:3], counts[3:]), strict=True
    ):
        for os_name, count in zip(RUNNERS, suite_counts, strict=True):
            arguments.extend([f"--{suite}-{os_name.lower()}-shards", str(count)])
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/ci_shards.py"),
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    "counts",
    [(1, 1, 1, 1, 1, 1), (3, 3, 2, 3, 3, 3), (2, 3, 4, 5, 6, 7), (85, 85, 85, 1, 1, 1)],
)
def test_matrix_outputs_cover_each_os_and_shard(counts: tuple[int, ...]):
    result = run_matrix(counts)
    assert result.returncode == 0, result.stderr
    outputs = {
        name: load_json_value(value)
        for name, value in (line.split("=", 1) for line in result.stdout.splitlines())
    }
    assert set(outputs) == {"pytest_matrix", "playwright_matrix"}
    for name, suite_counts in zip(
        ("pytest_matrix", "playwright_matrix"), (counts[:3], counts[3:]), strict=True
    ):
        entries = object_payload(outputs[name])["include"]
        assert len(array_payload(entries)) == sum(suite_counts)
        assert {
            (object_payload(row)["os"], object_payload(row)["runner"])
            for row in array_payload(entries)
        } == set(RUNNERS.items())
        for os_name, count in zip(RUNNERS, suite_counts, strict=True):
            rows = [
                row
                for row in array_payload(entries)
                if object_payload(row)["os"] == os_name
            ]
            assert [object_payload(row)["shard"] for row in rows] == list(
                range(1, count + 1)
            )
            assert {object_payload(row)["shards"] for row in rows} == {count}


@pytest.mark.parametrize("invalid", ["0", "-1", "1.5", "five", "86"])
@pytest.mark.parametrize("position", range(6))
def test_invalid_count_fails_before_emitting_either_matrix(invalid: str, position: int):
    counts = ["5"] * 6
    counts[position] = invalid
    result = run_matrix(counts)
    assert result.returncode == 2
    assert "shards" in result.stderr
    assert result.stdout == ""
