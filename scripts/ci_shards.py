"""Generate GitHub Actions test matrices from the workflow's shard counts.

To replay a shard locally, set PYTEST_ADDOPTS to
"--splits 5 --group 2 --splitting-algorithm=least_duration" and run the
pytest or playwright check through scripts/ci.ps1 or scripts/ci.sh.
"""

import argparse
import json

RUNNERS = {
    "Linux": "ubuntu-latest",
    "Windows": "windows-latest",
    "macOS": "macos-latest",
}
MAX_SHARDS = 256 // len(RUNNERS)


def shard_count(value: str) -> int:
    count = int(value)
    if not 1 <= count <= MAX_SHARDS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_SHARDS}")
    return count


def matrix(count: int) -> dict[str, list[dict[str, str | int]]]:
    return {
        "include": [
            {"os": os_name, "runner": runner, "shard": shard, "shards": count}
            for os_name, runner in RUNNERS.items()
            for shard in range(1, count + 1)
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytest-shards", required=True, type=shard_count)
    parser.add_argument("--playwright-shards", required=True, type=shard_count)
    args = parser.parse_args()
    for suite, count in (
        ("pytest", args.pytest_shards),
        ("playwright", args.playwright_shards),
    ):
        print(f"{suite}_matrix={json.dumps(matrix(count), separators=(',', ':'))}")


if __name__ == "__main__":
    main()
