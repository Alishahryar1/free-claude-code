"""Exercise the CI split options against pytest's actual collected items."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


@pytest.fixture
def suite(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "def pytest_collection_finish(session):\n"
        "    if session.config.option.collectonly:\n"
        "        Path('items.json').write_text(json.dumps([i.nodeid for i in session.items]))\n",
        encoding="utf-8",
    )
    parameterized = (
        "import pytest\n"
        "@pytest.mark.parametrize('value', range(7))\n"
        "def test_same_name(value):\n"
        "    assert value >= 0\n"
    )
    (tmp_path / "test_a.py").write_text(
        parameterized
        + "@pytest.mark.skip(reason='fixture skip')\ndef test_skipped(): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "test_b.py").write_text(
        parameterized + "def test_plain(): pass\n", encoding="utf-8"
    )
    return tmp_path


def run_pytest(suite, *args):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pytest_split.plugin",
            "-p",
            "xdist.plugin",
            *args,
        ],
        cwd=suite,
        env={**os.environ, "PYTEST_ADDOPTS": "", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def collect(suite, *args):
    result = run_pytest(suite, "--collect-only", "-q", *args)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((suite / "items.json").read_text())


def split_args(count, group):
    return [
        "--splits",
        str(count),
        "--group",
        str(group),
        "--splitting-algorithm=least_duration",
    ]


@pytest.mark.parametrize("count", [2, 3, 5])
def test_shards_partition_all_items_once_and_run_with_workstealing(suite, count):
    baseline = collect(suite)
    assert len(baseline) == 16
    assert collect(suite, *split_args(1, 1)) == baseline
    groups = [
        collect(suite, *split_args(count, group)) for group in range(1, count + 1)
    ]
    selected = [nodeid for group in groups for nodeid in group]
    assert all(groups)
    assert len(selected) == len(set(selected))
    assert set(selected) == set(baseline)
    assert "test_a.py::test_same_name[0]" in selected
    assert "test_b.py::test_same_name[0]" in selected

    # Execute the shard containing the skip, with real xdist workers.
    skip_group = next(
        i for i, group in enumerate(groups, 1) if "test_a.py::test_skipped" in group
    )
    result = run_pytest(
        suite,
        *split_args(count, skip_group),
        "-n",
        "2",
        "--dist=worksteal",
        "--junitxml=results.xml",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cases = ET.parse(suite / "results.xml").findall(".//testcase")
    executed = {
        case.attrib["classname"] + ".py::" + case.attrib["name"] for case in cases
    }
    assert executed == set(groups[skip_group - 1])
    assert len(cases) == len(executed)
    assert len([case for case in cases if case.find("skipped") is not None]) == 1


@pytest.mark.parametrize("count,group", [(0, 1), (5, 0), (5, 6), (17, 17)])
def test_invalid_or_empty_shards_fail(suite, count, group):
    result = run_pytest(suite, *split_args(count, group))
    assert result.returncode == (5 if count == 17 else 4), result.stdout + result.stderr
