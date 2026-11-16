"""Exercise actual xdist scheduling, not just marker construction."""

import json
import os
import subprocess
import sys
import textwrap

from smoke.conftest import provider_xdist_group
from smoke.lib.config import ProviderModel


def test_provider_groups_stay_on_one_worker(tmp_path):
    groups = {
        name: provider_xdist_group(ProviderModel(name, f"{name}/model", "test"))
        for name in ("alpha", "beta")
    }
    module = tmp_path / "test_grouping.py"
    module.write_text(
        textwrap.dedent(f"""
        import json
        from pathlib import Path
        import pytest

        @pytest.mark.parametrize('scenario', ['first', 'second'])
        @pytest.mark.parametrize('provider', [
            pytest.param('alpha', marks=pytest.mark.xdist_group({groups["alpha"]!r})),
            pytest.param('beta', marks=pytest.mark.xdist_group({groups["beta"]!r})),
        ])
        def test_group(provider, scenario, worker_id):
            path = Path(__file__).parent / (provider + '-' + scenario + '.json')
            path.write_text(json.dumps({{'provider': provider, 'worker': worker_id}}))
    """)
    )
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n")
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")
    }
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(config),
            "-p",
            "xdist.plugin",
            "-n",
            "2",
            "--dist=loadgroup",
            "-q",
            str(module),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assignments = {
        provider: {
            json.loads((tmp_path / f"{provider}-{scenario}.json").read_text())["worker"]
            for scenario in ("first", "second")
        }
        for provider in groups
    }
    assert all(len(workers) == 1 for workers in assignments.values()), assignments
    assert assignments["alpha"] != assignments["beta"], assignments
