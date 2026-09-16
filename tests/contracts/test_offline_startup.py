"""Contracts for starting FCC without token-encoder network access."""

import subprocess
import sys


def test_api_import_does_not_acquire_tiktoken_encoding() -> None:
    script = """
import os
import tempfile

import requests

calls = []


def fail(url, *args, **kwargs):
    del args, kwargs
    calls.append(url)
    raise requests.exceptions.ProxyError("Bearer secret")


with tempfile.TemporaryDirectory() as cache_dir:
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    requests.get = fail
    import free_claude_code.api.app

assert calls == [], calls
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "ProxyError" not in completed.stderr
    assert "Bearer secret" not in completed.stderr
