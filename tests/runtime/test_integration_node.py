"""The Node version probe runs controlled fixture processes, never installed clients."""

import os
import subprocess
import sys
import time

import pytest

from free_claude_code.runtime.integrations.discovery import LocalInstallations
from tests.integration_support import executable


@pytest.fixture
def version_probe(tmp_path, monkeypatch):
    node = executable(tmp_path / ("node.exe" if sys.platform == "win32" else "node"))
    node.write_bytes(
        b"MZ\x00\x00"
        if sys.platform == "win32"
        else b"\xcf\xfa\xed\xfe"
        if sys.platform == "darwin"
        else b"\x7fELF"
    )
    locator = LocalInstallations(
        tmp_path,
        sys.platform,
        {
            **os.environ,
            "NODE_OPTIONS": "--require private-module",
            "NODE_REPL_HISTORY": "private-path",
            "LD_PRELOAD": "private-library",
            "DYLD_INSERT_LIBRARIES": "private-library",
            "PATH": "private-path",
        },
        tmp_path,
        (),
    )
    actual_popen = subprocess.Popen
    children = []

    def probe(script):
        def spawn(argv, **kwargs):
            assert argv == [str(node), "--version"]
            assert kwargs["shell"] is False
            assert kwargs["stdin"] == subprocess.DEVNULL
            assert kwargs["stderr"] == subprocess.DEVNULL
            assert kwargs["cwd"] == tmp_path
            assert {key.upper() for key in kwargs["env"]} <= {
                "SYSTEMROOT",
                "WINDIR",
                "SYSTEMDRIVE",
                "LANG",
                "LC_ALL",
            }
            if sys.platform == "win32":
                assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
                assert any(key.upper() == "SYSTEMROOT" for key in kwargs["env"])
            child = actual_popen([sys.executable, "-c", script], **kwargs)
            children.append(child)
            return child

        monkeypatch.setattr(subprocess, "Popen", spawn)
        result = locator._node_version(node)
        assert len(children) == 1
        assert children[0].poll() is not None
        return result

    yield probe
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize(
    "output,exit_code,expected",
    [
        (b"v22.20.0\n", 0, (22, 20, 0)),
        (b"v24.1.2\r\n", 0, (24, 1, 2)),
        (b"v22.20.0\n", 1, None),
        (b"v22.20.0", 0, None),
        (b"v22.20.0\nv24.0.0\n", 0, None),
        (b"v22.20.0-nightly\n", 0, None),
        (b"Version 22\n", 0, None),
        (b"\xff\n", 0, None),
    ],
)
def test_node_probe_accepts_only_one_successful_complete_version(
    version_probe, output, exit_code, expected
):
    assert (
        version_probe(
            f"import sys; sys.stdout.buffer.write({output!r}); sys.exit({exit_code})"
        )
        == expected
    )


def test_node_probe_reads_a_version_split_across_pipe_writes(version_probe):
    assert version_probe(
        "import sys,time; sys.stdout.write('v22.'); sys.stdout.flush(); "
        "time.sleep(0.05); sys.stdout.write('20.0\\n')"
    ) == (22, 20, 0)


def test_node_probe_terminates_and_reaps_an_unresponsive_process(version_probe):
    started = time.monotonic()
    assert version_probe("import time; time.sleep(15)") is None
    assert time.monotonic() - started < 5


def test_node_probe_stops_after_excess_output_instead_of_buffering_it(version_probe):
    started = time.monotonic()
    assert (
        version_probe(
            "import sys,time; sys.stdout.buffer.write(b'x'*1000000); "
            "sys.stdout.flush(); time.sleep(15)"
        )
        is None
    )
    assert time.monotonic() - started < 5


@pytest.mark.parametrize(
    "platform,header",
    [("win32", b"MZ00"), ("linux", b"\x7fELF"), ("darwin", b"\xcf\xfa\xed\xfe")],
)
@pytest.mark.parametrize("location", ["script", "shims", "volta", "custom_volta"])
def test_node_wrappers_and_manager_shims_are_rejected_without_execution(
    tmp_path, monkeypatch, platform, header, location
):
    name = "node.exe" if platform == "win32" else "node"
    directories = {
        "script": tmp_path,
        "shims": tmp_path / "shims",
        "volta": tmp_path / ".volta/bin",
        "custom_volta": tmp_path / "custom-tools/bin",
    }
    node = executable(directories[location] / name)
    node.write_bytes(b"#!/bin/sh\n" if location == "script" else header)
    locator = LocalInstallations(
        tmp_path, platform, {"VOLTA_HOME": str(tmp_path / "custom-tools")}, tmp_path, ()
    )
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_a, **_kw: pytest.fail("wrapper must not execute")
    )
    assert locator._node_version(node) is None


def test_node_probe_execution_error_is_an_unknown_version(tmp_path, monkeypatch):
    node = executable(tmp_path / "node")
    node.write_bytes(b"\x7fELF")
    locator = LocalInstallations(tmp_path, "linux", {}, tmp_path, ())

    def unavailable(*_args, **_kwargs):
        raise OSError("private process error")

    monkeypatch.setattr(subprocess, "Popen", unavailable)
    assert locator._node_version(node) is None
