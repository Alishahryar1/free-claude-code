import hashlib
import io
import zipfile

import pytest

from scripts.build_windows_bundle import extract_runtime, runtime_download


def test_extract_runtime_requires_the_pinned_digest(tmp_path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("python.exe", b"runtime")
    expected = hashlib.sha256(archive.getvalue()).hexdigest()

    with pytest.raises(ValueError, match="digest"):
        extract_runtime(archive.getvalue(), "0" * 64, tmp_path)
    assert not (tmp_path / "python.exe").exists()

    extract_runtime(archive.getvalue(), expected, tmp_path)
    assert (tmp_path / "python.exe").read_bytes() == b"runtime"


def test_runtime_download_selects_the_full_pinned_x64_distribution():
    url = "https://www.python.org/ftp/python/3.14.10/python-3.14.10-amd64.zip"
    distribution = {"url": url, "hash": {"sha256": "a" * 64}}
    manifest = {"versions": [{"url": url.replace("amd64", "arm64")}, distribution]}

    assert runtime_download("3.14.10", manifest) == (url, "a" * 64)
    with pytest.raises(ValueError, match="x64"):
        runtime_download("3.14.9", manifest)
