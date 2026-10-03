"""DSH's cooperative file locks and YAML preservation are native contracts."""

import os

import pytest

from free_claude_code.harnesses import dsh_files


def test_lock_uses_native_pid_bytes_and_is_released(tmp_path):
    path = tmp_path / "package.json.lock"
    with dsh_files.file_lock(path, wait=0):
        assert path.read_bytes() == f"{os.getpid()}\n".encode()
    assert not path.exists()


def test_lock_does_not_steal_or_remove_an_existing_record(tmp_path):
    path = tmp_path / "package.json.lock"
    path.write_bytes(b"999999999\n")
    with pytest.raises(TimeoutError), dsh_files.file_lock(path, wait=0):
        pytest.fail("acquired another owner's lock")
    assert path.read_bytes() == b"999999999\n"


def test_lock_release_does_not_remove_a_replacement(tmp_path):
    path = tmp_path / "package.json.lock"
    with dsh_files.file_lock(path, wait=0):
        path.write_bytes(b"another-owner\n")
    assert path.read_bytes() == b"another-owner\n"


def test_yaml_errors_never_quote_credentials(tmp_path):
    path = tmp_path / ".credentials.yaml"
    path.write_text("refs: {KEY: super-secret, KEY: another-secret}")
    with pytest.raises(ValueError) as error:
        dsh_files.read_yaml(path)
    assert "secret" not in str(error.value)


def test_untouched_tags_comments_and_anchors_survive(tmp_path):
    path = tmp_path / "cordis.patch.yml"
    path.write_text("""# native comment
other: &keep
  expr: !!js "ctx.get('native')"
copy: *keep
""")
    doc = dsh_files.read_yaml(path)
    dsh_files.write_yaml(path, doc)
    result = path.read_text()
    assert "# native comment" in result
    assert "&keep" in result and "*keep" in result
    assert "!!js" in result
