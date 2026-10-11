import os

import pytest

from free_claude_code.core import interprocess_lock


@pytest.mark.parametrize("stage", ["acquire", "wait"])
@pytest.mark.parametrize("error_type", [PermissionError, KeyboardInterrupt])
def test_failed_acquisition_closes_handle_and_allows_retry(
    tmp_path, monkeypatch, stage, error_type
):
    lock = interprocess_lock.InterprocessFileLock(tmp_path / "lock")
    opened = []
    failure = error_type("acquisition interrupted")

    def try_lock(handle):
        opened.append(handle)
        if stage == "acquire":
            raise failure
        return False

    def sleep(_):
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(interprocess_lock, "_try_lock", try_lock)
            patch.setattr(interprocess_lock.time, "sleep", sleep)
            with pytest.raises(error_type) as caught:
                lock.acquire(wait=stage == "wait")
            assert caught.value is failure
            assert opened[0].closed
        assert lock.acquire()
    finally:
        lock.release()
        for handle in opened:
            handle.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows byte-range locking")
def test_empty_file_locked_by_another_handle_is_normal_contention(
    tmp_path, monkeypatch
):
    import msvcrt

    path = tmp_path / "lock"
    contender = interprocess_lock.InterprocessFileLock(path)
    opened = []
    try_lock = interprocess_lock._try_lock

    def capture(handle):
        opened.append(handle)
        return try_lock(handle)

    monkeypatch.setattr(interprocess_lock, "_try_lock", capture)
    with path.open("a+b") as holder:
        # Windows supports locks beyond EOF. Hold byte zero without writing it.
        msvcrt.locking(holder.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            assert contender.acquire() is False
            assert opened[0].closed
        finally:
            msvcrt.locking(holder.fileno(), msvcrt.LK_UNLCK, 1)
            contender.release()
            for handle in opened:
                handle.close()
    try:
        assert contender.acquire()
        assert path.stat().st_size == 0
    finally:
        contender.release()
