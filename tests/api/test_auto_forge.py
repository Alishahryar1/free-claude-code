from __future__ import annotations

import asyncio
import json

import pytest

from api import auto_forge


# ── Static accessors ─────────────────────────────────────────────────


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AUTO_FORGE_ENABLED", raising=False)
    assert auto_forge.is_enabled() is False


def test_enabled_env_values(monkeypatch):
    monkeypatch.setenv("AUTO_FORGE_ENABLED", "1")
    assert auto_forge.is_enabled() is True
    monkeypatch.setenv("AUTO_FORGE_ENABLED", "true")
    assert auto_forge.is_enabled() is True
    monkeypatch.setenv("AUTO_FORGE_ENABLED", "no")
    assert auto_forge.is_enabled() is False


def test_interval_defaults_to_24h(monkeypatch):
    monkeypatch.delenv("AUTO_FORGE_INTERVAL_HOURS", raising=False)
    assert auto_forge.interval_seconds() == 24.0 * 3600.0


def test_interval_respects_env(monkeypatch):
    monkeypatch.setenv("AUTO_FORGE_INTERVAL_HOURS", "6")
    assert auto_forge.interval_seconds() == 6 * 3600.0


def test_interval_floors_tiny_values(monkeypatch):
    monkeypatch.setenv("AUTO_FORGE_INTERVAL_HOURS", "0.0001")
    # Should clamp to the 60s floor
    assert auto_forge.interval_seconds() == 60.0


def test_interval_tolerates_garbage(monkeypatch):
    monkeypatch.setenv("AUTO_FORGE_INTERVAL_HOURS", "not-a-number")
    assert auto_forge.interval_seconds() == 24.0 * 3600.0


# ── run_now() captures outcome ───────────────────────────────────────


def _write_trajectories(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _row(provider, model, skill, latency, cost):
    return {
        "provider_id": provider,
        "model": model,
        "skill": skill,
        "latency_ms": latency,
        "cost_usd": cost,
        "status": "ok",
        "input_tokens": 100,
        "output_tokens": 100,
    }


@pytest.mark.asyncio
async def test_run_now_produces_policy_and_records_success(monkeypatch, tmp_path):
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    _write_trajectories(
        tmp_path / "trajectories.jsonl",
        [_row("openai", "gpt-4o", "edit", 500, 0.001)] * 6,
    )
    auto_forge.reset_for_tests()

    state = await auto_forge.run_now()

    assert state["last_run_status"] == "ok"
    assert state["runs_ok"] == 1
    assert state["runs_total"] == 1
    assert state["last_policy_skills"] == 1
    assert (tmp_path / "skillopt_policy.json").is_file()


@pytest.mark.asyncio
async def test_run_now_records_error_without_crashing(monkeypatch, tmp_path):
    """Point the forge at a dir with no corpus. The pipeline still runs
    (writes an empty policy), so this verifies the error path by injecting
    an exception into evoforge.run and checking state."""
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    auto_forge.reset_for_tests()

    def _boom(**kwargs):
        raise RuntimeError("fake failure")

    monkeypatch.setattr(auto_forge.evoforge, "run", _boom)

    state = await auto_forge.run_now()

    assert state["last_run_status"] == "error"
    assert state["runs_error"] == 1
    assert state["last_run_error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_run_now_disables_on_readonly_fs(monkeypatch, tmp_path):
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    auto_forge.reset_for_tests()

    def _no_write(**kwargs):
        raise OSError("Read-only file system")

    monkeypatch.setattr(auto_forge.evoforge, "run", _no_write)

    state = await auto_forge.run_now()

    assert state["last_run_status"] == "disabled"
    assert state["disabled_reason"] is not None
    assert state["runs_error"] == 1


# ── snapshot shape + start/stop idempotency ──────────────────────────


def test_snapshot_reports_disabled_initially(monkeypatch):
    monkeypatch.delenv("AUTO_FORGE_ENABLED", raising=False)
    auto_forge.reset_for_tests()
    snap = auto_forge.snapshot()
    assert snap["enabled"] is False
    assert snap["running"] is False
    assert snap["last_run_status"] == "never"
    assert snap["runs_total"] == 0


def test_start_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("AUTO_FORGE_ENABLED", raising=False)
    auto_forge.reset_for_tests()
    assert auto_forge.start() is None


@pytest.mark.asyncio
async def test_start_returns_task_when_enabled_and_stop_cleans_up(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AUTO_FORGE_ENABLED", "1")
    monkeypatch.setenv("AUTO_FORGE_INTERVAL_HOURS", "999")
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    auto_forge.reset_for_tests()

    task = auto_forge.start()
    assert task is not None
    # start() is idempotent — second call returns the same task
    assert auto_forge.start() is task

    # Give it a moment to begin its first run, then stop
    await asyncio.sleep(0.05)
    await auto_forge.stop()

    assert auto_forge.snapshot()["running"] is False
