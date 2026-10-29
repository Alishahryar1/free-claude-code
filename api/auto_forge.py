"""Periodic auto-forge — closes the EvoMetaClaw × EvoForge × SkillOpt loop.

Spins an asyncio background task that calls :func:`api.evoforge.run` on a
configured interval. The newly-published policy JSON is picked up by
``api.skillopt`` on its next lookup via the existing mtime cache — no
gateway restart, no SIGHUP.

Design properties:

* **Opt-in**. Driven by ``AUTO_FORGE_ENABLED`` (default off). Fresh
  installs behave identically; the task never starts.
* **Serverless-safe**. On a read-only filesystem the first write attempt
  raises ``OSError``; the task catches it, logs once, and disables
  itself for the process — same discipline as ``health_history`` and
  ``trajectory``.
* **Non-blocking**. The forge call is CPU-bound JSON work and runs under
  ``asyncio.to_thread`` so a long aggregation can't stall the event
  loop or request handlers.
* **Bounded**. Only one forge may be in-flight at a time. The scheduler
  skips the tick if the previous run hasn't finished.

Observability: :func:`snapshot` exposes last-run metadata (success /
error / timestamps / counts) for the admin UI.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from loguru import logger

from . import evoforge


_DEFAULT_INTERVAL_HOURS = 24.0
_MIN_INTERVAL_SECONDS = 60.0  # safety floor — never spin faster than this


def is_enabled() -> bool:
    """Whether auto-forge should run in this process."""
    return os.environ.get("AUTO_FORGE_ENABLED", "").lower() in {"1", "true", "yes"}


def interval_seconds() -> float:
    """Resolve the configured interval (hours → seconds, floor applied)."""
    raw = os.environ.get("AUTO_FORGE_INTERVAL_HOURS")
    try:
        hours = float(raw) if raw else _DEFAULT_INTERVAL_HOURS
    except ValueError:
        hours = _DEFAULT_INTERVAL_HOURS
    return max(_MIN_INTERVAL_SECONDS, hours * 3600.0)


@dataclass(slots=True)
class _State:
    """Observable last-run state. Resettable; never raised."""

    runs_total: int = 0
    runs_ok: int = 0
    runs_error: int = 0
    last_run_ts: float | None = None
    last_run_status: str = "never"
    last_run_error_type: str | None = None
    last_run_duration_ms: float = 0.0
    last_policy_skills: int = 0
    disabled_reason: str | None = None


_state = _State()
_task: asyncio.Task[None] | None = None
_lock = asyncio.Lock()


async def _run_once() -> None:
    """One forge pass. Captures outcome in ``_state`` and never raises."""
    global _state
    start = time.perf_counter()
    try:
        policy = await asyncio.to_thread(evoforge.run)
    except OSError as exc:
        # Read-only FS or similar — don't retry this process.
        _state.runs_error += 1
        _state.runs_total += 1
        _state.last_run_ts = time.time()
        _state.last_run_status = "disabled"
        _state.last_run_error_type = type(exc).__name__
        _state.last_run_duration_ms = round(
            (time.perf_counter() - start) * 1000, 1
        )
        _state.disabled_reason = f"{type(exc).__name__} on write"
        logger.warning(
            "Auto-forge disabled for this process: {} ({})",
            _state.disabled_reason,
            exc,
        )
        raise  # tells the loop to stop
    except Exception as exc:
        _state.runs_error += 1
        _state.runs_total += 1
        _state.last_run_ts = time.time()
        _state.last_run_status = "error"
        _state.last_run_error_type = type(exc).__name__
        _state.last_run_duration_ms = round(
            (time.perf_counter() - start) * 1000, 1
        )
        logger.warning(
            "Auto-forge run failed: exc_type={} ({}ms)",
            type(exc).__name__,
            _state.last_run_duration_ms,
        )
        return
    _state.runs_ok += 1
    _state.runs_total += 1
    _state.last_run_ts = time.time()
    _state.last_run_status = "ok"
    _state.last_run_error_type = None
    _state.last_run_duration_ms = round((time.perf_counter() - start) * 1000, 1)
    _state.last_policy_skills = len(policy.get("policies") or {})
    logger.info(
        "Auto-forge: {} skills, {}ms",
        _state.last_policy_skills,
        _state.last_run_duration_ms,
    )


async def _loop() -> None:
    """Background scheduler. One run every ``interval_seconds()``."""
    delay = interval_seconds()
    logger.info("Auto-forge loop started (interval={}s)", int(delay))
    try:
        while True:
            try:
                async with _lock:
                    await _run_once()
            except OSError:
                return  # read-only FS → stop this process's loop
            await asyncio.sleep(delay)
    except asyncio.CancelledError:
        logger.info("Auto-forge loop cancelled")
        raise


def start() -> asyncio.Task[None] | None:
    """Start the loop. Idempotent. Returns None when disabled."""
    global _task
    if not is_enabled():
        return None
    if _task is not None and not _task.done():
        return _task
    _task = asyncio.create_task(_loop(), name="fcc.auto_forge")
    return _task


async def stop() -> None:
    """Cancel the loop and wait briefly for it to exit."""
    global _task
    if _task is None:
        return
    _task.cancel()
    try:
        await asyncio.wait_for(_task, timeout=2.0)
    except (asyncio.CancelledError, TimeoutError):
        pass
    except Exception as exc:
        logger.warning(
            "Auto-forge shutdown: exc_type={}", type(exc).__name__
        )
    finally:
        _task = None


async def run_now() -> dict[str, Any]:
    """Force a single forge pass regardless of schedule — admin UI hook."""
    async with _lock:
        try:
            await _run_once()
        except OSError:
            pass
    return snapshot()


def snapshot() -> dict[str, Any]:
    """Non-sensitive view for the admin UI."""
    data = asdict(_state)
    data["enabled"] = is_enabled()
    data["interval_seconds"] = interval_seconds()
    data["running"] = _task is not None and not _task.done()
    return data


def reset_for_tests() -> None:
    """Test-only: clear state so each case starts fresh."""
    global _state, _task
    _state = _State()
    _task = None
