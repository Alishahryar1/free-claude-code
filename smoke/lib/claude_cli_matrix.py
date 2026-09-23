"""Claude Code CLI characterization helpers for provider smoke matrices."""

import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from free_claude_code.harnesses.claude import build_claude_proxy_env
from smoke.lib.child_process import run_captured_text
from smoke.lib.config import ProviderModel, SmokeConfig, redacted
from smoke.lib.outcomes import is_upstream_unavailable_text
from smoke.lib.server import RunningServer

REGRESSION_CLASSIFICATIONS = frozenset({"harness_bug", "product_failure"})

_HTTP_REGRESSION_PATTERNS = (
    r'POST /v1/messages[^"\n]* HTTP/1\.1" 4(?!01|03|04|08|09)\d\d',
    r'POST /v1/messages[^"\n]* HTTP/1\.1" 5\d\d',
)
_MISSING_ENV_MARKERS = (
    "api key",
    "not logged in",
    "authentication",
    "permission denied",
)
_EMPTY_MCP_CONFIG = '{"mcpServers":{}}'
_SUBAGENT_SYSTEM_PROMPT = (
    "You are a deterministic smoke-test coordinator. Use Agent when asked to "
    "use a subagent."
)


@dataclass(frozen=True, slots=True)
class ClaudeCliRun:
    command: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}\n{self.stderr}"


@dataclass(frozen=True, slots=True)
class CliMatrixOutcome:
    model: str
    full_model: str
    source: str
    feature: str
    outcome: str
    classification: str
    duration_s: float
    cli_returncode: int | None
    token_evidence: dict[str, Any]
    request_count: int
    log_path: str
    stdout_excerpt: str
    stderr_excerpt: str
    log_excerpt: str


def run_claude_cli(
    *,
    claude_bin: str,
    server: RunningServer,
    config: SmokeConfig,
    cwd: Path,
    prompt: str,
    tools: str | None,
    bare: bool = True,
    pre_tool_args: tuple[str, ...] = (),
    extra_args: tuple[str, ...] = (),
    session_id: str | None = None,
    resume_session_id: str | None = None,
    no_session_persistence: bool = True,
    auto_mode: bool = False,
) -> ClaudeCliRun:
    """Run Claude Code CLI against the local smoke proxy."""
    cwd.mkdir(parents=True, exist_ok=True)

    cmd = list(
        _build_claude_cli_command(
            claude_bin=claude_bin,
            prompt=prompt,
            tools=tools,
            bare=bare,
            pre_tool_args=pre_tool_args,
            extra_args=extra_args,
            session_id=session_id,
            resume_session_id=resume_session_id,
            no_session_persistence=no_session_persistence,
            auto_mode=auto_mode,
        )
    )

    env = build_claude_proxy_env(
        proxy_root_url=server.base_url,
        auth_token=config.settings.proxy_auth_token,
        base_env=os.environ,
    )
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    started = time.monotonic()
    try:
        result = run_captured_text(
            cmd,
            cwd=cwd,
            env=env,
            timeout=config.timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ClaudeCliRun(
            command=tuple(cmd),
            returncode=None,
            stdout=_coerce_timeout_text(exc.stdout),
            stderr=_coerce_timeout_text(exc.stderr),
            duration_s=time.monotonic() - started,
            timed_out=True,
        )

    return ClaudeCliRun(
        command=tuple(cmd),
        returncode=result.returncode,
        stdout=_coerce_timeout_text(result.stdout),
        stderr=_coerce_timeout_text(result.stderr),
        duration_s=time.monotonic() - started,
    )


def _build_claude_cli_command(
    *,
    claude_bin: str,
    prompt: str,
    tools: str | None,
    bare: bool = True,
    pre_tool_args: tuple[str, ...] = (),
    extra_args: tuple[str, ...] = (),
    session_id: str | None = None,
    resume_session_id: str | None = None,
    no_session_persistence: bool = True,
    auto_mode: bool = False,
) -> tuple[str, ...]:
    cmd: list[str] = [claude_bin]
    if bare:
        cmd.append("--bare")
    if resume_session_id:
        cmd.extend(["--resume", resume_session_id])
    if session_id:
        cmd.extend(["--session-id", session_id])
    cmd.extend(
        [
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--permission-mode",
            "auto" if auto_mode else "bypassPermissions",
        ]
    )
    if not auto_mode:
        cmd.append("--dangerously-skip-permissions")
    cmd.extend(
        [
            "--model",
            "sonnet",
        ]
    )
    if no_session_persistence:
        cmd.append("--no-session-persistence")
    cmd.extend(pre_tool_args)
    if tools is not None:
        cmd.extend(["--tools", tools])
        if tools and not auto_mode:
            cmd.extend(["--allowedTools", tools])
    cmd.extend(extra_args)
    cmd.extend(["-p", prompt])
    return tuple(cmd)


def run_cli_feature_probes(
    *,
    claude_bin: str,
    server: RunningServer,
    smoke_config: SmokeConfig,
    provider_model: ProviderModel,
    model_dir: Path,
    marker_prefix: str,
) -> list[CliMatrixOutcome]:
    return [
        _basic_text(
            claude_bin, server, smoke_config, provider_model, model_dir, marker_prefix
        ),
        _thinking(
            claude_bin, server, smoke_config, provider_model, model_dir, marker_prefix
        ),
        _tool_use_roundtrip(
            claude_bin, server, smoke_config, provider_model, model_dir, marker_prefix
        ),
        _interleaved_thinking_tool(
            claude_bin, server, smoke_config, provider_model, model_dir, marker_prefix
        ),
        _subagent_task(
            claude_bin, server, smoke_config, provider_model, model_dir, marker_prefix
        ),
        _compact_command(
            claude_bin, server, smoke_config, provider_model, model_dir, marker_prefix
        ),
    ]


def read_log_offset(log_path: Path) -> int:
    """Return the current text length of a smoke server log."""
    if not log_path.is_file():
        return 0
    return len(log_path.read_text(encoding="utf-8", errors="replace"))


def read_log_delta(log_path: Path, offset: int) -> str:
    """Return smoke server log text written after ``offset``."""
    if not log_path.is_file():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return text[offset:]


def token_evidence(
    *,
    feature: str,
    marker: str,
    run: ClaudeCliRun,
    log_delta: str,
) -> dict[str, Any]:
    """Collect compact evidence for a CLI feature probe."""
    combined = f"{run.combined_output}\n{log_delta}"
    lower = combined.lower()
    return {
        "feature": feature,
        "marker_present": bool(marker and marker in combined),
        "thinking_delta_count": combined.count("thinking_delta"),
        "tool_use_count": combined.count('"tool_use"'),
        "tool_result_count": combined.count('"tool_result"'),
        "agent_catalog_present": _tool_catalog_has(log_delta, "Agent"),
        "agent_tool_count": _agent_tool_count(combined),
        "agent_result_count": _agent_result_count(combined),
        "task_tool_count": combined.count('"name": "Task"')
        + combined.count('"name":"Task"'),
        "run_in_background_false": "run_in_background" in combined and "false" in lower,
        "compact_boundary": "compact_boundary" in combined,
        "compact_metadata": "compact_metadata" in combined,
        "http_422": 'HTTP/1.1" 422' in combined,
        "http_500": bool(re.search(r'HTTP/1\.1" 5\d\d', combined)),
        "timed_out": run.timed_out,
    }


def classify_probe(
    *,
    run: ClaudeCliRun,
    log_delta: str,
    marker: str,
    requires_tool_result: bool = False,
    requires_agent: bool = False,
    requires_task: bool = False,
    requires_compact: bool = False,
) -> tuple[str, str]:
    """Classify a probe without failing compatibility characterization failures."""
    combined = f"{run.combined_output}\n{log_delta}"
    lower = combined.lower()

    if _has_proxy_regression(log_delta):
        return "failed", "product_failure"
    if run.returncode != 0 and any(
        marker_text in lower for marker_text in _MISSING_ENV_MARKERS
    ):
        return "skipped", "missing_env"
    if run.timed_out:
        return "failed", "probe_timeout"
    marker_ok = not marker or marker in combined
    tool_ok = not requires_tool_result or '"tool_result"' in combined
    agent_ok = not requires_agent or (
        _agent_tool_count(combined) > 0 and _agent_result_count(combined) > 0
    )
    task_ok = not requires_task or (
        ('"name": "Task"' in combined or '"name":"Task"' in combined)
        and "run_in_background" in combined
        and "false" in lower
    )
    compact_ok = not requires_compact or (
        "compact_boundary" in combined
        or "compact_metadata" in combined
        or "/compact" in combined
        or "compact" in lower
    )
    cli_ok = run.returncode == 0

    if cli_ok and marker_ok and tool_ok and agent_ok and task_ok and compact_ok:
        return "passed", "passed"
    if is_upstream_unavailable_text(combined):
        return "failed", "upstream_unavailable"
    if not _has_proxy_request(log_delta):
        return "failed", "harness_bug"
    return "failed", "model_feature_failure"


def make_outcome(
    *,
    model: str,
    full_model: str,
    source: str,
    feature: str,
    marker: str,
    run: ClaudeCliRun,
    log_delta: str,
    log_path: Path,
    requires_tool_result: bool = False,
    requires_agent: bool = False,
    requires_task: bool = False,
    requires_compact: bool = False,
) -> CliMatrixOutcome:
    """Build one report outcome from a CLI run and its server log delta."""
    outcome, classification = classify_probe(
        run=run,
        log_delta=log_delta,
        marker=marker,
        requires_tool_result=requires_tool_result,
        requires_agent=requires_agent,
        requires_task=requires_task,
        requires_compact=requires_compact,
    )
    evidence = token_evidence(
        feature=feature,
        marker=marker,
        run=run,
        log_delta=log_delta,
    )
    return CliMatrixOutcome(
        model=model,
        full_model=full_model,
        source=source,
        feature=feature,
        outcome=outcome,
        classification=classification,
        duration_s=round(run.duration_s, 3),
        cli_returncode=run.returncode,
        token_evidence=evidence,
        request_count=_request_count(log_delta),
        log_path=str(log_path),
        stdout_excerpt=_excerpt(run.stdout),
        stderr_excerpt=_excerpt(run.stderr),
        log_excerpt=_excerpt(log_delta),
    )


def write_matrix_report(
    config: SmokeConfig,
    outcomes: list[CliMatrixOutcome],
    *,
    target: str,
    filename_prefix: str,
) -> Path:
    """Write a Claude CLI compatibility matrix report."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    path = (
        config.results_dir
        / f"{filename_prefix}-matrix-{config.worker_id}-{int(time.time())}.json"
    )
    payload = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "worker_id": config.worker_id,
        "target": target,
        "models": sorted({outcome.full_model for outcome in outcomes}),
        "outcomes": [asdict(outcome) for outcome in outcomes],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def regression_failures(outcomes: list[CliMatrixOutcome]) -> list[str]:
    """Return report lines for classifications that should fail pytest."""
    return [
        f"{outcome.full_model} {outcome.feature}: {outcome.classification}"
        for outcome in outcomes
        if outcome.classification in REGRESSION_CLASSIFICATIONS
    ]


def _basic_text(
    claude_bin: str,
    server: RunningServer,
    smoke_config: SmokeConfig,
    provider_model: ProviderModel,
    model_dir: Path,
    marker_prefix: str,
) -> CliMatrixOutcome:
    marker = _marker(marker_prefix, "BASIC")
    return _run_probe(
        claude_bin=claude_bin,
        server=server,
        smoke_config=smoke_config,
        provider_model=provider_model,
        workspace=model_dir / "basic_text",
        feature="basic_text",
        marker=marker,
        prompt=f"Reply with exactly {marker} and no other text.",
        tools="",
    )


def _thinking(
    claude_bin: str,
    server: RunningServer,
    smoke_config: SmokeConfig,
    provider_model: ProviderModel,
    model_dir: Path,
    marker_prefix: str,
) -> CliMatrixOutcome:
    marker = _marker(marker_prefix, "THINK")
    return _run_probe(
        claude_bin=claude_bin,
        server=server,
        smoke_config=smoke_config,
        provider_model=provider_model,
        workspace=model_dir / "thinking",
        feature="thinking",
        marker=marker,
        prompt=(
            "Think privately about the request, then reply with exactly "
            f"{marker} and no other text."
        ),
        tools="",
        extra_args=("--effort", "high"),
    )


def _tool_use_roundtrip(
    claude_bin: str,
    server: RunningServer,
    smoke_config: SmokeConfig,
    provider_model: ProviderModel,
    model_dir: Path,
    marker_prefix: str,
) -> CliMatrixOutcome:
    marker = _marker(marker_prefix, "TOOL")
    workspace = model_dir / "tool_use_roundtrip"
    (workspace / "smoke-read.txt").parent.mkdir(parents=True, exist_ok=True)
    (workspace / "smoke-read.txt").write_text(marker, encoding="utf-8")
    return _run_probe(
        claude_bin=claude_bin,
        server=server,
        smoke_config=smoke_config,
        provider_model=provider_model,
        workspace=workspace,
        feature="tool_use_roundtrip",
        marker=marker,
        prompt=(
            "Use the Read tool to read smoke-read.txt. Reply with exactly the "
            "secret token from that file and no other text."
        ),
        tools="Read",
        requires_tool_result=True,
    )


def _interleaved_thinking_tool(
    claude_bin: str,
    server: RunningServer,
    smoke_config: SmokeConfig,
    provider_model: ProviderModel,
    model_dir: Path,
    marker_prefix: str,
) -> CliMatrixOutcome:
    marker = _marker(marker_prefix, "INTERLEAVED")
    workspace = model_dir / "interleaved_thinking_tool"
    (workspace / "smoke-interleaved.txt").parent.mkdir(parents=True, exist_ok=True)
    (workspace / "smoke-interleaved.txt").write_text(marker, encoding="utf-8")
    return _run_probe(
        claude_bin=claude_bin,
        server=server,
        smoke_config=smoke_config,
        provider_model=provider_model,
        workspace=workspace,
        feature="interleaved_thinking_tool",
        marker=marker,
        prompt=(
            "Think privately, use Read on smoke-interleaved.txt, then reply with "
            "exactly the secret token from that file and no other text."
        ),
        tools="Read",
        extra_args=("--effort", "high"),
        requires_tool_result=True,
    )


def _subagent_task(
    claude_bin: str,
    server: RunningServer,
    smoke_config: SmokeConfig,
    provider_model: ProviderModel,
    model_dir: Path,
    marker_prefix: str,
) -> CliMatrixOutcome:
    marker = _marker(marker_prefix, "TASK")
    workspace = model_dir / "subagent_task"
    (workspace / "smoke-subagent.txt").parent.mkdir(parents=True, exist_ok=True)
    (workspace / "smoke-subagent.txt").write_text(marker, encoding="utf-8")
    agents = json.dumps(
        {
            "smoke_reader": {
                "description": "Reads one requested file and returns its token.",
                "prompt": (
                    "Read the requested file with Read and return only the token "
                    "inside it."
                ),
                "tools": ["Read"],
                "permissionMode": "bypassPermissions",
                "background": False,
            }
        }
    )
    bare, tools, pre_tool_args, extra_args = _subagent_probe_options(agents)
    return _run_probe(
        claude_bin=claude_bin,
        server=server,
        smoke_config=smoke_config,
        provider_model=provider_model,
        workspace=workspace,
        feature="subagent_task",
        marker=marker,
        prompt=(
            "Use the smoke_reader subagent to read smoke-subagent.txt. After the "
            "first agent result, reply with exactly the token and stop. Do not "
            "call any other tools."
        ),
        tools=tools,
        bare=bare,
        pre_tool_args=pre_tool_args,
        extra_args=extra_args,
        requires_tool_result=True,
        requires_agent=True,
    )


def _subagent_probe_options(
    agents: str,
) -> tuple[bool, str, tuple[str, ...], tuple[str, ...]]:
    return (
        False,
        "Agent,Read",
        (
            "--setting-sources",
            "local",
            "--strict-mcp-config",
            "--mcp-config",
            _EMPTY_MCP_CONFIG,
            "--system-prompt",
            _SUBAGENT_SYSTEM_PROMPT,
        ),
        ("--agents", agents),
    )


def _compact_command(
    claude_bin: str,
    server: RunningServer,
    smoke_config: SmokeConfig,
    provider_model: ProviderModel,
    model_dir: Path,
    marker_prefix: str,
) -> CliMatrixOutcome:
    marker = _marker(marker_prefix, "COMPACT")
    workspace = model_dir / "compact_command"
    session_id = str(uuid.uuid4())
    offset = read_log_offset(server.log_path)
    first = run_claude_cli(
        claude_bin=claude_bin,
        server=server,
        config=smoke_config,
        cwd=workspace,
        prompt=f"Remember this smoke token: {marker}. Reply with exactly {marker}.",
        tools="",
        session_id=session_id,
        no_session_persistence=False,
    )
    second = run_claude_cli(
        claude_bin=claude_bin,
        server=server,
        config=smoke_config,
        cwd=workspace,
        prompt=f"/compact preserve {marker}",
        tools="",
        resume_session_id=session_id,
        no_session_persistence=False,
    )
    log_delta = read_log_delta(server.log_path, offset)
    run = ClaudeCliRun(
        command=(*first.command, "&&", *second.command),
        returncode=second.returncode if first.returncode == 0 else first.returncode,
        stdout=f"{first.stdout}\n{second.stdout}",
        stderr=f"{first.stderr}\n{second.stderr}",
        duration_s=first.duration_s + second.duration_s,
        timed_out=first.timed_out or second.timed_out,
    )
    return make_outcome(
        model=provider_model.model_name,
        full_model=provider_model.full_model,
        source=provider_model.source,
        feature="compact_command",
        marker="",
        run=run,
        log_delta=log_delta,
        log_path=server.log_path,
        requires_compact=True,
    )


def _run_probe(
    *,
    claude_bin: str,
    server: RunningServer,
    smoke_config: SmokeConfig,
    provider_model: ProviderModel,
    workspace: Path,
    feature: str,
    marker: str,
    prompt: str,
    tools: str | None,
    bare: bool = True,
    pre_tool_args: tuple[str, ...] = (),
    extra_args: tuple[str, ...] = (),
    requires_tool_result: bool = False,
    requires_agent: bool = False,
    requires_task: bool = False,
) -> CliMatrixOutcome:
    offset = read_log_offset(server.log_path)
    run = run_claude_cli(
        claude_bin=claude_bin,
        server=server,
        config=smoke_config,
        cwd=workspace,
        prompt=prompt,
        tools=tools,
        bare=bare,
        pre_tool_args=pre_tool_args,
        extra_args=extra_args,
    )
    log_delta = read_log_delta(server.log_path, offset)
    return make_outcome(
        model=provider_model.model_name,
        full_model=provider_model.full_model,
        source=provider_model.source,
        feature=feature,
        marker=marker,
        run=run,
        log_delta=log_delta,
        log_path=server.log_path,
        requires_tool_result=requires_tool_result,
        requires_agent=requires_agent,
        requires_task=requires_task,
    )


def _has_proxy_regression(log_delta: str) -> bool:
    if "CREATE_MESSAGE_ERROR" in log_delta:
        return True
    return any(re.search(pattern, log_delta) for pattern in _HTTP_REGRESSION_PATTERNS)


def _has_proxy_request(log_delta: str) -> bool:
    return (
        "POST /v1/messages" in log_delta
        or "API_REQUEST:" in log_delta
        or '"event": "free_claude_code.api.request.received"' in log_delta
        or (
            '"http_method": "POST"' in log_delta
            and '"http_path": "/v1/messages"' in log_delta
        )
    )


def _tool_catalog_has(log_delta: str, tool_name: str) -> bool:
    catalog = _first_tool_catalog(log_delta)
    return (
        f"'name': '{tool_name}'" in catalog
        or f'"name": "{tool_name}"' in catalog
        or f'"name":"{tool_name}"' in catalog
    )


def _first_tool_catalog(log_delta: str) -> str:
    for line in log_delta.splitlines():
        if "FULL_PAYLOAD" not in line:
            continue
        single_index = line.find("'tools': [")
        double_index = line.find('"tools": [')
        if single_index == -1 and double_index == -1:
            continue
        start = single_index if single_index != -1 else double_index
        end_candidates = [
            index
            for marker in ("'tool_choice'", '"tool_choice"', "'thinking'", '"thinking"')
            if (index := line.find(marker, start)) != -1
        ]
        end = min(end_candidates) if end_candidates else len(line)
        return line[start:end]
    return ""


def _agent_tool_count(text: str) -> int:
    return (
        text.count('"name": "Agent"')
        + text.count('"name":"Agent"')
        + len(
            re.findall(
                r"'type': 'tool_use'[^}\n]+?'name': 'Agent'",
                text,
                flags=re.DOTALL,
            )
        )
    )


def _agent_result_count(text: str) -> int:
    return text.count("agentId:") + text.count('"agentId"') + text.count("'agentId'")


def _request_count(log_delta: str) -> int:
    access_log_count = log_delta.count("POST /v1/messages")
    service_log_count = log_delta.count("API_REQUEST:")
    structured_log_count = log_delta.count(
        '"event": "free_claude_code.api.request.received"'
    )
    return max(access_log_count, service_log_count, structured_log_count)


def _marker(scope: str, prefix: str) -> str:
    return f"FCC_{scope}_{prefix}_{uuid.uuid4().hex[:8].upper()}"


def _excerpt(value: str | None, *, max_chars: int = 2400) -> str:
    if value is None:
        value = ""
    if len(value) <= max_chars:
        return redacted(value)
    return redacted(value[-max_chars:])


def _coerce_timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
