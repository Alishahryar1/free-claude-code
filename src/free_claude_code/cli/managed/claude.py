"""Managed Claude Code task command, environment, and stdout parsing."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from loguru import logger

from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.harnesses.claude import (
    CLAUDE_BINARY_NAME,
    build_claude_proxy_env,
)

MANAGED_CLAUDE_MODEL_TIER = "fable"


@dataclass(frozen=True, slots=True)
class ManagedClaudeTaskRequest:
    """One prompt execution request for a managed Claude Code subprocess."""

    prompt: str
    session_id: str | None = None
    fork_session: bool = False


@dataclass(frozen=True, slots=True)
class ManagedClaudeInvocation:
    """Concrete subprocess invocation assembled for a managed Claude task."""

    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: str
    trace_metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ManagedClaudeConfig:
    """Configuration for a managed Claude Code subprocess."""

    workspace_path: str
    proxy_root_url: str
    auth_token: str
    allowed_dirs: list[str] = field(default_factory=list)
    claude_bin: str = CLAUDE_BINARY_NAME


@dataclass(slots=True)
class ManagedClaudeParseState:
    """Mutable stdout parser state for one managed Claude Code task run."""

    log_raw_cli_diagnostics: bool = False
    session_id_extracted: bool = False


def build_managed_claude_invocation(
    *,
    config: ManagedClaudeConfig,
    request: ManagedClaudeTaskRequest,
    base_env: Mapping[str, str],
) -> ManagedClaudeInvocation:
    """Build a Claude Code stream-json subprocess invocation."""

    cmd = build_managed_claude_command(
        claude_bin=config.claude_bin,
        prompt=request.prompt,
        session_id=request.session_id,
        fork_session=request.fork_session,
        allowed_dirs=config.allowed_dirs,
    )
    resume_session_id = (
        request.session_id
        if request.session_id and not request.session_id.startswith("pending_")
        else None
    )
    return ManagedClaudeInvocation(
        argv=tuple(cmd),
        env=build_managed_claude_env(
            proxy_root_url=config.proxy_root_url,
            auth_token=config.auth_token,
            base_env=base_env,
        ),
        cwd=config.workspace_path,
        trace_metadata={
            "client_cli_id": "claude",
            "resume_session_id": resume_session_id,
            "fork_session": request.fork_session,
            "prompt": request.prompt,
            "cwd": config.workspace_path,
            "claude_binary": config.claude_bin,
            "managed_model_tier": MANAGED_CLAUDE_MODEL_TIER,
            "cli_argv": cmd,
        },
    )


def build_managed_claude_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Return a Claude Code task environment that targets the local proxy."""

    env = build_claude_proxy_env(
        proxy_root_url=proxy_root_url,
        auth_token=auth_token,
        base_env=base_env,
    )
    env["DISABLE_TELEMETRY"] = "1"
    env["TERM"] = "dumb"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def build_managed_claude_command(
    *,
    claude_bin: str,
    prompt: str,
    session_id: str | None,
    fork_session: bool,
    allowed_dirs: list[str],
) -> list[str]:
    """Return the Claude Code stream-json command for a managed task."""

    if session_id and not session_id.startswith("pending_"):
        cmd = [
            claude_bin,
            "--resume",
            session_id,
        ]
        if fork_session:
            cmd.append("--fork-session")
        cmd += [
            "--model",
            MANAGED_CLAUDE_MODEL_TIER,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--verbose",
        ]
    else:
        cmd = [
            claude_bin,
            "--model",
            MANAGED_CLAUDE_MODEL_TIER,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--verbose",
        ]

    for directory in allowed_dirs:
        cmd.extend(["--add-dir", directory])

    return cmd


def parse_managed_claude_stdout_line(
    line: str, state: ManagedClaudeParseState
) -> Iterable[JsonObject]:
    """Parse one Claude Code stream-json stdout line."""

    try:
        parsed: JsonValue = json.loads(line)
    except json.JSONDecodeError:
        if state.log_raw_cli_diagnostics:
            logger.debug("Non-JSON output: {}", line)
        else:
            logger.debug("Non-JSON CLI line: char_len={}", len(line))
        yield {"type": "raw", "content": line}
        return

    if not isinstance(parsed, dict):
        if state.log_raw_cli_diagnostics:
            logger.debug("Non-object JSON output: {}", line)
        else:
            logger.debug("Non-object JSON CLI line: char_len={}", len(line))
        yield {"type": "raw", "content": line}
        return
    event = parsed

    if not state.session_id_extracted:
        extracted_id = extract_managed_claude_session_id(event)
        if extracted_id:
            state.session_id_extracted = True
            logger.info("Extracted session ID: {}", extracted_id)
            yield {"type": "session_info", "session_id": extracted_id}

    yield event


def extract_managed_claude_session_id(event: object) -> str | None:
    """Extract a Claude Code session ID from supported stream-json event shapes."""

    if not isinstance(event, dict):
        return None

    if session_id := _string_value(event.get("session_id")):
        return session_id
    if session_id := _string_value(event.get("sessionId")):
        return session_id

    for key in ("init", "system", "result", "metadata"):
        nested = event.get(key)
        if not isinstance(nested, dict):
            continue
        if session_id := _string_value(nested.get("session_id")):
            return session_id
        if session_id := _string_value(nested.get("sessionId")):
            return session_id

    conversation = event.get("conversation")
    if isinstance(conversation, dict):
        return _string_value(conversation.get("id"))

    return None


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None
