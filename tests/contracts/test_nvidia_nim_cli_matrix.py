import json
import subprocess
from pathlib import Path
from typing import cast

from free_claude_code.config.settings import Settings
from smoke.lib.claude_cli_matrix import (
    ClaudeCliRun,
    _build_claude_cli_command,
    _subagent_probe_options,
    make_outcome,
    regression_failures,
    run_claude_cli,
    write_matrix_report,
)
from smoke.lib.config import DEFAULT_TARGETS, SmokeConfig
from smoke.lib.server import RunningServer


def _smoke_config(tmp_path: Path) -> SmokeConfig:
    return SmokeConfig(
        root=tmp_path,
        results_dir=tmp_path / ".smoke-results",
        live=False,
        interactive=False,
        targets=DEFAULT_TARGETS,
        provider_matrix=frozenset(),
        timeout_s=45.0,
        prompt="Reply with exactly: FCC_SMOKE_PONG",
        claude_bin="claude",
        worker_id="test-worker",
        settings=Settings.model_construct(
            proxy_auth_enabled=False,
            proxy_auth_token="freecc",
        ),
    )


def test_nvidia_nim_cli_matrix_report_shape_and_redaction(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "secret-nim-key")
    run = ClaudeCliRun(
        command=("claude", "-p", "redacted"),
        returncode=0,
        stdout="FCC_NIM_BASIC secret-nim-key",
        stderr="",
        duration_s=1.25,
    )
    outcome = make_outcome(
        model="z-ai/glm-5.2",
        full_model="nvidia_nim/z-ai/glm-5.2",
        source="nvidia_nim_cli_default",
        feature="basic_text",
        marker="FCC_NIM_BASIC",
        run=run,
        log_delta='POST /v1/messages HTTP/1.1" 200 OK secret-nim-key',
        log_path=tmp_path / "server.log",
    )

    path = write_matrix_report(
        _smoke_config(tmp_path),
        [outcome],
        target="nvidia_nim_cli",
        filename_prefix="nvidia-nim-cli",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name.startswith("nvidia-nim-cli-matrix-test-worker-")
    assert payload["target"] == "nvidia_nim_cli"
    assert payload["models"] == ["nvidia_nim/z-ai/glm-5.2"]
    saved = payload["outcomes"][0]
    assert saved["feature"] == "basic_text"
    assert saved["classification"] == "passed"
    assert saved["request_count"] == 1
    assert saved["token_evidence"]["marker_present"] is True
    assert saved["token_evidence"]["agent_catalog_present"] is False
    assert saved["token_evidence"]["agent_tool_count"] == 0
    assert saved["token_evidence"]["agent_result_count"] == 0
    assert "secret-nim-key" not in path.read_text(encoding="utf-8")


def test_cli_matrix_normalizes_missing_captured_output(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_run_captured_text(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return cast(
            subprocess.CompletedProcess[str],
            subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=None,
                stderr=None,
            ),
        )

    monkeypatch.setattr(
        "smoke.lib.claude_cli_matrix.run_captured_text",
        fake_run_captured_text,
    )
    server = RunningServer(
        base_url="http://127.0.0.1:9999",
        port=9999,
        log_path=tmp_path / "server.log",
        process=cast(subprocess.Popen[bytes], object()),
    )

    run = run_claude_cli(
        claude_bin="claude",
        server=server,
        config=_smoke_config(tmp_path),
        cwd=tmp_path / "workspace",
        prompt="hello",
        tools="",
    )
    outcome = make_outcome(
        model="z-ai/glm-5.2",
        full_model="nvidia_nim/z-ai/glm-5.2",
        source="nvidia_nim_cli_default",
        feature="basic_text",
        marker="FCC_NIM_BASIC",
        run=run,
        log_delta='POST /v1/messages HTTP/1.1" 200 OK',
        log_path=tmp_path / "server.log",
    )

    assert run.stdout == ""
    assert run.stderr == ""
    assert outcome.stdout_excerpt == ""
    assert outcome.stderr_excerpt == ""


def test_openrouter_free_cli_matrix_report_shape_and_redaction(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-openrouter-key")
    run = ClaudeCliRun(
        command=("claude", "-p", "redacted"),
        returncode=0,
        stdout="FCC_OPENROUTER_FREE_BASIC secret-openrouter-key",
        stderr="",
        duration_s=1.25,
    )
    outcome = make_outcome(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        full_model="open_router/nvidia/nemotron-3-super-120b-a12b:free",
        source="openrouter_free_cli_default",
        feature="basic_text",
        marker="FCC_OPENROUTER_FREE_BASIC",
        run=run,
        log_delta='POST /v1/messages HTTP/1.1" 200 OK secret-openrouter-key',
        log_path=tmp_path / "server.log",
    )

    path = write_matrix_report(
        _smoke_config(tmp_path),
        [outcome],
        target="openrouter_free_cli",
        filename_prefix="openrouter-free-cli",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name.startswith("openrouter-free-cli-matrix-test-worker-")
    assert payload["target"] == "openrouter_free_cli"
    assert payload["models"] == ["open_router/nvidia/nemotron-3-super-120b-a12b:free"]
    saved = payload["outcomes"][0]
    assert saved["feature"] == "basic_text"
    assert saved["classification"] == "passed"
    assert saved["request_count"] == 1
    assert saved["token_evidence"]["marker_present"] is True
    assert saved["token_evidence"]["agent_catalog_present"] is False
    assert saved["token_evidence"]["agent_tool_count"] == 0
    assert saved["token_evidence"]["agent_result_count"] == 0
    assert "secret-openrouter-key" not in path.read_text(encoding="utf-8")


def test_nvidia_nim_cli_matrix_regression_detection(tmp_path: Path) -> None:
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=0,
        stdout="",
        stderr="",
        duration_s=0.1,
    )
    outcome = make_outcome(
        model="z-ai/glm-5.2",
        full_model="nvidia_nim/z-ai/glm-5.2",
        source="nvidia_nim_cli_default",
        feature="basic_text",
        marker="FCC_NIM_BASIC",
        run=run,
        log_delta='POST /v1/messages HTTP/1.1" 500 Internal Server Error',
        log_path=tmp_path / "server.log",
    )

    assert outcome.classification == "product_failure"
    assert regression_failures([outcome]) == [
        "nvidia_nim/z-ai/glm-5.2 basic_text: product_failure"
    ]


def test_nvidia_nim_cli_matrix_model_feature_failures_do_not_regress(
    tmp_path: Path,
) -> None:
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=0,
        stdout="ordinary answer",
        stderr="",
        duration_s=0.1,
    )
    outcome = make_outcome(
        model="z-ai/glm-5.2",
        full_model="nvidia_nim/z-ai/glm-5.2",
        source="nvidia_nim_cli_default",
        feature="tool_use_roundtrip",
        marker="FCC_NIM_TOOL",
        run=run,
        log_delta='POST /v1/messages HTTP/1.1" 200 OK',
        log_path=tmp_path / "server.log",
        requires_tool_result=True,
    )

    assert outcome.classification == "model_feature_failure"
    assert regression_failures([outcome]) == []


def test_nvidia_nim_cli_raw_payload_log_counts_as_proxy_request(
    tmp_path: Path,
) -> None:
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=0,
        stdout="ordinary answer",
        stderr="",
        duration_s=0.1,
    )
    outcome = make_outcome(
        model="z-ai/glm-5.2",
        full_model="nvidia_nim/z-ai/glm-5.2",
        source="nvidia_nim_cli_default",
        feature="subagent_task",
        marker="FCC_NIM_TASK",
        run=run,
        log_delta="API_REQUEST: request_id=req_1 model=z-ai/glm-5.2 messages=2",
        log_path=tmp_path / "server.log",
        requires_task=True,
    )

    assert outcome.classification == "model_feature_failure"
    assert outcome.request_count == 1
    assert regression_failures([outcome]) == []


def test_cli_matrix_missing_agent_catalog_without_agent_use_is_model_limitation(
    tmp_path: Path,
) -> None:
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=0,
        stdout="ordinary answer",
        stderr="",
        duration_s=0.1,
    )
    outcome = make_outcome(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        full_model="open_router/nvidia/nemotron-3-super-120b-a12b:free",
        source="openrouter_free_cli_default",
        feature="subagent_task",
        marker="FCC_OPENROUTER_FREE_TASK",
        run=run,
        log_delta=(
            "API_REQUEST: request_id=req_1 model=nvidia/nemotron-3-super-120b-a12b:free "
            "messages=1\n"
            "FULL_PAYLOAD [req_1]: {'messages': [], 'tools': [{'name': 'Read'}], "
            "'tool_choice': None}"
        ),
        log_path=tmp_path / "server.log",
        requires_agent=True,
    )

    assert outcome.classification == "model_feature_failure"
    assert outcome.token_evidence["agent_catalog_present"] is False


def test_cli_matrix_agent_catalog_without_agent_use_is_model_feature_failure(
    tmp_path: Path,
) -> None:
    marker = "FCC_OPENROUTER_FREE_TASK"
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=0,
        stdout=(
            f'{marker}\n{{"type":"tool_use","name":"Read"}}\n{{"type":"tool_result"}}'
        ),
        stderr="",
        duration_s=0.1,
    )
    outcome = make_outcome(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        full_model="open_router/nvidia/nemotron-3-super-120b-a12b:free",
        source="openrouter_free_cli_default",
        feature="subagent_task",
        marker=marker,
        run=run,
        log_delta=(
            "API_REQUEST: request_id=req_1 model=nvidia/nemotron-3-super-120b-a12b:free "
            "messages=1\n"
            "FULL_PAYLOAD [req_1]: {'messages': [], 'tools': "
            "[{'name': 'Agent'}, {'name': 'Read'}], 'tool_choice': None}"
        ),
        log_path=tmp_path / "server.log",
        requires_tool_result=True,
        requires_agent=True,
    )

    assert outcome.classification == "model_feature_failure"
    assert outcome.token_evidence["agent_catalog_present"] is True
    assert outcome.token_evidence["agent_tool_count"] == 0


def test_cli_matrix_agent_use_result_and_marker_pass(tmp_path: Path) -> None:
    marker = "FCC_OPENROUTER_FREE_TASK"
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=0,
        stdout=(
            f'{marker}\n{{"type":"tool_use","name":"Agent"}}\n'
            '{"type":"tool_result","content":"agentId: abc123"}'
        ),
        stderr="",
        duration_s=0.1,
    )
    outcome = make_outcome(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        full_model="open_router/nvidia/nemotron-3-super-120b-a12b:free",
        source="openrouter_free_cli_default",
        feature="subagent_task",
        marker=marker,
        run=run,
        log_delta=(
            "API_REQUEST: request_id=req_1 model=nvidia/nemotron-3-super-120b-a12b:free "
            "messages=1\n"
            "FULL_PAYLOAD [req_1]: {'messages': [], 'tools': "
            "[{'name': 'Read'}], 'tool_choice': None}"
        ),
        log_path=tmp_path / "server.log",
        requires_tool_result=True,
        requires_agent=True,
    )

    assert outcome.classification == "passed"
    assert outcome.token_evidence["agent_catalog_present"] is False
    assert outcome.token_evidence["agent_tool_count"] == 1
    assert outcome.token_evidence["agent_result_count"] == 1


def test_cli_matrix_agent_prompt_text_without_tool_evidence_does_not_pass(
    tmp_path: Path,
) -> None:
    marker = "FCC_OPENROUTER_FREE_TASK"
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=0,
        stdout=f"{marker}\nAgent should read the file.",
        stderr="",
        duration_s=0.1,
    )
    outcome = make_outcome(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        full_model="open_router/nvidia/nemotron-3-super-120b-a12b:free",
        source="openrouter_free_cli_default",
        feature="subagent_task",
        marker=marker,
        run=run,
        log_delta=(
            "API_REQUEST: request_id=req_1 model=nvidia/nemotron-3-super-120b-a12b:free "
            "messages=1\n"
            "FULL_PAYLOAD [req_1]: {'messages': [], 'tools': "
            "[{'name': 'Agent'}, {'name': 'Read'}], 'tool_choice': None}"
        ),
        log_path=tmp_path / "server.log",
        requires_agent=True,
    )

    assert outcome.classification == "model_feature_failure"
    assert outcome.token_evidence["agent_catalog_present"] is True
    assert outcome.token_evidence["agent_tool_count"] == 0


def test_cli_matrix_unclassified_provider_error_is_model_feature_failure(
    tmp_path: Path,
) -> None:
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=0,
        stdout="Provider API request failed. (request_id=req_123)",
        stderr="",
        duration_s=0.1,
    )
    outcome = make_outcome(
        model="poolside/laguna-s-2.1:free",
        full_model="open_router/poolside/laguna-s-2.1:free",
        source="openrouter_free_cli_default",
        feature="tool_use_roundtrip",
        marker="FCC_OPENROUTER_FREE_TOOL",
        run=run,
        log_delta=(
            '{"event": "free_claude_code.api.request.received", "http_method": "POST", '
            '"http_path": "/v1/messages"}\n'
            '{"event": "provider.response.error", "exc_type": "HTTPStatusError"}'
        ),
        log_path=tmp_path / "server.log",
        requires_tool_result=True,
    )

    assert outcome.classification == "model_feature_failure"
    assert outcome.request_count == 1


def test_nvidia_nim_cli_timeout_is_not_model_missing(
    tmp_path: Path,
) -> None:
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=None,
        stdout='{"type":"assistant","content":[{"type":"text","text":"FCC_NIM_TOOL"}]}',
        stderr="",
        duration_s=45.0,
        timed_out=True,
    )
    outcome = make_outcome(
        model="z-ai/glm-5.2",
        full_model="nvidia_nim/z-ai/glm-5.2",
        source="nvidia_nim_cli_default",
        feature="tool_use_roundtrip",
        marker="FCC_NIM_TOOL",
        run=run,
        log_delta="API_REQUEST: request_id=req_1 model=z-ai/glm-5.2 messages=2",
        log_path=tmp_path / "server.log",
    )

    assert outcome.classification == "probe_timeout"
    assert outcome.token_evidence["timed_out"] is True
    assert regression_failures([outcome]) == []


def test_nvidia_nim_cli_success_beats_verbose_timeout_words(tmp_path: Path) -> None:
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=0,
        stdout="FCC_NIM_THINK",
        stderr="",
        duration_s=0.1,
    )
    outcome = make_outcome(
        model="z-ai/glm-5.2",
        full_model="nvidia_nim/z-ai/glm-5.2",
        source="nvidia_nim_cli_default",
        feature="thinking",
        marker="FCC_NIM_THINK",
        run=run,
        log_delta=(
            "API_REQUEST: request_id=req_1 model=z-ai/glm-5.2 messages=1 "
            "read_timeout_s=300"
        ),
        log_path=tmp_path / "server.log",
    )

    assert outcome.classification == "passed"
    assert outcome.request_count == 1


def test_cli_matrix_uuid_429_does_not_count_as_upstream_unavailable(
    tmp_path: Path,
) -> None:
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=0,
        stdout='{"uuid":"d3c76eea-3634-4299-aec0-e7634b3716da"}',
        stderr="",
        duration_s=0.1,
    )
    outcome = make_outcome(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        full_model="open_router/nvidia/nemotron-3-super-120b-a12b:free",
        source="openrouter_free_cli_default",
        feature="subagent_task",
        marker="FCC_OPENROUTER_FREE_TASK",
        run=run,
        log_delta=(
            "API_REQUEST: request_id=req_1 "
            "model=nvidia/nemotron-3-super-120b-a12b:free messages=2"
        ),
        log_path=tmp_path / "server.log",
        requires_task=True,
    )

    assert outcome.classification == "model_feature_failure"


def test_cli_matrix_real_http_429_counts_as_upstream_unavailable(
    tmp_path: Path,
) -> None:
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=0,
        stdout="ordinary answer",
        stderr="",
        duration_s=0.1,
    )
    outcome = make_outcome(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        full_model="open_router/nvidia/nemotron-3-super-120b-a12b:free",
        source="openrouter_free_cli_default",
        feature="subagent_task",
        marker="FCC_OPENROUTER_FREE_TASK",
        run=run,
        log_delta=(
            "API_REQUEST: request_id=req_1 model=nvidia/nemotron-3-super-120b-a12b:free "
            'messages=2 upstream HTTP/1.1" 429 Too Many Requests'
        ),
        log_path=tmp_path / "server.log",
        requires_task=True,
    )

    assert outcome.classification == "upstream_unavailable"


def test_cli_matrix_deterministic_provider_400_is_not_upstream_unavailable(
    tmp_path: Path,
) -> None:
    run = ClaudeCliRun(
        command=("claude", "-p", "x"),
        returncode=1,
        stdout="",
        stderr=(
            "Upstream provider GROQ returned HTTP 400: "
            "property reasoning_content is unsupported"
        ),
        duration_s=0.1,
    )
    outcome = make_outcome(
        model="openai/gpt-oss-120b",
        full_model="groq/openai/gpt-oss-120b",
        source="groq_cli_default",
        feature="thinking",
        marker="FCC_GROQ_THINK",
        run=run,
        log_delta="API_REQUEST: request_id=req_1 model=openai/gpt-oss-120b messages=2",
        log_path=tmp_path / "server.log",
    )

    assert outcome.classification == "model_feature_failure"


def test_cli_matrix_default_command_uses_bare_mode() -> None:
    command = _build_claude_cli_command(
        claude_bin="claude",
        prompt="hello",
        tools="Read",
    )

    assert command[:2] == ("claude", "--bare")
    assert "--tools" in command
    assert "Read" in command


def test_cli_matrix_auto_mode_exposes_tool_without_preapproval() -> None:
    command = _build_claude_cli_command(
        claude_bin="claude",
        prompt="read the marker",
        tools="Bash",
        auto_mode=True,
    )

    assert command[command.index("--permission-mode") + 1] == "auto"
    assert command[command.index("--tools") + 1] == "Bash"
    assert "--dangerously-skip-permissions" not in command
    assert "--allowedTools" not in command


def test_cli_matrix_subagent_command_uses_agent_without_bare_or_task() -> None:
    bare, tools, pre_tool_args, extra_args = _subagent_probe_options("{}")
    command = _build_claude_cli_command(
        claude_bin="claude",
        prompt="hello",
        tools=tools,
        bare=bare,
        pre_tool_args=pre_tool_args,
        extra_args=extra_args,
    )

    assert "--bare" not in command
    assert command[command.index("--setting-sources") + 1] == "local"
    assert "--strict-mcp-config" in command
    assert command[command.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert command[command.index("--tools") + 1] == "Agent,Read"
    assert command[command.index("--allowedTools") + 1] == "Agent,Read"
    assert command[command.index("--agents") + 1] == "{}"
    assert "Task,Read" not in command
