import json
import os
import subprocess
import sys

import httpx
import pytest

from free_claude_code.cli import doctor
from free_claude_code.config import paths
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.providers.openai_codex import auth as openai_auth
from free_claude_code.runtime import diagnostics


@pytest.fixture(autouse=True)
def offline_discovery_status(monkeypatch):
    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("private-canary")

    monkeypatch.setattr(httpx, "get", unavailable)


def test_live_discovery_report_projects_only_safe_current_errors(monkeypatch):
    requests = []
    message = "The network proxy rejected or failed the connection. Check your proxy settings."

    def get(url, **kwargs):
        requests.append((url, kwargs))
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "status": "running",
                "token": "private-canary",
                "startup": {
                    "providers": {
                        "nvidia_nim": "failed",
                        "open_router": "ready",
                        "lmstudio": "unavailable",
                        "llamacpp": "unavailable",
                        "ollama": "unavailable",
                        "private-canary": "failed",
                    },
                    "provider_errors": {
                        "nvidia_nim": message,
                        "open_router": "private-canary",
                        "ollama": "private-canary",
                    },
                },
            },
        )

    monkeypatch.setattr(httpx, "get", get)
    result = diagnostics.model_discovery_report(Settings())
    assert result["status"] == "available"
    assert result["providers"]["nvidia_nim"] == {"state": "failed", "message": message}
    assert result["providers"]["open_router"] == {"state": "ready"}
    for provider_id in ("lmstudio", "llamacpp", "ollama"):
        assert result["providers"][provider_id] == {"state": "unavailable"}
    assert "private-canary" not in json.dumps(result)
    assert requests == [
        (
            "http://127.0.0.1:8082/admin/api/status",
            {
                "timeout": 2.0,
                "trust_env": False,
                "follow_redirects": False,
            },
        )
    ]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"status": "running"},
        {"status": "running", "startup": {"providers": []}},
    ],
)
def test_missing_or_incompatible_live_status_is_unavailable(monkeypatch, payload):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            200, request=httpx.Request("GET", url), json=payload
        ),
    )
    assert diagnostics.model_discovery_report(Settings())["status"] == "unavailable"


@pytest.mark.parametrize(
    "message",
    [
        "private-canary",
        "The model-list request returned HTTP 403. private-canary",
        {},
        None,
    ],
)
def test_doctor_does_not_trust_arbitrary_server_failure_text(monkeypatch, message):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "status": "running",
                "startup": {
                    "providers": {"nvidia_nim": "failed"},
                    "provider_errors": {"nvidia_nim": message},
                },
            },
        ),
    )
    result = diagnostics.model_discovery_report(Settings())
    assert result["providers"]["nvidia_nim"] == {
        "state": "failed",
        "message": "Could not load models. Check the provider's settings and retry.",
    }
    assert "private-canary" not in json.dumps(result)


@pytest.mark.parametrize(
    "status, body",
    [(302, "private-canary"), (500, "private-canary"), (200, "private-canary")],
)
def test_unusable_local_http_response_is_unavailable(monkeypatch, status, body):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            status,
            request=httpx.Request("GET", url),
            text=body,
            headers={"Location": "https://private-canary.test"},
        ),
    )
    result = diagnostics.model_discovery_report(Settings())
    assert result["status"] == "unavailable"
    assert "private-canary" not in json.dumps(result)


def test_discovery_status_failure_does_not_expose_exception_or_contact_remote_host(
    monkeypatch,
):
    assert diagnostics.model_discovery_report(Settings())["status"] == "unavailable"

    def forbidden(*args, **kwargs):
        pytest.fail("doctor must only contact the local server")

    monkeypatch.setattr(httpx, "get", forbidden)
    assert (
        diagnostics.model_discovery_report(Settings(HOST="remote.test"))["status"]
        == "unavailable"
    )


@pytest.mark.parametrize(
    "host", ["http://127.0.0.1", "[::1", "[::1]:8082", "local\nhost"]
)
def test_malformed_host_still_produces_a_doctor_report(monkeypatch, host):
    ManagedConfigStore().initialize({})
    monkeypatch.setenv("HOST", host)

    # Exercise real URL validation without connecting to a user-owned server.
    def validate_url(url, **kwargs):
        httpx.Request("GET", url)
        raise httpx.ConnectError("unavailable")

    monkeypatch.setattr(httpx, "get", validate_url)
    monkeypatch.setattr(diagnostics, "harness_report", lambda: {})
    monkeypatch.setattr(diagnostics, "integration_report", lambda settings: {})
    report = diagnostics.collect_report()
    assert report["configuration"]["status"] == "available"
    assert report["runtime"]["host"] == host
    assert report["model_discovery"]["status"] == "unavailable"
    assert json.dumps(report)


def test_missing_configuration_still_produces_report(monkeypatch):
    monkeypatch.setattr(diagnostics, "harness_report", lambda: {})
    report = diagnostics.collect_report()
    assert next(iter(report)) == "version"
    assert report["configuration"]["status"] == "missing"
    assert report["runtime"] is None
    assert report["providers"] is None
    assert report["system"]["python_version"]


def test_settings_projection_excludes_secrets_and_url_credentials():
    settings = Settings(
        OPENAI_API_KEY="secret-api-key",
        ANTHROPIC_API_KEY="secret-anthropic-key",
        ANTHROPIC_WORKSPACE_ID="secret-workspace",
        ANTHROPIC_AUTH_TOKEN="secret-proxy-token",
        TELEGRAM_BOT_TOKEN="secret-bot-token",
        ALLOWED_TELEGRAM_USER_ID="secret-user-id",
        LM_STUDIO_BASE_URL="http://user:secret-password@localhost:1234/private?key=secret-query",
        MODEL="openai_api/gpt-example",
    )
    result = diagnostics.settings_report(settings, {})
    encoded = json.dumps(result)
    assert "secret-" not in encoded
    assert "/private" not in encoded
    assert result["providers"]["openai_api"]["configured"] is True
    assert result["providers"]["openai_api"]["referenced"] is True
    assert (
        result["providers"]["lmstudio"]["base_url"]["origin"] == "http://localhost:1234"
    )
    assert result["model_routing"]["model"] == "openai_api/gpt-example"


def test_all_settings_have_an_explicit_reporting_policy():
    assert diagnostics.classified_settings() == set(Settings.model_fields)


def test_collection_does_not_initialize_or_expose_validation_input(monkeypatch):
    store = ManagedConfigStore()
    store.path.parent.mkdir(parents=True)
    store.path.write_text("FCC_CONFIG_SCHEMA=invalid\nOPENAI_API_KEY=secret-value\n")

    def forbidden(*args, **kwargs):
        pytest.fail("doctor must not initialize configuration")

    monkeypatch.setattr(ManagedConfigStore, "initialize", forbidden)
    monkeypatch.setattr(diagnostics, "harness_report", lambda: {})
    report = diagnostics.collect_report()
    assert report["configuration"]["status"] == "invalid"
    assert "secret-value" not in json.dumps(report)


def test_complete_report_reads_real_files_without_mutation(
    monkeypatch, tmp_path, capsys
):
    store = ManagedConfigStore()
    store.initialize({})
    store.commit(
        dict(store.read({}).managed)
        | {"MODEL": "openai_api/saved", "ANTHROPIC_AUTH_TOKEN": "managed-secret"}
    )
    monkeypatch.setenv("MODEL", "openai_api/process")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ignored-secret")
    settings = store.read().settings
    assert settings.proxy_auth_token == "managed-secret"
    claude = diagnostics.claude_integration
    claude.configure(
        claude.settings_path(),
        local_proxy_root_url(settings),
        settings.proxy_auth_token,
        connected=True,
    )
    copilot_path = paths.github_copilot_auth_path()
    copilot_path.parent.mkdir(parents=True)
    copilot_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "enabled": True,
                "revision": 1,
                "identity": {"host": "github.com", "login": "private-account"},
            }
        )
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("diagnostics must not initialize config or construct HTTP clients")

    monkeypatch.setattr(ManagedConfigStore, "initialize", forbidden)
    monkeypatch.setattr(openai_auth.httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(diagnostics.shutil, "which", lambda binary: None)
    copied = []
    monkeypatch.setattr(doctor, "copy_text", copied.append)
    doctor.main([])
    stdout = capsys.readouterr().out
    report = json.loads(stdout)
    assert copied == [stdout]
    assert report["model_routing"]["model"] == "openai_api/process"
    assert report["integrations"]["claude-vscode"]["connected"] is True
    assert report["providers"]["github_copilot"]["saved_connection"] == "connected"
    assert report["errors"] == []
    assert report["model_discovery"]["status"] == "unavailable"
    assert all(not item["installed"] for item in report["harnesses"].values())
    for private in (
        "managed-secret",
        "ignored-secret",
        "private-account",
        str(tmp_path),
    ):
        assert private not in stdout
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_every_credential_and_private_identifier_is_excluded():
    values = dict.fromkeys(diagnostics.EXCLUDED_FIELDS, "private-canary")
    encoded = json.dumps(
        diagnostics.settings_report(Settings.model_validate(values), {})
    )
    assert "private-canary" not in encoded


def test_one_unreadable_integration_does_not_hide_other_sections(monkeypatch):
    store = ManagedConfigStore()
    store.initialize({})
    codex_path = diagnostics.codex_integration.config_path()
    codex_path.parent.mkdir(parents=True)
    codex_path.write_text("not valid toml private-content")
    monkeypatch.setattr(diagnostics.shutil, "which", lambda binary: None)
    report = diagnostics.collect_report()
    assert report["runtime"] is not None
    assert report["integrations"]["codex"]["connected"] is None
    assert report["integrations"]["claude-vscode"]["status"] == "available"
    assert "private-content" not in json.dumps(report)


@pytest.mark.parametrize(
    "banner, redirect", [("", ""), ("echo private-banner\n", ""), ("", " >&2")]
)
def test_native_version_probe_works_with_platform_command_shim(
    monkeypatch, tmp_path, banner, redirect
):
    suffix = ".cmd" if sys.platform == "win32" else ""
    binary = tmp_path / ("doctor-test-harness" + suffix)
    binary.write_text(
        ("@echo off\n" if suffix else "#!/bin/sh\n")
        + banner
        + f"echo 1.2.3{redirect}\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    assert diagnostics.probe_harness("doctor-test-harness", ("--version",)) == {
        "installed": True,
        "version": "1.2.3",
        "status": "available",
    }


def test_command_prints_and_copies_identical_json(monkeypatch, capsys):
    copied = []
    monkeypatch.setattr(
        doctor, "collect_report", lambda: {"version": "1.2.3", "errors": []}
    )
    monkeypatch.setattr(doctor, "copy_text", copied.append)
    doctor.main([])
    captured = capsys.readouterr()
    assert copied == [captured.out]
    assert captured.out.startswith('{\n  "version": "1.2.3"')
    assert json.loads(captured.out)["errors"] == []


def test_clipboard_failure_leaves_valid_stdout(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "collect_report", lambda: {"version": "1.2.3"})

    def unavailable(text):
        raise doctor.ClipboardUnavailable

    monkeypatch.setattr(doctor, "copy_text", unavailable)
    doctor.main([])
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"version": "1.2.3"}
    assert "manually" in captured.err


@pytest.mark.parametrize(
    "failure, status",
    [
        (subprocess.TimeoutExpired("stub", 5), "timeout"),
        (OSError("secret-path"), "unavailable"),
    ],
)
def test_version_probe_failure_is_bounded_and_sanitized(monkeypatch, failure, status):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda binary: "stub")

    def run(args, **kwargs):
        assert kwargs["timeout"] == 5
        assert kwargs["stdin"] == subprocess.DEVNULL
        raise failure

    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    result = diagnostics.probe_harness("claude", ("--version",))
    assert result == {"installed": True, "version": None, "status": status}


@pytest.mark.parametrize(
    "name, output, stderr, expected",
    [
        ("claude", "2.1.7 (Claude Code)", "secret-error", "2.1.7"),
        ("dsh", "0.1.0-rc.8", "secret-error", "0.1.0-rc.8"),
        ("dsh", "0.1.0-rc.8+build.1", "secret-error", "0.1.0-rc.8+build.1"),
        (
            "grok",
            '{"currentVersion":"1.0.5","path":"secret-path"}',
            "secret-warning 9.8.7",
            "1.0.5",
        ),
        ("grok", '{"path":"secret-path"}', "secret-warning 9.8.7", None),
        ("claude", "unknown secret-output", "secret-error", None),
    ],
)
def test_probe_returns_only_version(monkeypatch, name, output, stderr, expected):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda binary: "stub")
    monkeypatch.setattr(
        diagnostics.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, output, stderr),
    )
    result = diagnostics.probe_harness(name, ("--version",))
    assert result["version"] == expected
    assert "secret-" not in json.dumps(result)


def test_clipboard_adapter_maps_backend_failure(monkeypatch):
    from free_claude_code.cli import clipboard

    def unavailable(text):
        raise clipboard.pyperclip.PyperclipException("no desktop")

    monkeypatch.setattr(clipboard.pyperclip, "copy", unavailable)
    with pytest.raises(clipboard.ClipboardUnavailable):
        clipboard.copy_text("diagnostic text")


def test_doctor_version_does_not_collect_or_copy(monkeypatch, capsys):
    from free_claude_code.cli import entrypoints

    def forbidden():
        pytest.fail("version must not collect or copy a report")

    monkeypatch.setattr(doctor, "collect_report", forbidden)
    monkeypatch.setattr(entrypoints, "package_version", lambda: "1.2.3")
    entrypoints.doctor(["--version"])
    assert capsys.readouterr().out == "free-claude-code 1.2.3\n"


def test_fallback_model_efforts_are_json_serializable_in_doctor_report():
    entries = [
        {"model": "groq/one", "reasoning": "low"},
        {"model": "open_router/vendor/two", "reasoning": "client"},
    ]
    settings = Settings(MODEL_FALLBACKS=entries)
    report = json.loads(json.dumps(diagnostics.settings_report(settings, {})))
    assert report["model_routing"]["model_fallbacks"] == [
        entry | {"api_key_number": 1} for entry in entries
    ]
    assert report["providers"]["groq"]["referenced"] is True
    assert report["providers"]["open_router"]["referenced"] is True


def test_disconnect_mode_does_not_collect_or_copy_diagnostics(monkeypatch, capsys):
    from free_claude_code.harnesses import integration_cleanup

    calls = []
    monkeypatch.setattr(
        integration_cleanup,
        "disconnect_integrations",
        lambda: calls.append("disconnect"),
    )

    def forbidden(*args):
        pytest.fail("cleanup must not collect diagnostics or use the clipboard")

    monkeypatch.setattr(doctor, "collect_report", forbidden)
    monkeypatch.setattr(doctor, "copy_text", forbidden)
    doctor.main(["--disconnect-integrations"])
    assert calls == ["disconnect"]
    assert capsys.readouterr() == (
        "Disconnected recognized FCC editor/app integrations.\n",
        "",
    )


def test_disconnect_mode_reports_failure_and_nonzero_exit(monkeypatch, capsys):
    from free_claude_code.harnesses import integration_cleanup

    def fail():
        raise integration_cleanup.IntegrationCleanupError("Could not disconnect Codex.")

    monkeypatch.setattr(integration_cleanup, "disconnect_integrations", fail)
    with pytest.raises(SystemExit) as error:
        doctor.main(["--disconnect-integrations"])
    assert error.value.code == 1
    assert capsys.readouterr() == ("", "Could not disconnect Codex.\n")
