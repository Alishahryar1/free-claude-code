"""Shareable local diagnostics, without starting or changing FCC."""

import importlib
import json
import platform
import re
import shutil
import subprocess
from typing import Any
from urllib.parse import urlsplit

from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.paths import (
    claude_desktop_disconnect_path,
    codex_model_catalog_path,
)
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG, ProviderAuthKind
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.core.version import package_version
from free_claude_code.harnesses import (
    claude_desktop_integration,
    claude_integration,
    codex_integration,
    jetbrains_acp_integration,
)
from free_claude_code.providers.github_copilot.auth import CopilotAuthManager
from free_claude_code.providers.openai_codex.auth import saved_connection_state
from free_claude_code.providers.runtime.config import has_provider_configuration
from free_claude_code.providers.runtime.discovery import referenced_provider_ids

RUNTIME_FIELDS = (
    "host",
    "port",
    "open_admin_browser",
    "proxy_auth_enabled",
    "provider_rate_limit",
    "provider_rate_window",
    "provider_max_concurrency",
    "provider_progress_timeout",
    "http_read_timeout",
    "http_write_timeout",
    "http_connect_timeout",
    "fast_prefix_detection",
    "enable_network_probe_mock",
    "enable_title_generation_skip",
    "enable_suggestion_mode_skip",
    "enable_filepath_extraction_mock",
    "log_level",
    "log_raw_api_payloads",
    "log_raw_sse_events",
    "log_api_error_tracebacks",
    "log_raw_messaging_content",
    "log_raw_cli_diagnostics",
    "log_messaging_error_details",
    "debug_platform_edits",
    "debug_subagent_stack",
    "voice_note_enabled",
    "whisper_device",
    "whisper_model",
    "messaging_platform",
    "messaging_rate_limit",
    "messaging_rate_window",
    "max_message_log_entries_per_chat",
)
ROUTING_FIELDS = (
    "model",
    "model_fable",
    "model_opus",
    "model_sonnet",
    "model_haiku",
    "model_fallbacks",
)
REASONING_FIELDS = (
    "reasoning_policy",
    "reasoning_fable",
    "reasoning_opus",
    "reasoning_sonnet",
    "reasoning_haiku",
)
WEB_FIELDS = (
    "enable_web_server_tools",
    "web_fetch_allowed_schemes",
    "web_fetch_allow_private_networks",
)
NIM_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "min_p",
    "repetition_penalty",
    "seed",
    "parallel_tool_calls",
    "ignore_eos",
    "min_tokens",
)
EXCLUDED_FIELDS = {
    "proxy_auth_token",
    "telegram_bot_token",
    "allowed_telegram_user_id",
    "discord_bot_token",
    "allowed_discord_channels",
    "allowed_dir",
    "cloudflare_account_id",
    "vertex_project_id",
} | {d.credential_attr for d in PROVIDER_CATALOG.values() if d.credential_attr}
URL_FIELDS = {
    attr
    for d in PROVIDER_CATALOG.values()
    for attr in (d.base_url_attr, d.proxy_attr)
    if attr
} | {"telegram_proxy_url"}
HARNESS_NAMES = (
    "claude",
    "codex",
    "pi",
    "opencode",
    "cline",
    "hermes",
    "dsh",
    "grok",
    "muse",
    "aider",
)
_VERSION = re.compile(
    r"(?<![\w./\\])v?(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)(?![\w./\\+-])"
)


def classified_settings() -> set[str]:
    return (
        set(RUNTIME_FIELDS + ROUTING_FIELDS + REASONING_FIELDS + WEB_FIELDS)
        | EXCLUDED_FIELDS
        | URL_FIELDS
        | {"nim", "vertex_location", "custom_providers"}
    )


def _url_summary(value: str | None) -> dict[str, Any]:
    origin = None
    if value:
        try:
            parsed = urlsplit(value)
            if (
                parsed.scheme in {"http", "https", "socks5", "socks5h", "socks4"}
                and parsed.hostname
            ):
                host = parsed.hostname
                if ":" in host:
                    host = f"[{host}]"
                origin = f"{parsed.scheme}://{host}"
                if parsed.port is not None:
                    origin += f":{parsed.port}"
        except ValueError:
            pass
    return {"configured": bool(value), "origin": origin}


def settings_report(settings: Settings, accounts: dict[str, str]) -> dict[str, Any]:
    referenced = referenced_provider_ids(settings)
    providers = {}
    for name, descriptor in sorted(PROVIDER_CATALOG.items()):
        account = accounts.get(name, "unavailable")
        configured = has_provider_configuration(descriptor, settings)
        if descriptor.auth_kind is ProviderAuthKind.CONNECTED_ACCOUNT:
            configured = account == "connected" if account != "unavailable" else None
        row = {"configured": configured, "referenced": name in referenced}
        if descriptor.auth_kind is ProviderAuthKind.CONNECTED_ACCOUNT:
            row["saved_connection"] = account or "unavailable"
        for key, attr in (
            ("base_url", descriptor.base_url_attr),
            ("proxy", descriptor.proxy_attr),
        ):
            if attr:
                row[key] = _url_summary(getattr(settings, attr))
        if name == "vertex":
            row["location"] = settings.vertex_location
        providers[name] = row
    for definition in settings.custom_providers:
        providers[definition.provider_id] = {
            "display_name": definition.display_name,
            "configured": True,
            "referenced": definition.provider_id in referenced,
            "base_url": _url_summary(definition.base_url),
            "api_key_configured": bool(definition.api_key),
            "api_format": definition.api_format,
            "reasoning_format": definition.reasoning_format,
            "reasoning_history_format": definition.reasoning_history_format,
            "model_ids": list(definition.model_ids),
        }
    runtime = {name: getattr(settings, name) for name in RUNTIME_FIELDS}
    runtime["telegram_proxy"] = _url_summary(settings.telegram_proxy_url)
    runtime["nim"] = {name: getattr(settings.nim, name) for name in NIM_FIELDS}
    return {
        "providers": providers,
        "runtime": runtime,
        "model_routing": {name: getattr(settings, name) for name in ROUTING_FIELDS},
        "reasoning": {name: getattr(settings, name) for name in REASONING_FIELDS},
        "web_tools": {name: getattr(settings, name) for name in WEB_FIELDS},
    }


def integration_report(settings: Settings) -> dict[str, Any]:
    url, token = local_proxy_root_url(settings), settings.proxy_auth_token
    readers = {
        "claude-vscode": lambda: claude_integration.configure(
            claude_integration.settings_path(),
            claude_integration.claude_state_path(),
            url,
            token,
        ),
        "codex": lambda: codex_integration.configure(
            codex_integration.config_path(), codex_model_catalog_path(), url
        ),
        "claude-desktop": lambda: claude_desktop_integration.configure(
            claude_desktop_integration.config_root(),
            url,
            token,
            disconnect_path=claude_desktop_disconnect_path(),
        ),
        "jetbrains-acp": lambda: jetbrains_acp_integration.configure(
            jetbrains_acp_integration.config_path(), url, token
        ),
    }
    result: dict[str, Any] = {}
    for name, read in readers.items():
        try:
            status = read()
            result[name] = {"status": "available", "connected": status["connected"]}
            if "disconnect_pending" in status:
                result[name]["disconnect_pending"] = status["disconnect_pending"]
        except OSError, ValueError:
            result[name] = {"status": "unavailable", "connected": None}
    return result


def probe_harness(binary: str, args: tuple[str, ...]) -> dict[str, Any]:
    path = shutil.which(binary)
    result = {"installed": path is not None, "version": None, "status": "not_installed"}
    if path is None:
        return result
    try:
        probe = subprocess.run(
            [path, *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if probe.returncode != 0:
            result["status"] = "failed"
            return result
        output = probe.stdout.strip()
        if binary == "grok":
            payload = json.loads(output)
            output = (
                payload.get("currentVersion", "") if isinstance(payload, dict) else ""
            )
        else:
            output = f"{output}\n{probe.stderr}"
        match = _VERSION.search(output) if isinstance(output, str) else None
        result["version"] = match[1] if match else None
        result["status"] = "available" if match else "unrecognized_version"
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except OSError:
        result["status"] = "unavailable"
    except ValueError:
        result["status"] = "unrecognized_version"
    return result


def harness_report() -> dict[str, Any]:
    result = {}
    for name in HARNESS_NAMES:
        spec = importlib.import_module(f"free_claude_code.cli.launchers.{name}").SPEC
        args = (
            tuple(spec.compatibility_check.args)
            if spec.compatibility_check
            else ("--version",)
        )
        result[name] = probe_harness(spec.binary_name, args)
    return result


def collect_report() -> dict[str, Any]:
    report = {
        "version": package_version(),
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "configuration": {
            "source": "saved_config_with_process_overrides",
            "status": "available",
        },
        "providers": None,
        "runtime": None,
        "model_routing": None,
        "reasoning": None,
        "web_tools": None,
        "integrations": None,
        "harnesses": {},
        "errors": [],
    }
    store = ManagedConfigStore()
    try:
        if not store.path.is_file():
            raise FileNotFoundError
        settings = store.read().settings
    except FileNotFoundError:
        report["configuration"]["status"] = "missing"
    except ValueError:
        report["configuration"]["status"] = "invalid"
    except OSError:
        report["configuration"]["status"] = "unavailable"
    else:
        accounts = {"openai": saved_connection_state()}
        copilot = CopilotAuthManager().status().state.value
        accounts["github_copilot"] = (
            copilot if copilot in {"connected", "disconnected"} else "unavailable"
        )
        report.update(settings_report(settings, accounts))
        report["integrations"] = integration_report(settings)
        for name, state in accounts.items():
            if state not in {"connected", "disconnected"}:
                report["errors"].append(
                    {"section": f"providers.{name}", "code": "unavailable"}
                )
        for name, status in report["integrations"].items():
            if status["status"] != "available":
                report["errors"].append(
                    {"section": f"integrations.{name}", "code": "unavailable"}
                )
    if report["configuration"]["status"] != "available":
        report["errors"].append(
            {"section": "configuration", "code": report["configuration"]["status"]}
        )
    report["harnesses"] = harness_report()
    return report
