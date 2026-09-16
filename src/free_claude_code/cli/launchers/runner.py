"""One setup, execution, and cleanup path for installed native harnesses."""

import os
import re
import secrets
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Literal

from free_claude_code.application.model_catalog import ModelCatalog
from free_claude_code.config.loader import get_settings
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .catalog_http import fetch_proxy_model_catalog
from .common import preflight_proxy, resolve_client_binary, run_client_process


@dataclass(frozen=True, slots=True)
class LaunchContext:
    binary_path: str
    settings: Settings = field(repr=False)
    proxy_root_url: str
    auth_token: str = field(repr=False)
    base_env: Mapping[str, str] = field(repr=False)
    catalog: ModelCatalog | None
    launch_id: str = field(default_factory=lambda: secrets.token_hex(16))

    def require_catalog(self) -> ModelCatalog:
        if self.catalog is None:
            raise ValueError("harness configuration requires a model catalog")
        return self.catalog


@dataclass(frozen=True, slots=True)
class HarnessSpec:
    binary_name: str
    display_name: str
    install_hint: str
    configure: Callable[[LaunchContext, list[str], LaunchResources], PreparedLaunch]
    catalog_view: Literal["messages", "responses"] | None = None
    compatibility_check: NativeCheck | None = None


class LaunchError(Exception):
    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def version_at_least(
    output: str, pattern: re.Pattern[str], minimum: tuple[int, int, int]
) -> bool:
    match = pattern.search(output)
    return match is not None and tuple(map(int, match.groups())) >= minimum


def _check_native(
    binary_path: str,
    check: NativeCheck,
    env: Mapping[str, str],
    *,
    exit_code: int,
    install_hint: str = "",
) -> None:
    try:
        result = subprocess.run(
            [binary_path, *check.args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=check.timeout_seconds,
            env=dict(env),
        )
        if result.returncode == 0 and check.accepts(result.stdout):
            return
    except OSError, subprocess.TimeoutExpired:
        pass
    message = check.failure_message
    if install_hint:
        message += f"\n{install_hint}"
    raise LaunchError(message, exit_code)


def launch_harness(spec: HarnessSpec, argv: Sequence[str] | None = None) -> None:
    """Prepare the FCC connection and leave command semantics to the harness."""

    args = list(sys.argv[1:] if argv is None else argv)
    base_env = dict(os.environ)
    auth_token = base_env.get("ANTHROPIC_AUTH_TOKEN", "")
    stage = "load settings"
    try:
        binary_path = resolve_client_binary(
            binary_name=spec.binary_name,
            display_name=spec.display_name,
            install_hint=spec.install_hint,
        )
        if spec.compatibility_check:
            _check_native(
                binary_path,
                spec.compatibility_check,
                base_env,
                exit_code=126,
                install_hint=spec.install_hint,
            )
        settings = get_settings()
        auth_token = settings.proxy_auth_token.strip()
        if not auth_token:
            raise LaunchError("Free Claude Code proxy authentication token is empty.")
        proxy_root_url = local_proxy_root_url(settings)
        if error := preflight_proxy(proxy_root_url):
            raise LaunchError(
                f"Free Claude Code proxy is not reachable at {proxy_root_url}: {error}\n"
                "Start it in another terminal with: fcc-server"
            )
        catalog = None
        if spec.catalog_view is not None:
            stage = "prepare model catalog"
            catalog = fetch_proxy_model_catalog(
                proxy_root_url, auth_token, view=spec.catalog_view
            )
        context = LaunchContext(
            binary_path, settings, proxy_root_url, auth_token, base_env, catalog
        )
        with ExitStack() as stack:
            stage = "prepare configuration"
            prepared = spec.configure(context, args, LaunchResources(stack))
            if prepared.activation_check:
                _check_native(
                    binary_path, prepared.activation_check, prepared.env, exit_code=1
                )
            stage = "start process"
            run_client_process(
                command=prepared.command,
                env=prepared.env,
                binary_name=spec.binary_name,
                display_name=spec.display_name,
                install_hint=spec.install_hint,
            )
    except (LaunchError, OSError, ValueError) as exc:
        message = (
            str(exc)
            if isinstance(exc, LaunchError)
            else f"Could not {stage} for {spec.display_name}: {exc}"
        )
        if auth_token:
            message = message.replace(auth_token, "[redacted]")
        print(message, file=sys.stderr)
        raise SystemExit(exc.exit_code if isinstance(exc, LaunchError) else 1) from None
