"""Launcher tests exercise public entrypoints without native processes or network."""

import io
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import Request

import pytest

from free_claude_code.cli import local_http
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.core.json_types import JsonObject


class JsonResponse(io.BytesIO):
    status = 200


@dataclass
class LaunchCapture:
    requests: list[Request] = field(default_factory=list)
    probes: list[list[str]] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    environments: list[dict[str, str]] = field(default_factory=list)
    catalog: JsonObject = field(
        default_factory=lambda: {
            "data": [
                {
                    "id": "nvidia_nim/catalog-model:variant",
                    "provider_model_ref": "nvidia_nim/catalog-model:variant",
                    "supportsReasoning": False,
                    "inputModalities": ["text", "image"],
                    "contextWindow": 32000,
                    "maxCompletionTokens": 4096,
                }
            ],
        }
    )
    catalog_error: Exception | None = None
    health_error: Exception | None = None
    on_start: Callable[[list[str], Mapping[str, str]], None] | None = None
    exit_code: int = 23
    versions: dict[str, str] = field(
        default_factory=lambda: {
            "pi": "0.80.4",
            "opencode": "1.18.18",
            "cline": "3.0.55",
            "hermes": "Hermes Agent 0.20.4",
            "dsh": "0.1.0-rc.8",
            "grok": '{"currentVersion":"1.0.5"}',
            "muse": "Muse Code 1.0.3 (1.0.3-R2198.1)",
        }
    )

    def open(self, request: Request, *, timeout: float) -> JsonResponse:
        self.requests.append(request)
        if request.full_url.endswith("/health"):
            if self.health_error:
                raise self.health_error
            return JsonResponse(b"{}")
        assert "/v1/models?view=" in request.full_url
        assert request.get_header("Authorization") == "Bearer launcher-test-token"
        if self.catalog_error:
            raise self.catalog_error
        return JsonResponse(json.dumps(self.catalog).encode())

    def probe(
        self, command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.probes.append(command)
        if command[1:] == ["config", "get", "model.provider", "--json"]:
            env = kwargs["env"]
            assert isinstance(env, Mapping)
            config = json.loads(
                (Path(env["HERMES_MANAGED_DIR"]) / "config.yaml").read_text()
            )
            return subprocess.CompletedProcess(
                command, 0, json.dumps(config["model"]["provider"]), ""
            )
        return subprocess.CompletedProcess(command, 0, self.versions[command[0]], "")

    def start(self, command: list[str], *, env: Mapping[str, str]) -> object:
        self.commands.append(command)
        self.environments.append(dict(env))
        if self.on_start:
            self.on_start(command, env)
        capture = self

        class Process:
            pid = 0

            def wait(self) -> int:
                return capture.exit_code

        return Process()


@pytest.fixture
def launch_capture(monkeypatch: pytest.MonkeyPatch) -> LaunchCapture:
    from free_claude_code.cli.launchers import common

    capture = LaunchCapture()
    store = ManagedConfigStore()
    store.initialize({})
    store.commit(
        dict(store.read({}).managed) | {"ANTHROPIC_AUTH_TOKEN": "launcher-test-token"}
    )
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8182")
    monkeypatch.setenv("MODEL", "nvidia_nim/catalog-model:variant")
    for key in (
        "OPENCODE_CONFIG",
        "OPENCODE_CONFIG_CONTENT",
        "GROK_CONFIG",
        "GROK_CONFIG_FILE",
        "HERMES_MANAGED_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(common.shutil, "which", lambda name: name)
    monkeypatch.setattr(local_http._DIRECT_OPENER, "open", capture.open)
    monkeypatch.setattr(subprocess, "run", capture.probe)
    monkeypatch.setattr(subprocess, "Popen", capture.start)
    return capture
