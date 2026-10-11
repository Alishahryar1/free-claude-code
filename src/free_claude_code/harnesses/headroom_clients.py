"""Native CLI registration contracts used only by the bundle installer."""

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import cast

from packaging.version import Version

from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses.headroom_mcp import (
    McpSetupError,
    Registration,
    json_registration,
    read_object,
)

AGENTS = (
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


class NativeClients:
    """Resolve native user scopes without changing FCC's launch configuration."""

    def __init__(self, headroom: Path, env: Mapping[str, str], cwd: Path) -> None:
        self.headroom = headroom
        self.env = dict(env)
        self.cwd = cwd
        self.outcomes: list[str] = []

    def run(
        self,
        args: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout: float = 30,
    ) -> str:
        try:
            with subprocess.Popen(
                args,
                cwd=self.cwd,
                env=dict(env if env is not None else self.env),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            ) as process:
                try:
                    stdout, _ = process.communicate(timeout=timeout)
                finally:
                    if process.poll() is None:
                        if os.name == "nt":
                            subprocess.run(
                                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW,
                                timeout=5,
                            )
                        else:
                            with suppress(ProcessLookupError):
                                os.killpg(process.pid, signal.SIGKILL)
                        process.kill()
                        process.communicate()
        except OSError, subprocess.TimeoutExpired:
            raise McpSetupError(
                f"Could not run {Path(args[0]).name} for MCP setup. Check the native installation and retry."
            ) from None
        if process.returncode:
            raise McpSetupError(
                f"{Path(args[0]).name} MCP setup command failed (exit {process.returncode}). Check the native client's configuration and policy before retrying."
            )
        return stdout.strip()

    def root(self, variable: str, default: Path) -> Path:
        value = self.env.get(variable, "").strip()
        return Path(value).expanduser().absolute() if value else default

    def _skip(self, agent: str, reason: str) -> None:
        self.outcomes.append(f"{agent}: skipped ({reason})")

    def _supports(
        self, binary: str, command: Sequence[str], flags: Sequence[str]
    ) -> bool:
        try:
            output = self.run([binary, *command, "--help"])
        except McpSetupError:
            return False
        return all(flag in output for flag in flags)

    def _version_at_least(self, binary: str, minimum: str) -> bool:
        output = self.run([binary, "--version"])
        match = re.search(r"\b(\d+\.\d+\.\d+)\b", output)
        return match is not None and Version(match[1]) >= Version(minimum)

    def prepare(self, agent: str) -> Registration | None:
        if agent not in AGENTS:
            raise McpSetupError(f"Unknown coding agent: {agent}")
        if agent == "aider":
            self._skip(agent, "no supported native MCP client")
            return None
        binary = shutil.which(agent, path=self.env.get("PATH"))
        if binary is None:
            self._skip(agent, "native command is not available")
            return None
        home = Path.home()
        entry: JsonObject = {"command": str(self.headroom), "args": ["mcp", "serve"]}
        args = [str(self.headroom), "mcp", "serve"]
        table = ("mcpServers",)
        command = [binary, "mcp", "add", "headroom", "--", *args]
        if agent == "pi":
            if not self._version_at_least(binary, "0.99.0"):
                self._skip(agent, "Headroom setup requires Pi 0.99.0 or newer")
                return None
            path = self.root("PI_CODING_AGENT_DIR", home / ".pi/agent") / "mcp.json"
        elif agent == "muse":
            if not self._version_at_least(binary, "1.3.0"):
                self._skip(agent, "Headroom setup supports Muse 1.3.0 or newer")
                return None
            return self._muse()
        elif agent == "dsh":
            from free_claude_code.harnesses.headroom_dsh import dsh_registration

            target = dsh_registration(
                self.headroom,
                self.root("DSH_HOME", home / ".dsh"),
                lambda: self.run([binary, "--profile", "web", "--dump-config"]),
            )
            if target is None:
                self._skip("dsh", "native web profile does not exist")
            return target
        elif agent == "hermes":
            return self._hermes(binary)
        elif agent == "claude":
            if not self._supports(binary, ["mcp", "add"], ["--scope", "--transport"]):
                self._skip(agent, "native user-scope MCP registration is unavailable")
                return None
            path = self.root("CLAUDE_CONFIG_DIR", home) / ".claude.json"
            command = [
                binary,
                "mcp",
                "add",
                "--scope",
                "user",
                "--transport",
                "stdio",
                "headroom",
                "--",
                *args,
            ]
        elif agent == "codex":
            if not self._supports(binary, ["mcp", "add"], ["--"]):
                self._skip(agent, "native MCP registration is unavailable")
                return None
            path = self.root("CODEX_HOME", home / ".codex") / "config.toml"
            table = ("mcp_servers",)
        elif agent == "opencode":
            if not self._supports(binary, ["mcp", "add"], ["--global"]):
                self._skip(
                    agent, "native OpenCode 2 global MCP registration is unavailable"
                )
                return None
            root = self.root(
                "OPENCODE_CONFIG_DIR",
                self.root("XDG_CONFIG_HOME", home / ".config") / "opencode",
            )
            candidates = [
                root / name
                for name in (
                    "opencode.json",
                    "opencode.jsonc",
                    ".opencode/opencode.json",
                    ".opencode/opencode.jsonc",
                )
            ]
            path = next((path for path in candidates if path.exists()), candidates[0])
            table = ("mcp", "servers")
            entry = {"type": "local", "command": args}
            command = [binary, "mcp", "add", "headroom", "--global", "--", *args]
        elif agent == "cline":
            if not self._supports(binary, ["mcp", "install"], ["--yes", "--json"]):
                self._skip(
                    agent, "native noninteractive MCP registration is unavailable"
                )
                return None
            data = self.root(
                "CLINE_DATA_DIR", self.root("CLINE_DIR", home / ".cline") / "data"
            )
            path = self.root(
                "CLINE_MCP_SETTINGS_PATH", data / "settings/cline_mcp_settings.json"
            )
            entry = {"transport": {"type": "stdio", **entry}}
            command = [
                binary,
                "mcp",
                "install",
                "headroom",
                "--yes",
                "--json",
                "--",
                *args,
            ]
        else:
            assert agent == "grok"
            if not self._supports(
                binary, ["mcp", "add"], ["--scope"]
            ) or not self._supports(binary, ["mcp", "list"], ["--json"]):
                self._skip(agent, "native user-scope MCP registration is unavailable")
                return None
            path = self.root("GROK_HOME", home / ".grok") / "config.toml"
            table = ("mcp_servers",)
            command = [binary, "mcp", "add", "--scope", "user", "headroom", "--", *args]

        def write() -> None:
            self.run(command)

        target = json_registration(
            agent, path, table, self.headroom, expected=entry, write=write
        )
        if agent == "grok":

            def read_grok() -> JsonObject | None:
                target.read()  # Validate the user file before native merge/defaults.
                try:
                    servers = json.loads(self.run([binary, "mcp", "list", "--json"]))
                except ValueError:
                    raise McpSetupError(
                        "Grok returned invalid MCP configuration."
                    ) from None
                if not isinstance(servers, list) or any(
                    not isinstance(server, dict) for server in servers
                ):
                    raise McpSetupError("Grok returned an invalid MCP server list.")
                return next(
                    (server for server in servers if server.get("name") == "headroom"),
                    None,
                )

            return Registration(agent, path, entry, read_grok, write)
        return target

    def _muse(self) -> Registration:
        path = (
            self.root("XDG_CONFIG_HOME", Path.home() / ".config") / "muse/settings.json"
        )
        document = read_object(path)
        if "mcpServers" in document and "mcp_servers" in document:
            raise McpSetupError(
                f"Muse has conflicting MCP aliases at {path}. Correct them before retrying."
            )
        legacy = "mcp_servers" in document
        entry: JsonObject = {"command": str(self.headroom), "args": ["mcp", "serve"]}
        entry.update(
            {"transport": "stdio", "mode": "optional"}
            if legacy
            else {"type": "stdio", "required": False}
        )
        return json_registration(
            "muse",
            path,
            ("mcp_servers" if legacy else "mcpServers",),
            self.headroom,
            expected=entry,
            defaults={"schema_version": 1},
        )

    def _hermes(self, binary: str) -> Registration | None:
        if not self._supports(binary, ["config", "get"], ["--json"]):
            self._skip("hermes", "structured native config commands are unavailable")
            return None
        probe: JsonObject = {
            "command": "fcc-mcp-probe",
            "args": ["one", "two"],
            "enabled": True,
        }
        with tempfile.TemporaryDirectory(prefix="fcc-hermes-check-") as folder:
            env = dict(self.env)
            env["HERMES_HOME"] = folder
            env.pop("HERMES_PROFILE", None)
            env.pop("HERMES_MANAGED_DIR", None)
            try:
                self.run(
                    [
                        binary,
                        "config",
                        "set",
                        "mcp_servers.fcc_probe",
                        json.dumps(probe),
                    ],
                    env=env,
                )
                observed = json.loads(
                    self.run(
                        [binary, "config", "get", "mcp_servers.fcc_probe", "--json"],
                        env=env,
                    )
                )
            except McpSetupError, ValueError:
                observed = None
        if observed != probe:
            self._skip("hermes", "native structured MCP configuration is unsupported")
            return None
        path = Path(self.run([binary, "config", "path"]))
        if not path.is_absolute():
            raise McpSetupError(
                "Hermes did not return an absolute native configuration path."
            )
        expected: JsonObject = {
            "command": str(self.headroom),
            "args": ["mcp", "serve"],
            "enabled": True,
        }

        def read() -> JsonObject | None:
            # Use the native profile-aware reader without persisting merged defaults.
            source = self.run([binary, "config", "get", "mcp_servers", "--json"])
            try:
                document = json.loads(source)
            except ValueError:
                raise McpSetupError(
                    "Hermes returned invalid MCP configuration."
                ) from None
            if document is None:
                return None
            if not isinstance(document, dict):
                raise McpSetupError("Hermes returned an invalid MCP server map.")
            if "headroom" not in document:
                return None
            if not isinstance(document["headroom"], dict):
                raise McpSetupError("Hermes headroom configuration must be an object.")
            return cast(JsonObject, document["headroom"])

        def write() -> None:
            self.run(
                [binary, "config", "set", "mcp_servers.headroom", json.dumps(expected)]
            )

        return Registration("hermes", path, expected, read, write)
