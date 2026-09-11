"""Read installed-client metadata without starting clients or installers."""

import json
import os
import plistlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_bytes())
    except OSError, ValueError:
        return None


def windows_codex_locations() -> tuple[Path, ...]:
    """The package identity survives changes to the desktop executable's name."""
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-AppxPackage -Name OpenAI.Codex | Select-Object -ExpandProperty InstallLocation | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        values = (
            json.loads(result.stdout)
            if result.returncode == 0 and result.stdout.strip()
            else []
        )
        if isinstance(values, str):
            values = [values]
        return (
            tuple(
                Path(value)
                for value in values
                if isinstance(value, str) and Path(value).is_absolute()
            )
            if isinstance(values, list)
            else ()
        )
    except OSError, ValueError, subprocess.TimeoutExpired:
        return ()


@dataclass(frozen=True)
class InstalledClients:
    claude_vscode: bool
    codex_vscode: bool
    codex_app: bool
    jetbrains: bool
    claude_command: Path | None
    fcc_command: Path | None
    acp_command: tuple[str, ...]
    scope_issue: str


@dataclass(frozen=True)
class LocalInstallations:
    home: Path
    platform: str
    environ: dict[str, str]
    python_bin: Path
    app_roots: tuple[Path, ...]

    @classmethod
    def current(cls) -> LocalInstallations:
        home, env = Path.home(), dict(os.environ)
        if sys.platform == "win32":
            roots = [
                Path(value)
                for key in ("ProgramFiles", "ProgramFiles(x86)")
                if (value := env.get(key))
            ]
            if local := env.get("LOCALAPPDATA"):
                roots.extend(
                    (Path(local) / "Programs", Path(local) / "JetBrains/Toolbox/apps")
                )
        elif sys.platform == "darwin":
            roots = [home / "Applications", Path("/Applications")]
        else:
            roots = [
                Path("/usr/share"),
                Path("/opt"),
                home / ".local/share/JetBrains/Toolbox/apps",
            ]
        return cls(home, sys.platform, env, Path(sys.executable).parent, tuple(roots))

    @property
    def vscode_settings(self) -> Path:
        if self.platform == "win32":
            root = Path(self.environ.get("APPDATA", str(self.home / "AppData/Roaming")))
        elif self.platform == "darwin":
            root = self.home / "Library/Application Support"
        else:
            root = Path(self.environ.get("XDG_CONFIG_HOME", str(self.home / ".config")))
        return root / "Code/User/settings.json"

    def command(self, name: str) -> Path | None:
        directories = [
            Path(value)
            for value in self.environ.get("PATH", "").split(os.pathsep)
            if value
        ]
        if name == "fcc-codex":
            directories.insert(0, self.python_bin)
        suffixes = (".exe", ".cmd", "") if self.platform == "win32" else ("",)
        for directory in directories:
            for suffix in suffixes:
                candidate = directory / (name + suffix)
                if candidate.is_file() and (
                    self.platform == "win32" or os.access(candidate, os.X_OK)
                ):
                    return candidate.resolve()
        return None

    def _vscode_installed(self) -> bool:
        for root in self.app_roots:
            if self.platform == "win32":
                app = root / "Microsoft VS Code"
                product, executable = (
                    app / "resources/app/product.json",
                    app / "Code.exe",
                )
            elif self.platform == "darwin":
                app = root / "Visual Studio Code.app/Contents"
                product, executable = (
                    app / "Resources/app/product.json",
                    app / "MacOS/Electron",
                )
            else:
                app = root / "code"
                product, executable = app / "resources/app/product.json", app / "code"
            metadata = _json(product)
            if (
                isinstance(metadata, dict)
                and metadata.get("applicationName") == "code"
                and executable.is_file()
            ):
                return True
        return False

    def _extensions(self) -> set[str]:
        root = self.home / ".vscode/extensions"
        index = _json(root / "extensions.json")
        result: set[str] = set()
        if not isinstance(index, list):
            return result
        for descriptor in index:
            if not isinstance(descriptor, dict):
                continue
            identity, relative = (
                descriptor.get("identifier"),
                descriptor.get("relativeLocation"),
            )
            if not isinstance(identity, dict) or not isinstance(relative, str):
                continue
            identifier = identity.get("id")
            if identifier not in {"anthropic.claude-code", "openai.chatgpt"}:
                continue
            package = (root / relative).resolve()
            if not package.is_relative_to(root.resolve()):
                continue
            manifest = _json(package / "package.json")
            if (
                isinstance(manifest, dict)
                and f"{manifest.get('publisher')}.{manifest.get('name')}" == identifier
            ):
                result.add(identifier)
        return result

    def _codex_app(self) -> bool:
        if self.platform == "win32":
            for root in windows_codex_locations():
                try:
                    manifest = ElementTree.parse(root / "AppxManifest.xml").getroot()
                    identity = next(
                        (
                            node
                            for node in manifest.iter()
                            if node.tag.rsplit("}", 1)[-1] == "Identity"
                        ),
                        None,
                    )
                    if identity is None or identity.get("Name") != "OpenAI.Codex":
                        continue
                    for node in manifest.iter():
                        if node.tag.rsplit("}", 1)[-1] == "Application" and (
                            relative := node.get("Executable")
                        ):
                            executable = (root / relative).resolve()
                            if (
                                executable.is_relative_to(root.resolve())
                                and executable.is_file()
                            ):
                                return True
                except OSError, ElementTree.ParseError:
                    continue
            return False
        if self.platform == "darwin":
            for root in self.app_roots:
                for bundle in root.glob("*.app"):
                    try:
                        metadata = plistlib.loads(
                            (bundle / "Contents/Info.plist").read_bytes()
                        )
                        executable = metadata.get("CFBundleExecutable")
                        if metadata.get(
                            "CFBundleIdentifier"
                        ) == "com.openai.codex" and isinstance(executable, str):
                            path = bundle / "Contents/MacOS" / executable
                            if (
                                path.resolve().is_relative_to(bundle.resolve())
                                and path.is_file()
                            ):
                                return True
                    except OSError, ValueError, plistlib.InvalidFileException:
                        continue
            return False
        return self.command("chatgpt") is not None

    def _jetbrains(self) -> bool:
        for root in self.app_roots:
            for pattern in (
                "*/product-info.json",
                "JetBrains/*/product-info.json",
                "*/*/product-info.json",
                "*.app/Contents/Resources/product-info.json",
            ):
                for path in root.glob(pattern):
                    metadata = _json(path)
                    if not isinstance(metadata, dict) or metadata.get(
                        "productCode"
                    ) not in {
                        "IU",
                        "IC",
                        "PY",
                        "PC",
                        "WS",
                        "PS",
                        "GO",
                        "CL",
                        "RD",
                        "RM",
                        "RR",
                        "DB",
                    }:
                        continue
                    launches = metadata.get("launch", [])
                    if isinstance(launches, list):
                        for launch in launches:
                            if isinstance(launch, dict) and isinstance(
                                relative := launch.get("launcherPath"), str
                            ):
                                base = (
                                    path.parent.parent
                                    if self.platform == "darwin"
                                    else path.parent
                                )
                                executable = (base / relative).resolve()
                                if (
                                    executable.is_relative_to(base.resolve())
                                    and executable.is_file()
                                ):
                                    return True
        return False

    def _acp_command(self) -> tuple[str, ...]:
        node = self.command("node")
        if node is None or node.suffix.lower() == ".cmd":
            return ()
        roots = [node.parent.parent / "lib/node_modules"]
        if self.platform == "win32":
            roots.append(
                Path(self.environ.get("APPDATA", str(self.home / "AppData/Roaming")))
                / "npm/node_modules"
            )
        adapter = self.command("claude-agent-acp")
        candidates = [root / "@agentclientprotocol/claude-agent-acp" for root in roots]
        if adapter is not None:
            candidates.extend(adapter.resolve().parents[:3])
            candidates.append(
                adapter.parent / "node_modules/@agentclientprotocol/claude-agent-acp"
            )
        found: set[Path] = set()
        for package in candidates:
            metadata = _json(package / "package.json")
            if (
                not isinstance(metadata, dict)
                or metadata.get("name") != "@agentclientprotocol/claude-agent-acp"
            ):
                continue
            entry = metadata.get("bin")
            if isinstance(entry, dict):
                entry = entry.get("claude-agent-acp")
            if isinstance(entry, str):
                executable = (package / entry).resolve()
                if (
                    executable.is_relative_to(package.resolve())
                    and executable.is_file()
                ):
                    found.add(executable)
        return (str(node), str(next(iter(found)))) if len(found) == 1 else ()

    def scan(self) -> InstalledClients:
        scope = ""
        if self.environ.get("WSL_DISTRO_NAME") or self.environ.get("WSL_INTEROP"):
            scope = "WSL setup is manual. Use FCC and the client in the same supported native environment."
        extensions = self._extensions() if self._vscode_installed() else set()
        return InstalledClients(
            "anthropic.claude-code" in extensions,
            "openai.chatgpt" in extensions,
            self._codex_app(),
            self._jetbrains(),
            self.command("claude"),
            self.command("fcc-codex"),
            self._acp_command(),
            scope,
        )
