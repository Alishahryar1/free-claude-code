"""Read installed-client metadata and probe Node without starting clients or installers."""

import json
import os
import plistlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Timer
from time import monotonic
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


type NodeVersion = tuple[int, int, int]


@dataclass(frozen=True)
class AcpInstallation:
    command: tuple[str, ...] = ()
    manifest: Path | None = None
    node_version: NodeVersion | None = None
    issue: str = ""


@dataclass(frozen=True)
class InstalledClients:
    claude_vscode: bool
    codex_vscode: bool
    codex_app: bool
    jetbrains: bool
    claude_command: Path | None
    fcc_command: Path | None
    acp: AcpInstallation
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
            override = self.environ.get("XDG_CONFIG_HOME", "")
            root = (
                Path(override)
                if override and Path(override).is_absolute()
                else self.home / ".config"
            )
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
        patterns = (
            ("*.app/Contents/Resources/product-info.json",)
            if self.platform == "darwin"
            else (
                "*/product-info.json",
                "JetBrains/*/product-info.json",
                "*/*/product-info.json",
            )
        )
        for root in self.app_roots:
            for pattern in patterns:
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
                                if (
                                    Path(relative).is_absolute()
                                    or Path(relative).anchor
                                ):
                                    continue
                                boundary = (
                                    path.parent.parent
                                    if self.platform == "darwin"
                                    else path.parent
                                )
                                executable = (path.parent / relative).resolve()
                                if (
                                    executable.is_relative_to(boundary.resolve())
                                    and executable.is_file()
                                ):
                                    return True
        return False

    def _node_version(self, node: Path) -> NodeVersion | None:
        try:
            node = node.resolve(strict=True)
            expected_name = "node.exe" if self.platform == "win32" else "node"
            if node.name.lower() != expected_name or not node.is_file():
                return None
            parent = node.parent
            if (
                "shims" in {part.lower() for part in node.parts}
                or parent.name.lower() in {"volta", ".volta"}
                or (
                    parent.name.lower() == "bin"
                    and parent.parent.name.lower() in {"volta", ".volta"}
                )
            ):
                return None
            if (volta := self.environ.get("VOLTA_HOME")) and parent == (
                Path(volta) / "bin"
            ).resolve():
                return None
            with node.open("rb") as executable:
                header = executable.read(4)
            native = (
                header.startswith(b"MZ")
                if self.platform == "win32"
                else header
                in {
                    b"\xfe\xed\xfa\xce",
                    b"\xce\xfa\xed\xfe",
                    b"\xfe\xed\xfa\xcf",
                    b"\xcf\xfa\xed\xfe",
                    b"\xca\xfe\xba\xbe",
                    b"\xbe\xba\xfe\xca",
                    b"\xca\xfe\xba\xbf",
                    b"\xbf\xba\xfe\xca",
                }
                if self.platform == "darwin"
                else header == b"\x7fELF"
            )
            if not native:
                return None
            environment = {"LC_ALL": "C", "LANG": "C"}
            if self.platform == "win32":
                environment.update(
                    (key, value)
                    for key, value in self.environ.items()
                    if key.upper() in {"SYSTEMROOT", "WINDIR", "SYSTEMDRIVE"}
                )
            with subprocess.Popen(
                [str(node), "--version"],
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                cwd=self.home,
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ) as process:
                deadline = monotonic() + 2
                timer = Timer(2, process.kill)
                timer.start()
                try:
                    assert process.stdout is not None
                    output = bytearray()
                    while len(output) <= 256:
                        chunk = process.stdout.read(257 - len(output))
                        if not chunk:
                            break
                        output.extend(chunk)
                    if len(output) > 256:
                        return None
                    if process.wait() != 0 or monotonic() >= deadline:
                        return None
                finally:
                    timer.cancel()
                    process.kill()
                    process.wait()
                    timer.join()
            match = re.fullmatch(rb"v([0-9]+)\.([0-9]+)\.([0-9]+)\r?\n", output)
            if match is not None:
                return int(match[1]), int(match[2]), int(match[3])
        except OSError:
            return None
        return None

    def _acp_installation(self) -> AcpInstallation:
        missing = "Install the Claude ACP adapter and Node.js 22 or newer. A compatible installed adapter was not detected."
        node = self.command("node")
        if node is None or node.suffix.lower() == ".cmd":
            return AcpInstallation(issue=missing)
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
        found: dict[Path, tuple[Path, object]] = {}
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
                    engines = metadata.get("engines")
                    found[executable] = (
                        (package / "package.json").resolve(),
                        engines.get("node") if isinstance(engines, dict) else None,
                    )
        if len(found) != 1:
            return AcpInstallation(issue=missing)
        executable, (manifest, requirement) = next(iter(found.items()))
        minimum = (
            re.fullmatch(
                r">=\s*(0|[1-9][0-9]*)(?:\.(0|[1-9][0-9]*))?(?:\.(0|[1-9][0-9]*))?",
                requirement.strip(),
            )
            if isinstance(requirement, str) and len(requirement.strip()) <= 64
            else None
        )
        if minimum is None:
            return AcpInstallation(
                manifest=manifest,
                issue="The installed ACP adapter has an unsupported Node requirement. Use manual setup.",
            )
        required = max(
            (22, 0, 0),
            (int(minimum[1]), int(minimum[2] or 0), int(minimum[3] or 0)),
        )
        version = self._node_version(node)
        if version is None:
            return AcpInstallation(
                manifest=manifest,
                issue="Could not verify a direct Node.js 22+ runtime. Use an installed Node executable or manual setup; wrappers and unresolved runtime managers are unsupported.",
            )
        if version < required:
            return AcpInstallation(
                manifest=manifest,
                node_version=version,
                issue=f"Detected Node.js {'.'.join(map(str, version))}; this ACP adapter needs {'.'.join(map(str, required))} or newer.",
            )
        return AcpInstallation((str(node), str(executable)), manifest, version)

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
            AcpInstallation(issue=scope) if scope else self._acp_installation(),
            scope,
        )
