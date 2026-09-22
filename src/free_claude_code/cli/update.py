"""Prepare a uv update, then exit before the native host replaces FCC."""

import argparse
import ast
import csv
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ARCHIVE_URL = (
    "https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"
)
PACKAGE = "free-claude-code"
FCC_COMMANDS = frozenset(
    [
        "fcc-desktop",
        "fcc-server",
        "fcc-claude",
        "fcc-codex",
        "fcc-pi",
        "fcc-opencode",
        "fcc-cline",
        "fcc-hermes",
        "fcc-dsh",
        "fcc-grok",
        "fcc-muse",
        "fcc-aider",
        "fcc-init",
        "fcc-update",
        "free-claude-code",
    ]
)


def capture(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"Command failed: {args[0]}")
    return result.stdout.strip()


def installed_version() -> str:
    return importlib.metadata.version(PACKAGE)


def running_fcc() -> list[str]:
    if os.name == "nt":
        rows = csv.reader(capture(["tasklist.exe", "/FO", "CSV", "/NH"]).splitlines())
        return [
            f"{row[0]} (PID {row[1]})"
            for row in rows
            if len(row) >= 2 and row[0].lower().removesuffix(".exe") in FCC_COMMANDS
        ]
    pattern = re.compile(r"(?:^|/)(" + "|".join(sorted(FCC_COMMANDS)) + r")(?:\s|$)")
    found = []
    for line in capture(["ps", "-A", "-o", "pid=", "-o", "args="]).splitlines():
        pid, _, args = line.strip().partition(" ")
        if (
            pid.isdecimal()
            and int(pid) not in {os.getpid(), os.getppid()}
            and (match := pattern.search(args.strip()))
        ):
            found.append(f"{match[1]} (PID {pid})")
    return found


def torch_build_version() -> str:
    distribution = importlib.metadata.distribution("torch")
    version = distribution.version
    if "+" in version:
        return version
    # PyPI metadata can omit the backend that the installed build records.
    # Inspect literal build metadata without importing torch or loading its DLLs.
    source = Path(str(distribution.locate_file("torch/version.py"))).read_text(
        encoding="utf-8"
    )
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if (
            any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in targets
            )
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            return value.value
    return version


def torch_backend(uv: Path, options: dict) -> str | None:
    backend = options.get("torch-backend")
    if backend is None:
        try:
            build = torch_build_version()
        except (OSError, SyntaxError, importlib.metadata.PackageNotFoundError) as exc:
            raise ValueError(
                "Cannot identify local voice build; rerun the matching voice installer."
            ) from exc
        _, separator, backend = build.partition("+")
        if (
            not separator
            and sys.platform == "darwin"
            and re.fullmatch(r"\d+\.\d+\.\d+", build)
        ):
            return None
        if not separator or not re.fullmatch(
            r"cpu|cu\d+|rocm\d+(?:\.\d+)+|xpu", backend
        ):
            raise ValueError(
                "Cannot identify local voice backend; rerun the matching voice installer with --torch-backend."
            )
    if not isinstance(backend, str) or not backend:
        raise ValueError(
            "Invalid recorded voice backend; rerun the matching voice installer."
        )
    capture([str(uv), "tool", "install", "--torch-backend", backend, "--help"])
    return backend


def prepare(uv: Path, launcher: Path) -> dict:
    version = capture([str(uv), "--version"])
    match = re.match(r"uv (\d+)\.(\d+)\.(\d+)(?:\s|$)", version)
    if not match or tuple(map(int, match.groups())) < (0, 12, 13):
        raise ValueError("uv 0.12.13 or newer is required. Rerun the FCC installer.")
    root = Path(capture([str(uv), "tool", "dir"])) / PACKAGE
    bin_dir = Path(capture([str(uv), "tool", "dir", "--bin"]))
    if not root.samefile(sys.prefix) or not Path(__file__).resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError(
            "This is not the uv-managed FCC installation. Use its original installation method."
        )
    receipt = tomllib.loads((root / "uv-receipt.toml").read_text(encoding="utf-8"))[
        "tool"
    ]
    requirements = receipt.get("requirements", [])
    if (
        len(requirements) != 1
        or requirements[0].get("name") != PACKAGE
        or requirements[0].get("url") != ARCHIVE_URL
    ):
        raise ValueError(
            "fcc-update supports the official FCC installer source. Use this installation's original method."
        )
    entry_name = "fcc-update.cmd" if os.name == "nt" else "fcc-update"
    entries = [
        entry
        for entry in receipt.get("entrypoints", [])
        if entry.get("name") == entry_name and entry.get("from") == PACKAGE
    ]
    if (
        len(entries) != 1
        or not launcher.samefile(bin_dir / entry_name)
        or not launcher.samefile(entries[0]["install-path"])
    ):
        raise ValueError("The updater launcher belongs to a different installation.")
    extras = sorted(
        {extra.replace("_", "-") for extra in requirements[0].get("extras", [])}
    )
    if set(extras) - {"voice", "voice-local"}:
        raise ValueError("Unsupported FCC extras. Use the original installer.")
    python_request = receipt.get("python")
    if not isinstance(python_request, str) or not python_request:
        raise ValueError(
            "The installed Python request is missing. Rerun the FCC installer."
        )
    if processes := running_fcc():
        raise ValueError(
            f"Stop running FCC processes before updating: {', '.join(processes)}"
        )
    arguments = [
        "tool",
        "install",
        "--force",
        "--refresh-package",
        PACKAGE,
        "--python",
        python_request,
    ]
    if "voice-local" in extras and (
        backend := torch_backend(uv, receipt.get("options", {}))
    ):
        arguments += ["--torch-backend", backend]
    spec = PACKAGE + (f"[{','.join(extras)}]" if extras else "")
    arguments.append(f"{spec} @ {ARCHIVE_URL}")
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return {
        "uv": str(uv),
        "arguments": arguments,
        "version": installed_version(),
        "verify": [
            str(python),
            "-I",
            "-c",
            "from importlib.metadata import version; print('FCC refreshed (version ' + version('free-claude-code') + ')')",
        ],
    }


def shell_worker(plan: dict) -> str:
    fd, name = tempfile.mkstemp(prefix="fcc-update-", suffix=".sh")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                "#!/bin/sh\nset -eu\ntrap 'rm -f -- \"$0\"' EXIT\n"
                "trap 'exit 130' INT\ntrap 'exit 143' TERM HUP\n"
                + "printf '%s\\n' "
                + shlex.quote(f"Updating FCC {plan['version']}...")
                + "\n"
                + shlex.join([plan["uv"], *plan["arguments"]])
                + "\n"
                + shlex.join(plan["verify"])
                + "\n"
            )
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    try:
        plan = prepare(args.uv, args.launcher)
        print(shell_worker(plan) if args.shell else json.dumps(plan))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        importlib.metadata.PackageNotFoundError,
    ) as exc:
        print(f"fcc-update: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
