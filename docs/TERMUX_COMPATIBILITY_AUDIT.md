# Termux compatibility audit

## Scope and evidence

This audit covers FCC's Python package and command launchers, install/update
paths, filesystem and process behavior, local server, and dependencies for
native Android Termux on AArch64/Bionic. The inspected repository is the current
checkout. Current Termux package recipes advertise Python 3.14.10 and uv
0.13.7; those upstream recipe versions are evidence of package availability,
not evidence that FCC has run on Android hardware.

**Verification status:** static source review and host automated tests only.
No Android device, Termux emulator, or Android/Bionic ARM64 runner is available
in this environment. No entry below should be read as a real Termux integration
test.

## Findings

| Severity | Finding | Action | Verification |
|---|---|---|---|
| High | Package metadata pinned Python to exactly 3.14.10, rejecting later compatible patch releases. FCC source also uses Python 3.14 exception syntax. | Changed metadata/lock policy to `>=3.14,<3.15`; kept the supported minor aligned to source syntax and Termux Python. | Static metadata check and host package tests; not run on Termux. |
| High | The main dependency set pulled NVIDIA Riva, `grpcio`, and `grpcio-tools`, despite Riva being imported only when voice transcription is requested. These compiled components have no Android wheel guarantee. | Keep Riva and its gRPC dependency chain off Android kernel releases; surface an actionable runtime message if NIM voice is selected without Riva. | Marker and diagnostics tests on host; source builds not verified on Android. |
| Medium | `fastapi[standard]` selected optional server/desktop-oriented extras (including native runtime optimizations) that do not have a guaranteed Bionic wheel. | Use plain FastAPI on Android and preserve the existing standard extra elsewhere. | Lock/marker checks on host; Android resolver behavior not verified. |
| Medium | The existing shell installer is designed around the uv tool manager and managed CPython release, while Termux needs its native Python and writable Termux prefix. | Added a separate, idempotent native installer using `pkg`, Termux Python/uv, a private venv, and guarded command links. | POSIX syntax, help, and non-Termux guard tests on host; installation not run on Android. |
| Medium | The regular doctor reports general host/runtime details but did not check Android/Bionic, Termux paths, localhost, or Termux dependencies. | Added `fcc-doctor --termux`, including platform, dependency, storage, loopback, configured port, launcher, browser, and configuration checks. | Unit tests on host with controlled platform probes; live Android diagnostics not verified. |
| Low | Clipboard and browser helpers are desktop-centric; Termux may not provide a clipboard backend or a registered browser command. | Termux doctor avoids clipboard use and advises manual localhost browser access when `termux-open-url` is absent. | Static path and unit test coverage; device UI handoff not verified. |
| Medium | The general server uses Python sockets, SQLite, POSIX signals/locks, and `~/.fcc`; those primitives are present in Termux but their behavior has not been exercised on Bionic. | Keep generic implementations and add loopback/config-path diagnostics rather than introducing a duplicate Android server. | Static analysis only for Android; host regression suite remains applicable. |
| High | Third-party coding-agent binaries are independently distributed and may not support Android or Bionic. FCC cannot make that compatibility claim. | Installer installs FCC only. Launchers remain available, and diagnostics inspect agent executables on `PATH`; docs state agent support is unverified. | Launcher guardrails and host tests only; individual agents not verified on Termux. |

## Dependency compatibility

| Dependency/area | Role in FCC | Termux assessment |
|---|---|---|
| FastAPI, Uvicorn, AnyIO, HTTPX, Pydantic, SQLAlchemy, JSON/TOML libraries | Core HTTP server, requests, config, and persistence | Primarily Python packages, but `pydantic-core` and some transitive modules use native code. Termux installer provides Clang and Rust for source builds. Not built on device here. |
| `tiktoken` | Improved token estimates; `core/token_estimation.py` already falls back to approximate counts if initialization fails | Excluded on Android because the inspected lock has no Android wheel; approximate token estimates are used instead. The dependency remains enabled on non-Android platforms. |
| `aiohttp`, `cryptography`, `pydantic-core` and transitive native modules | Runtime HTTP, Google auth, validation | Android wheels are not assumed. Native builds depend on Termux's Clang/Rust/OpenSSL/pkg-config setup and remain unverified. |
| `nvidia-riva-client`, `grpcio`, `grpcio-tools` | NVIDIA NIM voice transcription | Excluded when `platform_release` contains `android`; this is an optional voice path, not required to start the FCC server. NIM voice is not included in the Termux install. |
| `fastapi[standard]` extras | Optional server enhancements and convenience integrations | Kept for non-Android platforms; Termux resolves plain FastAPI to avoid depending on optional native server wheels. |
| `pystray`, Pillow | Desktop UI and icon rendering | Already limited to Windows/macOS by platform markers; Termux uses the server and browser UI, not the tray application. |
| Telegram, Discord, OpenAI, Copilot SDK | Provider and optional messaging integrations | Their primary Python distributions are platform-neutral, but transitive modules and any external CLIs need runtime verification. Core server startup should not activate disabled messaging backends. |
| `voice_local` extra (PyTorch/Transformers/Librosa) | Local Whisper transcription | Not installed by the Termux installer; CUDA/desktop wheel indexes do not establish native Termux support. |

`uv.lock` records reproducible package versions/hashes but cannot provide
Android wheels where upstream publishes none. Termux source builds must still
be validated on an actual device. The platform markers are necessary but are
not proof that all remaining dependencies compile on Bionic.

## Platform review

- **Python:** source uses Python 3.14 syntax. The Termux support policy is
  `>=3.14,<3.15`; the exact release currently advertised by the Termux recipe
  is 3.14.10. Lowering the minor version would require a deliberate syntax and
  dependency compatibility project.
- **Paths:** FCC configuration uses `Path.home() / ".fcc"` and install paths
  use `$HOME`, `$PREFIX`, and a private home venv. No `/usr/bin`,
  `/usr/local/bin`, `/etc`, or systemd assumption is added to the Termux
  installer.
- **Process and signals:** FCC uses `subprocess`, POSIX process sessions,
  signals, and advisory file locks. Termux is POSIX-like, but descendant
  process and signal behavior needs real device testing.
- **Networking:** the server uses Uvicorn sockets and defaults from existing
  FCC settings. Termux doctor tests a loopback connection and configured port;
  Android VPN/firewall and external-client reachability are not covered by
  host tests.
- **SQLite:** FCC uses Python SQLite/SQLAlchemy and user-owned database files.
  The Termux Python package includes SQLite support, but database migration,
  contention, and shutdown behavior need a device run.
- **Terminal/UI:** command output is terminal-based. `webbrowser` may not find
  an Android browser; the Termux guide permits manual use of the localhost URL.
  The desktop tray is intentionally unsupported.
- **Updates:** Termux `fcc-update` upgrades the package inside its venv with
  uv. The standard installer/updater remains unchanged for existing desktop
  and Linux installations.
- **Launchers:** FCC launches external commands through `PATH`. No coding-agent
  binary has been verified for Android ARM64; FCC installation is independent
  of agent installation.

## Verification matrix

| Check | Status |
|---|---|
| Source/platform review | VERIFIED THROUGH STATIC ANALYSIS |
| Python 3.14 policy and marker tests | VERIFIED THROUGH STATIC ANALYSIS / HOST AUTOMATED TESTS |
| Termux detection and doctor unit tests | VERIFIED THROUGH HOST AUTOMATED TESTS |
| Installer shell syntax/help/guard tests | VERIFIED THROUGH HOST AUTOMATED TESTS |
| Linux ARM64 | NOT VERIFIED |
| Android/Bionic ARM64 | NOT VERIFIED |
| Real Termux install, server, Admin UI, config persistence, update, and shutdown | NOT VERIFIED |
| External coding-agent launchers on Termux | NOT VERIFIED |

Before calling Termux support device-verified, run the installer and full core
test suite on an actual ARM64 Termux device; start the server; request the
localhost health endpoint and Admin UI; configure and reload a provider; test
SQLite/config persistence, update, launcher subprocesses, and graceful signal
shutdown; then attach the exact Android/Termux/Python package versions and logs.
