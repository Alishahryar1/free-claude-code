# Native Termux support

FCC's Termux support targets the native Termux environment on Android AArch64.
It does not use proot, a Linux distribution container, root, or a virtual
machine.

## Requirements

- Termux installed from a supported source.
- Android ARM64/AArch64; 32-bit ARM and x86 Android are not supported.
- Termux's Python 3.14.x and `uv` packages.
- Internet access to install FCC and its Python dependencies.
- Free storage for the Termux compiler toolchain and Python package builds.

The upstream Termux package recipes currently advertise Python 3.14.10 and uv
0.13.7. FCC accepts Python 3.14 patch releases (`>=3.14,<3.15`) because its
source uses Python 3.14 syntax. If Termux changes its available Python minor
version, FCC must be updated before that version can run it.

## Install

In Termux, fetch the installer as a file, inspect it if desired, and then run
it. Do not pipe a remote script directly to a shell:

```sh
pkg install curl
curl -fsSLo "$TMPDIR/fcc-install-termux.sh" \
  "https://raw.githubusercontent.com/Alishahryar1/free-claude-code/main/scripts/install-termux.sh"
sh "$TMPDIR/fcc-install-termux.sh"
```

The installer validates Termux, Android, AArch64, and the Python runtime; uses
`pkg` to install Python, uv, Clang, Rust, and build prerequisites; and installs
FCC into `~/.local/share/free-claude-code/termux-venv`. It links FCC commands
into `$PREFIX/bin`. It does not change `~/.fcc`, install external coding
agents, or require root. Re-running it updates the same private environment;
it will refuse to replace conflicting commands.

Dependency compilation can take several minutes and use significant temporary
storage. The installer reports failures from native builds rather than
pretending that a Linux wheel is compatible with Android/Bionic.

## Start FCC and open Admin

```sh
fcc-server
```

Keep the Termux session running. Open the localhost Admin URL printed by FCC in
Android's browser. If automatic browser handoff is unavailable, copy that URL
into the browser manually. To restrict the listener to the device, set
`HOST=127.0.0.1` in `~/.fcc/.env`; FCC's existing configuration and Admin
authentication behavior otherwise remains unchanged.

Run the platform-specific diagnostics with:

```sh
fcc-doctor --termux
```

The command returns JSON with `PASS`, `WARNING`, or `FAIL` checks. It does not
copy diagnostic data to a clipboard and never includes API keys or OAuth
tokens. A running FCC instance may cause a port warning; this is not itself a
failure.

## Configure providers and coding agents

Configure providers from the Admin UI as on other platforms. Configuration is
stored in `~/.fcc` and survives FCC environment updates.

FCC does not install or certify external coding agents on Termux. A launcher
such as `fcc-claude`, `fcc-codex`, `fcc-pi`, `fcc-opencode`, `fcc-cline`,
`fcc-hermes`, `fcc-dsh`, `fcc-grok`, `fcc-muse`, `fcc-agy`, or `fcc-aider`
requires that agent's executable to be installed separately, available on
`PATH`, and usable in native Termux. Check one with `command -v <agent>` and
verify that agent's own Android support before connecting it to FCC.
`fcc-desktop` is not supported on Termux; FCC's server and browser UI are the
native interface.

The NVIDIA Riva/gRPC voice client and FastAPI's desktop-oriented `standard`
extras are excluded for Android kernel releases. Core FCC serving does not
need them. NVIDIA NIM voice transcription and local Whisper are therefore not
part of this Termux install. Termux local CUDA/Whisper support is not claimed.

Editor/desktop integrations that expect macOS, Windows, or desktop Linux
configuration directories or applications may be unavailable. The integration
cards report what FCC can detect in the current environment.

## Update

Stop FCC and run:

```sh
fcc-update
```

This upgrades FCC and its Python dependencies in the private environment.
Termux system packages are not upgraded by this command. Re-run the installer
to repair the FCC environment or refresh its compiler prerequisites.

## Troubleshooting

- `fcc-doctor --termux` reports the failed check and a suggested fix.
- If an extension build fails, install/update the Termux packages named by the
  installer (`clang`, `rust`, `pkg-config`, `openssl`, and `libffi`) and rerun
  the installer. Preserve the full build error when reporting the issue.
- If `fcc-server` cannot bind its port, check the `FCC port` diagnostic and
  change `PORT` in `~/.fcc/.env` to a free port.
- If an agent launcher says its executable is unavailable, install that agent
  separately and confirm it is on `PATH`.
- If no browser opens, use the localhost URL printed by `fcc-server`.

## Uninstall

The FCC Python environment and command links are separate from configuration.
Remove only links still pointing to FCC's Termux environment, then remove that
environment:

```sh
for command in fcc-desktop fcc-server fcc-doctor fcc-claude fcc-codex fcc-pi \
  fcc-opencode fcc-cline fcc-hermes fcc-dsh fcc-grok fcc-muse fcc-agy \
  fcc-aider; do
  link="$PREFIX/bin/$command"
  [ "$(readlink "$link" 2>/dev/null || true)" = \
    "$HOME/.local/share/free-claude-code/termux-venv/bin/$command" ] &&
    rm "$link"
done
if [ "$(readlink "$PREFIX/bin/fcc-update" 2>/dev/null || true)" = \
  "$HOME/.local/share/free-claude-code/fcc-update" ]; then
  rm "$PREFIX/bin/fcc-update"
fi
rm -rf "$HOME/.local/share/free-claude-code"
```

`~/.fcc` is intentionally retained. Remove it separately only if you also want
to delete FCC configuration, credentials, databases, and logs. Packages
installed with `pkg` are shared Termux packages and are not removed.

## Compatibility audit and verification

See [TERMUX_COMPATIBILITY_AUDIT.md](./TERMUX_COMPATIBILITY_AUDIT.md) for the
dependency/platform audit and verification classifications. The automated
`tests/termux/` suite exercises detection, diagnostics, and installer
guardrails on the test host; it is not a substitute for a physical or emulated
Termux ARM64 integration run.
