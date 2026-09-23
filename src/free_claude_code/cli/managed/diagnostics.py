"""Managed Claude Code diagnostic classification."""

from dataclasses import dataclass

_BENIGN_STDERR_MARKERS = ("claude.ai connectors are disabled",)


@dataclass(frozen=True, slots=True)
class ManagedClaudeStderrDiagnostics:
    """Classified stderr lines from one managed Claude Code invocation."""

    benign_lines: tuple[str, ...]
    fatal_lines: tuple[str, ...]

    @property
    def fatal_text(self) -> str | None:
        text = "\n".join(self.fatal_lines).strip()
        return text or None

    @property
    def has_benign(self) -> bool:
        return bool(self.benign_lines)


def classify_managed_claude_stderr(
    stderr_text: str,
) -> ManagedClaudeStderrDiagnostics:
    """Classify known benign Claude Code diagnostics separately from failures."""
    benign_lines: list[str] = []
    fatal_lines: list[str] = []
    for line in _stderr_lines(stderr_text):
        lowered = line.lower()
        if any(marker in lowered for marker in _BENIGN_STDERR_MARKERS):
            benign_lines.append(line)
        else:
            fatal_lines.append(line)
    return ManagedClaudeStderrDiagnostics(
        benign_lines=tuple(benign_lines),
        fatal_lines=tuple(fatal_lines),
    )


def _stderr_lines(stderr_text: str) -> tuple[str, ...]:
    stripped = stderr_text.strip()
    if not stripped:
        return ()
    return tuple(line.strip() for line in stripped.splitlines() if line.strip())
