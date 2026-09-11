"""Fixed local-client setup actions exposed by FCC Admin."""

from enum import StrEnum


class IntegrationId(StrEnum):
    CLAUDE_VSCODE = "claude-vscode"
    CODEX = "codex"
    CLAUDE_JETBRAINS = "claude-jetbrains"
    CLAUDE_LOGIN = "claude-login"


class IntegrationAction(StrEnum):
    SETUP = "setup"
    UPDATE = "update"
    DISCONNECT = "disconnect"
    REPAIR = "repair"


class IntegrationError(Exception):
    """A safe, user-facing configuration error, without source-file contents."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def validate_action(target: IntegrationId, action: IntegrationAction) -> None:
    if (target == IntegrationId.CLAUDE_LOGIN) != (action == IntegrationAction.REPAIR):
        raise IntegrationError("This action is not available for this item.")
