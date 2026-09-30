"""Best-effort text copy for the diagnostic command."""

import pyperclip


class ClipboardUnavailable(Exception):
    """The current desktop cannot accept clipboard text."""


def copy_text(text: str) -> None:
    try:
        pyperclip.copy(text)
    except (pyperclip.PyperclipException, OSError) as error:
        raise ClipboardUnavailable from error
