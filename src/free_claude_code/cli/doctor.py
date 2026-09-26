"""Print and copy a shareable local diagnostic report."""

import argparse
import json
import sys
from collections.abc import Sequence

from free_claude_code.runtime.diagnostics import collect_report

from .clipboard import ClipboardUnavailable, copy_text


def main(argv: Sequence[str] | None = None) -> None:
    argparse.ArgumentParser(
        prog="fcc-doctor",
        description="Print and copy FCC diagnostics for a bug report.",
    ).parse_args(argv)
    text = json.dumps(collect_report(), indent=2, ensure_ascii=False) + "\n"
    sys.stdout.write(text)
    try:
        copy_text(text)
    except ClipboardUnavailable:
        print("Clipboard unavailable. Copy the JSON above manually.", file=sys.stderr)
    else:
        print("FCC diagnostics copied to the clipboard.", file=sys.stderr)
