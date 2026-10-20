"""Run a command behind a PTY from a fresh single-threaded process."""

import os
import pty
import sys


def main() -> int:
    spawn = vars(pty)["spawn"]
    status = spawn(sys.argv[1:])
    return os.waitstatus_to_exitcode(status)


if __name__ == "__main__":
    raise SystemExit(main())
