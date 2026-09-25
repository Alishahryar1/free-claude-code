"""Temporary comparison of the existing parallel suite and isolated Pi tests."""

import os
import subprocess
import sys

commands = (
    (
        "parallel",
        ["-v", "--tb=short", "--durations=50", "--exitfirst"],
    ),
    (
        "isolated",
        ["tests/cli/test_pi_extension.py", "-n", "0", "-v", "--tb=short", "-rP"],
    ),
)
exit_code = 0
for phase, arguments in commands:
    print(f"Starting Node diagnostic phase: {phase}", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *arguments],
        env=os.environ | {"FCC_PI_DIAGNOSTIC_PHASE": phase},
        check=False,
    )
    # A successful isolated comparison must never hide a parallel-suite failure.
    exit_code = exit_code or result.returncode
raise SystemExit(exit_code)
