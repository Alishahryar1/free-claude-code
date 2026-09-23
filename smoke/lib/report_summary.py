"""Summarize smoke JSON reports for local and workflow triage."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SmokeSummary:
    reports: int
    outcomes: int
    classifications: dict[str, int]

    @property
    def has_regression(self) -> bool:
        return bool(
            self.classifications.get("product_failure", 0)
            or self.classifications.get("harness_bug", 0)
        )


def summarize_reports(results_dir: Path) -> SmokeSummary:
    """Read all report JSON files and count outcome classifications."""
    counts: Counter[str] = Counter()
    reports = 0
    outcomes = 0
    for path in sorted(results_dir.glob("report-*.json")):
        reports += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        for outcome in payload.get("outcomes", []):
            if not isinstance(outcome, dict):
                continue
            outcomes += 1
            counts[str(outcome.get("classification") or "unknown")] += 1
    return SmokeSummary(
        reports=reports,
        outcomes=outcomes,
        classifications=dict(sorted(counts.items())),
    )


def format_summary(summary: SmokeSummary) -> str:
    """Return a compact human-readable summary."""
    parts = [
        f"reports={summary.reports}",
        f"outcomes={summary.outcomes}",
    ]
    parts.extend(f"{name}={count}" for name, count in summary.classifications.items())
    status = "regression" if summary.has_regression else "ok"
    return f"smoke_summary status={status} " + " ".join(parts)
