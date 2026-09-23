"""Small JSON report writer for smoke runs."""

import json
import time
from dataclasses import asdict, dataclass

from .config import SmokeConfig, redacted
from .outcomes import classify_outcome


@dataclass(slots=True)
class SmokeOutcome:
    nodeid: str
    outcome: str
    classification: str
    duration_s: float
    markers: list[str]
    detail: str


class SmokeReport:
    def __init__(self, config: SmokeConfig) -> None:
        self.config = config
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.outcomes: list[SmokeOutcome] = []

    def add(
        self,
        *,
        nodeid: str,
        outcome: str,
        duration_s: float,
        markers: list[str],
        detail: str = "",
    ) -> None:
        self.outcomes.append(
            SmokeOutcome(
                nodeid=nodeid,
                outcome=outcome,
                classification=classify_outcome(
                    nodeid=nodeid, outcome=outcome, detail=detail
                ),
                duration_s=duration_s,
                markers=markers,
                detail=redacted(detail),
            )
        )

    def write(self) -> None:
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        path = (
            self.config.results_dir
            / f"report-{self.config.worker_id}-{int(time.time())}.json"
        )
        payload = {
            "started_at": self.started_at,
            "worker_id": self.config.worker_id,
            "targets": sorted(self.config.targets),
            "outcomes": [asdict(outcome) for outcome in self.outcomes],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
