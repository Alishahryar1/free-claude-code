"""Which repairs a failed attempt may try before admission judges the failure."""

from dataclasses import dataclass
from typing import Literal

import pytest

from free_claude_code.providers.stream_failure_resolution import (
    Repair,
    Stage,
    repairs,
)

type Delivery = Literal["none", "fresh", "this_attempt", "earlier_attempt"]
type RepairName = Literal["refresh", "correct"]


@dataclass(frozen=True, slots=True)
class Row:
    delivery: Delivery = "fresh"
    auth_status: int | None = None
    can_refresh: bool = True
    refreshed: bool = False
    normal_stop_seen: bool = False
    continuing: bool = False


ROWS: list[tuple[Row, tuple[RepairName, ...]]] = [
    (Row(delivery="none", auth_status=401), ("refresh", "correct")),
    (Row(delivery="none"), ("correct",)),
    (Row(auth_status=401), ("refresh", "correct")),
    (Row(auth_status=403), ("refresh", "correct")),
    (Row(auth_status=429), ("correct",)),
    (Row(auth_status=401, refreshed=True), ("correct",)),
    (Row(auth_status=401, can_refresh=False), ("correct",)),
    # Content the client has seen rules out a refresh; content from this
    # attempt also rules out replacing the request.
    (Row(delivery="this_attempt", auth_status=401), ()),
    (Row(delivery="earlier_attempt", auth_status=401), ("correct",)),
    # A model stop proves upstream accepted the request.
    (Row(normal_stop_seen=True, auth_status=401), ()),
    (Row(delivery="earlier_attempt", normal_stop_seen=True), ("correct",)),
    (Row(delivery="earlier_attempt", normal_stop_seen=True, continuing=True), ()),
    (
        Row(delivery="none", normal_stop_seen=True, auth_status=401),
        ("refresh", "correct"),
    ),
]


@pytest.mark.parametrize(("row", "expected"), ROWS)
def test_repairs(row: Row, expected: tuple[RepairName, ...]) -> None:
    released = {
        "none": None,
        "fresh": False,
        "this_attempt": True,
        "earlier_attempt": True,
    }[row.delivery]
    assert repairs(
        stage=Stage.OPEN,
        auth_status=row.auth_status,
        can_refresh=row.can_refresh,
        refreshed=row.refreshed,
        content_released=released,
        attempt_content_released=row.delivery == "this_attempt",
        normal_stop_seen=row.normal_stop_seen,
        continuing=row.continuing,
    ) == tuple(Repair(name) for name in expected)


def test_nothing_is_repaired_before_an_attempt() -> None:
    assert (
        repairs(
            stage=Stage.BEFORE_ATTEMPT,
            auth_status=401,
            can_refresh=True,
            refreshed=False,
            content_released=None,
            attempt_content_released=False,
            normal_stop_seen=False,
            continuing=False,
        )
        == ()
    )
