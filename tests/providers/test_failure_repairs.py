"""Which repairs a failed attempt may try before admission judges the failure."""

from dataclasses import dataclass
from typing import Literal

import pytest

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.request_recovery import RequestRecovery
from tests.providers.support import immediate_admission

BODY: JsonObject = {"model": "m"}
CORRECTED: JsonObject = {"model": "m", "corrected": True}

type Delivery = Literal["none", "fresh", "this_attempt", "earlier_attempt"]
type Repair = Literal["refresh", "correct"]


@dataclass(frozen=True, slots=True)
class Row:
    delivery: Delivery = "fresh"
    auth_status: int | None = None
    can_refresh: bool = True
    refreshed: bool = False
    normal_stop_seen: bool = False
    continuing: bool = False


ROWS: list[tuple[Row, tuple[Repair, ...]]] = [
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


class _Account:
    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        return HttpEndpoint("https://upstream.invalid", {})


def _delivery(kind: Delivery) -> StreamDeliveryState | None:
    if kind == "none":
        return None
    state = StreamDeliveryState()
    if kind != "fresh":
        state.release_content()
    if kind == "earlier_attempt":
        state.begin_attempt()
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize(("row", "expected"), ROWS)
async def test_repairs(row: Row, expected: tuple[Repair, ...]) -> None:
    with bind_stream_delivery(_delivery(row.delivery)):
        execution = immediate_admission().start_execution(request_id="r")
    recovery = RequestRecovery(
        execution,
        endpoint=RequestEndpoint(_Account()) if row.can_refresh else None,
    )
    recovery._refreshed = row.refreshed
    attempt = await execution.open_attempt(ProviderOperationKind.MODEL_DISCOVERY)
    error = ExecutionFailure(FailureKind.UPSTREAM, 400, "rejected", False)

    result = await recovery.retry_request(
        error,
        row.auth_status,
        attempt,
        BODY,
        operation_kind=(
            ProviderOperationKind.CONTINUATION
            if row.continuing
            else ProviderOperationKind.GENERATION
        ),
        propose_correction=lambda: CORRECTED,
        normal_stop_seen=row.normal_stop_seen,
    )

    # Admission authorizes every repair here, so the first one allowed wins.
    first = {"refresh": BODY, "correct": CORRECTED}
    assert result == (first[expected[0]] if expected else None)
