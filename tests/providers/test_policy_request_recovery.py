from unittest.mock import MagicMock

import pytest

from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_strict_routing,
)
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.request_recovery import (
    RequestCorrections,
    RequestRecovery,
)
from free_claude_code.providers.stream_recovery import (
    RecoveryController,
    RecoveryHoldbackBuffer,
)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.asyncio
async def test_request_correction_respects_strict_metadata_commitment(strict):
    execution = MagicMock()
    execution.delivery = StreamDeliveryState()
    execution.can_attempt = True
    attempt = MagicMock()
    attempt.accepted = True
    recovery = RecoveryController(execution.delivery)
    recovery.push(": metadata\n\n")
    recovery.flush()
    correction = MagicMock(return_value={"model": "same", "temperature": 0})
    with bind_strict_routing(strict):
        result = await RequestRecovery(execution, stream=recovery).retry_request(
            ValueError("correctable"),
            None,
            attempt,
            {"model": "same"},
            operation_kind=ProviderOperationKind.GENERATION,
            propose_correction=correction,
        )
    assert (result is None) is strict
    assert correction.call_count == (0 if strict else 1)


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.asyncio
async def test_real_adapter_does_not_correct_after_strict_metadata(
    monkeypatch, protocol, wire
):
    from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
    from tests.api.test_hidden_stream_retries import delivered
    from tests.providers.test_history_transports import _events_for, _harness

    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    corrections = []

    def correct(self, error, body, **kwargs):
        corrections.append(True)
        return body | {"temperature": 0}

    monkeypatch.setattr(RequestCorrections, "next_body", correct)
    metadata = _events_for(protocol)[0]
    if protocol == "chat":
        metadata = metadata | {
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ]
        }
    async with _harness(protocol, lambda bodies: (200, [metadata]), max_attempts=2) as (
        send,
        bodies,
        _,
    ):
        with bind_strict_routing(True):
            result = await delivered(
                send(wire, [{"role": "user", "content": "hello"}]), wire
            )
    assert len(bodies) == 1
    assert corrections == []
    assert parse_sse_text(result)[-1].event == (
        "error" if wire == "messages" else "response.failed"
    )
