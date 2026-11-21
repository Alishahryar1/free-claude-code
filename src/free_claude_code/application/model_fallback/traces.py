"""Trace model fallback transitions."""

from loguru import logger

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.trace import trace_event

from ..routing import ProviderModelTarget
from .types import WireApi


def trace_fallback_started(
    *,
    request_id: str,
    generation_id: int | None,
    wire_api: WireApi,
    failed: ProviderModelTarget,
    selected: ProviderModelTarget,
    failure: ExecutionFailure,
    candidate_index: int,
    candidate_count: int,
    recovery_action: str,
    recovery_reason: str | None,
) -> None:
    fields: dict[str, object] = {
        "request_id": request_id,
        "wire_api": wire_api,
        "from_provider_model_ref": failed.provider_model_ref,
        "to_provider_model_ref": selected.provider_model_ref,
        "from_api_key_number": failed.api_key_number,
        "to_api_key_number": selected.api_key_number,
        "candidate_index": candidate_index,
        "candidate_count": candidate_count,
        "failure_kind": failure.kind.value,
        "status_code": failure.status_code,
        "provider_retryable": failure.retryable,
        "recovery_action": recovery_action,
        "recovery_reason": recovery_reason,
    }
    if generation_id is not None:
        fields["generation_id"] = generation_id
    trace_event(
        stage="execution",
        event="free_claude_code.model_fallback.started",
        source="application",
        **fields,
    )
    logger.info(
        "Model fallback: request_id={} from={} to={} candidate={}/{} "
        "failure_kind={} status_code={}",
        request_id,
        failed.provider_model_ref,
        selected.provider_model_ref,
        candidate_index,
        candidate_count,
        failure.kind.value,
        failure.status_code,
    )


def trace_fallback_selected(
    *,
    request_id: str,
    generation_id: int | None,
    wire_api: WireApi,
    selected: ProviderModelTarget,
    candidate_index: int,
    candidate_count: int,
) -> None:
    fields: dict[str, object] = {
        "request_id": request_id,
        "wire_api": wire_api,
        "selected_provider_model_ref": selected.provider_model_ref,
        "api_key_number": selected.api_key_number,
        "candidate_index": candidate_index,
        "candidate_count": candidate_count,
    }
    if generation_id is not None:
        fields["generation_id"] = generation_id
    trace_event(
        stage="execution",
        event="free_claude_code.model_fallback.selected",
        source="application",
        **fields,
    )
