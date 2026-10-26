"""Validate Gemini call associations without changing caller-owned history."""

from dataclasses import dataclass

from .errors import GeminiConversionError
from .models import Content, GeminiRequest, Part
from .replay import ReplayBundle, decode_bundle


@dataclass(frozen=True, slots=True)
class GeminiTurn:
    content: Content
    replay: tuple[ReplayBundle, ...]


def normalized_history(request: GeminiRequest) -> tuple[GeminiTurn, ...]:
    pending: dict[str, str] = {}
    seen: set[str] = set()
    bundles: dict[str, ReplayBundle] = {}
    turns = []
    for index, original in enumerate(request.contents):
        parts: list[Part] = []
        replay = []
        for part_index, part in enumerate(original.parts):
            if part.thought_signature is not None:
                bundle = decode_bundle(part.thought_signature)
                if bundle is None:
                    raise GeminiConversionError(
                        "Caller-native Gemini signatures cannot be replayed through this gateway. "
                        "Only FCC-issued thoughtSignature replay is supported."
                    )
                previous = bundles.get(bundle.response_id)
                if previous is not None and previous != bundle:
                    raise GeminiConversionError("Conflicting echoed replay bundles")
                if previous is None:
                    bundles[bundle.response_id] = bundle
                    replay.append(bundle)
            call, response = part.function_call, part.function_response
            if call is not None:
                call_id = call.id or f"call_gemini_{index}_{part_index}"
                if call_id in seen:
                    raise GeminiConversionError("Duplicate functionCall id")
                seen.add(call_id)
                pending[call_id] = call.name
                part = part.model_copy(
                    update={"function_call": call.model_copy(update={"id": call_id})}
                )
            elif response is not None:
                matches = [
                    call_id
                    for call_id, name in pending.items()
                    if name == response.name
                ]
                call_id = response.id or (matches[0] if len(matches) == 1 else None)
                if call_id is None or pending.get(call_id) != response.name:
                    raise GeminiConversionError(
                        "Ambiguous, orphaned or duplicate functionResponse"
                    )
                del pending[call_id]
                part = part.model_copy(
                    update={
                        "function_response": response.model_copy(update={"id": call_id})
                    }
                )
            parts.append(part)
        for bundle in replay:
            current_calls = {
                part.function_call.id
                for part in parts
                if part.function_call is not None
            }
            if any(record.call_id not in current_calls for record in bundle.calls):
                raise GeminiConversionError(
                    "Replay metadata does not match this generation's function calls"
                )
        turns.append(
            GeminiTurn(original.model_copy(update={"parts": parts}), tuple(replay))
        )
    if pending:
        raise GeminiConversionError("Gemini history has unresolved function calls")
    return tuple(turns)
