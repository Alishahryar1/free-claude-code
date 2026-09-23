"""Gateway-safe model ID encoding shared by API and CLI adapters."""

from dataclasses import dataclass

GATEWAY_MODEL_ID_PREFIX = "anthropic"

# Claude Code currently treats any model id containing ``claude-3-`` as not
# supporting thinking. This intentionally uses that client-side capability
# heuristic while keeping the real provider/model ref reversible for routing.
NO_THINKING_GATEWAY_MODEL_ID_PREFIX = "claude-3-freecc-no-thinking"
DESKTOP_MODEL_PREFIX = "claude-fcc"
DESKTOP_NO_THINKING_PREFIX = "claude-3-fcc"


def desktop_model_id(provider_model_ref: str, *, no_thinking: bool = False) -> str:
    prefix = DESKTOP_NO_THINKING_PREFIX if no_thinking else DESKTOP_MODEL_PREFIX
    return f"{prefix}/{provider_model_ref.encode('utf-8').hex()}"


@dataclass(frozen=True, slots=True)
class DecodedGatewayModelId:
    provider_id: str
    provider_model: str
    force_reasoning_off: bool = False


def gateway_model_id(provider_model_ref: str) -> str:
    """Return the normal Claude Code-discoverable id for a provider/model ref."""
    return f"{GATEWAY_MODEL_ID_PREFIX}/{provider_model_ref}"


def no_thinking_gateway_model_id(provider_model_ref: str) -> str:
    """Return a Claude Code-discoverable id that disables client thinking."""
    return f"{NO_THINKING_GATEWAY_MODEL_ID_PREFIX}/{provider_model_ref}"


def decode_gateway_model_id(model_name: str) -> DecodedGatewayModelId | None:
    """Decode a model id advertised by this gateway, if it is one."""
    prefix, separator, remainder = model_name.partition("/")
    if not separator:
        if prefix in {DESKTOP_MODEL_PREFIX, DESKTOP_NO_THINKING_PREFIX}:
            raise ValueError("Invalid Desktop model ID")
        return None

    if prefix in {DESKTOP_MODEL_PREFIX, DESKTOP_NO_THINKING_PREFIX}:
        try:
            decoded = bytes.fromhex(remainder).decode("utf-8")
        except ValueError, UnicodeError:
            raise ValueError("Invalid Desktop model ID") from None
        if not remainder or decoded.encode("utf-8").hex() != remainder:
            raise ValueError("Invalid Desktop model ID")
        provider_id, separator, provider_model = decoded.partition("/")
        if not provider_id or not separator or not provider_model:
            raise ValueError("Invalid Desktop model ID")
        return DecodedGatewayModelId(
            provider_id, provider_model, prefix == DESKTOP_NO_THINKING_PREFIX
        )
    if prefix == GATEWAY_MODEL_ID_PREFIX:
        force_reasoning_off = False
    elif prefix == NO_THINKING_GATEWAY_MODEL_ID_PREFIX:
        force_reasoning_off = True
    else:
        return None

    provider_id, provider_separator, provider_model = remainder.partition("/")
    if not provider_separator or not provider_model:
        return None

    return DecodedGatewayModelId(
        provider_id=provider_id,
        provider_model=provider_model,
        force_reasoning_off=force_reasoning_off,
    )
