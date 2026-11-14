"""Presentation policy for current MiniMax text choices, not request validation."""

# Gateway IDs are only shown when discovered; never synthesize gateway routes.
_MINIMAX_CHOICES = {
    "minimax": ("MiniMax-", ("MiniMax-M3.1-Flash", "MiniMax-M3")),
    "open_router": (
        "minimax/minimax-",
        ("minimax/minimax-m3.1-flash:free", "minimax/minimax-m3"),
    ),
    "opencode_zen": (
        "minimax-",
        ("minimax-m3.1-flash-free", "minimax-m3"),
    ),
    "opencode_go": (
        "minimax-",
        ("minimax-m3.1-flash-free", "minimax-m3"),
    ),
    "nvidia_nim": ("minimaxai/minimax-", ("minimaxai/minimax-m3",)),
    "wafer": ("MiniMax-", ("MiniMax-M3",)),
}


def is_current_model_choice(provider: str, model: str) -> bool:
    """Filter ordinary suggestions while explicit references remain routable."""
    policy = _MINIMAX_CHOICES.get(provider)
    if policy is None:
        return True
    prefix, current = policy
    name = model.casefold()
    suffix = name.removeprefix(prefix.casefold())
    is_text_model = suffix.startswith("text-") or (
        len(suffix) > 1 and suffix[0] == "m" and suffix[1].isdigit()
    )
    is_minimax_text = (name.startswith(prefix.casefold()) and is_text_model) or (
        provider == "minimax" and name.startswith("abab")
    )
    return not is_minimax_text or model in current


def model_choice_order(provider: str, model: str) -> tuple[str, int]:
    """Place Flash first within MiniMax's current choices."""
    policy = _MINIMAX_CHOICES.get(provider)
    current = policy[1] if policy else ()
    if len(current) == 2 and model in current:
        return current[1].casefold(), current.index(model)
    return model.casefold(), 0
