"""Decide what follows a failed stream attempt.

Pure rules: given what the attempt left behind and admission's answers so far,
return the repairs worth trying and the next step. ``attempt_loop`` makes the
calls these rules depend on and carries out what they decide.
"""

from .decision import (
    NextStep,
    Repair,
    Stage,
    can_restart,
    is_hidden_stop,
    next_step,
    repairs,
)

__all__ = [
    "NextStep",
    "Repair",
    "Stage",
    "can_restart",
    "is_hidden_stop",
    "next_step",
    "repairs",
]
