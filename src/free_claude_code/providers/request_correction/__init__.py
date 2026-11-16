"""Decide how to correct a rejected request, or report that nothing can.

Pure apart from remembering which corrections the request already used: what
a rejection teaches a longer-lived memory comes back as a ``Lesson`` for the
attempt loop to apply.
"""

from .corrector import RequestCorrector
from .rejection import (
    Correction,
    CorrectionDecision,
    CorrectionStep,
    Lesson,
    NextHistoryMode,
    Proposal,
    Rejection,
)

__all__ = [
    "Correction",
    "CorrectionDecision",
    "CorrectionStep",
    "Lesson",
    "NextHistoryMode",
    "Proposal",
    "Rejection",
    "RequestCorrector",
]
