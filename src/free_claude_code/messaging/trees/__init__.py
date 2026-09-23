"""Internal messaging tree package facade."""

from .identity import TreeIdentity
from .manager import TreeQueueManager
from .node import MessageReferenceKind, MessageState
from .snapshot import ConversationSnapshot, TreeSnapshot
from .transitions import (
    AdmissionRejection,
    CancellationReason,
    CancellationResult,
    CancellationUiOwner,
    FailureResult,
    MessageSubtreeRemovalResult,
    NodeClaim,
    NodeUiTarget,
    NodeView,
    QueueDecision,
    QueueEntry,
    ReplyTarget,
)

__all__ = [
    "AdmissionRejection",
    "CancellationReason",
    "CancellationResult",
    "CancellationUiOwner",
    "ConversationSnapshot",
    "FailureResult",
    "MessageReferenceKind",
    "MessageState",
    "MessageSubtreeRemovalResult",
    "NodeClaim",
    "NodeUiTarget",
    "NodeView",
    "QueueDecision",
    "QueueEntry",
    "ReplyTarget",
    "TreeIdentity",
    "TreeQueueManager",
    "TreeSnapshot",
]
