"""Detached transition values crossing the messaging tree ownership boundary."""

from dataclasses import dataclass
from enum import Enum

from ..models import MessageScope
from .identity import TreeIdentity
from .node import MessageReferenceKind, MessageState
from .snapshot import TreeSnapshot


class CancellationUiOwner(Enum):
    """Component responsible for the final user-visible cancellation edit."""

    RUNNER = "runner"
    WORKFLOW = "workflow"


class CancellationReason(Enum):
    """Customer operation that cancelled a running messaging claim."""

    STOP = "stop"
    CLEAR = "clear"


class AdmissionRejection(Enum):
    """Why a turn was not admitted to an execution tree."""

    DUPLICATE = "duplicate"
    PARENT_REMOVED = "parent_removed"


@dataclass(frozen=True, slots=True)
class NodeUiTarget:
    """Copied node coordinates needed for an external UI effect."""

    scope: MessageScope
    node_id: str
    status_message_id: str


@dataclass(frozen=True, slots=True)
class NodeClaim:
    """Exclusive permission to execute one node in a conversation tree."""

    identity: TreeIdentity
    claim_id: str
    node: NodeUiTarget
    prompt: str
    parent_session_id: str | None


@dataclass(frozen=True, slots=True)
class QueueEntry:
    """One immutable queue-position update."""

    node: NodeUiTarget
    position: int


@dataclass(frozen=True, slots=True)
class QueueDecision:
    """Atomic result of admitting a node to a tree."""

    claim: NodeClaim | None
    position: int | None
    snapshot: TreeSnapshot | None
    rejection: AdmissionRejection | None = None

    @property
    def accepted(self) -> bool:
        return self.snapshot is not None


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Atomic result of releasing a claim and selecting its successor."""

    next_claim: NodeClaim | None
    queue: tuple[QueueEntry, ...]


@dataclass(frozen=True, slots=True)
class CancellationEffect:
    """Copied cancellation fact for the UI layer."""

    node: NodeUiTarget
    ui_owner: CancellationUiOwner


@dataclass(frozen=True, slots=True)
class TreeCancellation:
    """Single-tree cancellation transition consumed by the manager."""

    nodes: tuple[NodeUiTarget, ...]
    active_claim: NodeClaim | None
    queue_update: tuple[QueueEntry, ...] | None


@dataclass(frozen=True, slots=True)
class CancellationResult:
    """External effects and persistence snapshots from a cancellation request."""

    effects: tuple[CancellationEffect, ...] = ()
    snapshots: tuple[TreeSnapshot, ...] = ()


@dataclass(frozen=True, slots=True)
class MessageSubtreeRemoval:
    """Single-tree atomic cancellation and reference-subtree removal."""

    cancellation: TreeCancellation
    removed_message_ids: frozenset[str]
    removed_entire_tree: bool


@dataclass(frozen=True, slots=True)
class MessageSubtreeRemovalResult:
    """Manager result for a literal message-subtree clear."""

    cancellation: CancellationResult
    removed_tree_identity: TreeIdentity | None
    delete_message_ids: frozenset[str]
    tree_matched: bool


@dataclass(frozen=True, slots=True)
class ReplyTarget:
    """Resolved reply destination and advisory position before admission."""

    node_id: str
    reference_id: str
    reference_kind: MessageReferenceKind
    queue_position: int | None


@dataclass(frozen=True, slots=True)
class NodeView:
    """Immutable read model for diagnostics, tests, and smoke assertions."""

    identity: TreeIdentity
    node_id: str
    state: MessageState
    parent_id: str | None
    session_id: str | None


@dataclass(frozen=True, slots=True)
class FailureResult:
    """Atomic node failure plus copied child UI effects."""

    affected: tuple[NodeUiTarget, ...]
    queue_update: tuple[QueueEntry, ...] | None
    snapshot: TreeSnapshot | None


__all__ = [
    "AdmissionRejection",
    "CancellationEffect",
    "CancellationReason",
    "CancellationResult",
    "CancellationUiOwner",
    "CompletionResult",
    "FailureResult",
    "MessageSubtreeRemoval",
    "MessageSubtreeRemovalResult",
    "NodeClaim",
    "NodeUiTarget",
    "NodeView",
    "QueueDecision",
    "QueueEntry",
    "ReplyTarget",
    "TreeCancellation",
]
