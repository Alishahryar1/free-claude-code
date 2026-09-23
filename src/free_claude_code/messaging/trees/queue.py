"""FIFO queue state for one messaging conversation tree."""

from collections import deque


class MessageNodeQueue:
    """Queue with snapshot/remove helpers, backed by a deque and a set index."""

    def __init__(self, items: list[str] | None = None) -> None:
        self._deque: deque[str] = deque()
        self._set: set[str] = set()
        for item in items or []:
            self.put(item)

    def put(self, item: str) -> bool:
        """Append a unique item and report whether the queue changed."""
        if item in self._set:
            return False
        self._deque.append(item)
        self._set.add(item)
        return True

    def pop(self) -> str | None:
        if not self._deque:
            return None
        item = self._deque.popleft()
        self._set.discard(item)
        return item

    def qsize(self) -> int:
        return len(self._deque)

    def items(self) -> tuple[str, ...]:
        return tuple(self._deque)

    def remove(self, item: str) -> bool:
        if item not in self._set:
            return False
        self._set.discard(item)
        self._deque = deque(x for x in self._deque if x != item)
        return True

    def drain(self) -> tuple[str, ...]:
        items = tuple(self._deque)
        self._deque.clear()
        self._set.clear()
        return items
