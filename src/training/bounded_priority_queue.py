import heapq
from typing import Generic, TypeVar, Optional, List, Tuple

T = TypeVar('T')

from queue import PriorityQueue


class BoundedPriorityQueue(Generic[T]):
    """
    A priority queue with fixed capacity. Keeps only up to `capacity` items:
    when full, on push, the item is inserted only if its priority is 'better'
    than the current worst in the queue.

    By default, higher numeric `priority` means better. Internally uses a min-heap
    so that heap[0] is the smallest priority among kept items.
    """

    def __init__(self, capacity: int, mode: str = "min", data: Optional[List[Tuple[float, T]]] = None):
        assert capacity > 0, "Capacity must be positive"
        self.capacity = capacity
        self.mode: str = mode
        self._heap: List[Tuple[float, T]] = [] if data is None else data  # min-heap of (priority, item)

    def push(self, priority: float, item: T) -> T:
        """
        Insert (priority, item). If queue not full, always added.
        If full and priority > smallest priority in heap, replace that one.
        Otherwise drop this item.
        """
        if self.mode == "max":
            priority = -priority

        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, (priority, item))
            return None
        else:
            # heap[0] is the smallest priority in kept items
            if priority > self._heap[0][0]:
                # replace smallest with new
                return heapq.heapreplace(self._heap, (priority, item))[1]



    def pop(self) -> Tuple[float, T]:
        """
        Pop and return the smallest-priority item currently in the queue.
        (Which is the “worst” among those kept.) Raises IndexError if empty.
        """
        return heapq.heappop(self._heap)

    def peek(self) -> Optional[Tuple[float, T]]:
        """
        Peek at the smallest-priority item (the one that would be popped next),
        or None if empty.
        """
        if not self._heap:
            return None
        return self._heap[0]

    def __len__(self) -> int:
        return len(self._heap)

    def items(self) -> List[Tuple[float, T]]:
        """
        Return a list of (priority, item) currently in the queue, in arbitrary order.
        """
        return list(self._heap)

    def sorted_items(self, reverse: bool = True) -> List[Tuple[float, T]]:
        """
        Return items sorted by priority. By default reverse=True gives highest first.
        """
        return sorted(self._heap, key=lambda x: x[0], reverse=reverse)
