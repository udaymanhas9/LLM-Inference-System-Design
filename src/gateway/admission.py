import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(order=False)
class QueuedRequest:
    tenant_id: str
    prompt: str
    max_tokens: int
    priority: int = 0          # lower value = higher priority (dequeued first)
    enqueued_at: float = field(default_factory=time.monotonic)

    def __lt__(self, other: "QueuedRequest") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.enqueued_at < other.enqueued_at


class AdmissionController:
    """
    Priority queue that decouples ingress spikes from inference capacity.

    When a burst arrives, requests queue here rather than immediately
    overloading workers or being dropped. Returns 429 only when the queue
    is genuinely full — providing backpressure instead of connection drops.
    """

    def __init__(
        self, max_queue_depth: int = 1_000, request_timeout_s: float = 30.0
    ) -> None:
        self._max_queue_depth = max_queue_depth
        self._request_timeout_s = request_timeout_s
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._depth = 0

    async def enqueue(self, request: QueuedRequest) -> bool:
        """Returns False if the queue is full (caller should respond 429)."""
        if self._depth >= self._max_queue_depth:
            return False
        self._depth += 1
        await self._queue.put(request)
        return True

    async def dequeue(self, timeout_s: Optional[float] = None) -> Optional[QueuedRequest]:
        timeout = timeout_s or self._request_timeout_s
        try:
            request = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            self._depth = max(0, self._depth - 1)
            return request
        except asyncio.TimeoutError:
            return None

    def is_full(self) -> bool:
        return self._depth >= self._max_queue_depth

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def retry_after_seconds(self) -> int:
        """Rough estimate of how long a caller should wait before retrying."""
        return max(1, self._depth // 100)
