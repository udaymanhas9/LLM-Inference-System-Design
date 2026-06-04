import hashlib
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WorkerInfo:
    worker_id: str
    address: str
    queue_depth: int = 0
    max_queue_depth: int = 50


@dataclass
class RouteResult:
    worker: WorkerInfo
    cache_affinity_hit: bool  # True if we routed to the affinity-preferred worker


class CacheAffinityRouter:
    """
    Routes requests to workers based on a hash of the prompt prefix.

    Rationale: tenants commonly share a long system prompt across many users.
    Routing those requests to the same worker means that system prompt's KV
    cache blocks stay warm — reducing prefill cost from O(prompt_len) to O(0)
    for cache hits (PagedAttention prefix caching).

    Falls back to least-loaded worker when the affinity target is saturated.
    """

    PREFIX_CHARS = 512  # proxy for ~128 tokens

    def __init__(self, workers: List[WorkerInfo]) -> None:
        self._workers = workers

    def _affinity_index(self, prompt: str) -> int:
        prefix = prompt[: self.PREFIX_CHARS].encode()
        digest = hashlib.sha256(prefix).digest()
        return int.from_bytes(digest[:4], "big") % len(self._workers)

    def route(self, tenant_id: str, prompt: str) -> Optional[RouteResult]:
        if not self._workers:
            return None

        idx = self._affinity_index(prompt)
        affinity_worker = self._workers[idx]

        if affinity_worker.queue_depth < affinity_worker.max_queue_depth:
            affinity_worker.queue_depth += 1
            return RouteResult(worker=affinity_worker, cache_affinity_hit=True)

        # Affinity target saturated — fall back to least-loaded
        candidates = sorted(self._workers, key=lambda w: w.queue_depth)
        for candidate in candidates:
            if candidate.queue_depth < candidate.max_queue_depth:
                candidate.queue_depth += 1
                return RouteResult(worker=candidate, cache_affinity_hit=False)

        return None  # all workers saturated — admission controller should 503

    def release(self, worker: WorkerInfo) -> None:
        worker.queue_depth = max(0, worker.queue_depth - 1)

    def add_worker(self, worker: WorkerInfo) -> None:
        self._workers.append(worker)
