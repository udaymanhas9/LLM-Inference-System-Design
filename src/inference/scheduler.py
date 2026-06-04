import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import List


@dataclass
class ScheduledRequest:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    max_tokens: int = 200
    tenant_id: str = ""
    priority: int = 0
    created_at: float = field(default_factory=time.monotonic)

    # Scheduler-internal state
    prefill_chunks_remaining: int = 0
    is_prefilling: bool = True
    tokens_generated: int = 0


class ChunkedPrefillScheduler:
    """
    Illustrates chunked-prefill scheduling to prevent prefill-decode interference.

    Problem: a 500-token prefill monopolizes the GPU for one full forward pass,
    blocking all in-flight decode sequences for that iteration and spiking their
    P95 latency.

    Solution: slice each prefill into chunks of MAX_PREFILL_CHUNK_TOKENS tokens.
    Each iteration: process one chunk, then run a decode step for all active
    decode sequences. Slightly increases TTFT for the chunked request but
    dramatically reduces P95 decode latency across the system.

    In production this logic lives inside vLLM's PagedAttentionScheduler.
    This stub exposes the scheduling model for illustration.
    """

    def __init__(
        self,
        max_prefill_chunk_tokens: int = 256,
        max_batch_size: int = 64,
    ) -> None:
        self._max_prefill_chunk_tokens = max_prefill_chunk_tokens
        self._max_batch_size = max_batch_size
        self._prefill_queue: asyncio.Queue[ScheduledRequest] = asyncio.Queue()
        self._decode_active: List[ScheduledRequest] = []

    async def submit(self, request: ScheduledRequest) -> None:
        approx_tokens = max(1, len(request.prompt.split()))
        request.prefill_chunks_remaining = (
            approx_tokens + self._max_prefill_chunk_tokens - 1
        ) // self._max_prefill_chunk_tokens
        request.is_prefilling = True
        await self._prefill_queue.put(request)

    async def next_batch(self) -> List[ScheduledRequest]:
        """
        Returns a mixed batch: one prefill chunk + decode sequences filling the rest.
        Called once per scheduler iteration.
        """
        batch: List[ScheduledRequest] = []

        if not self._prefill_queue.empty():
            req = await self._prefill_queue.get()
            req.prefill_chunks_remaining -= 1

            if req.prefill_chunks_remaining > 0:
                # Still has chunks remaining — re-enqueue
                await self._prefill_queue.put(req)
            else:
                # Prefill done — move to decode phase
                req.is_prefilling = False
                self._decode_active.append(req)

            batch.append(req)

        # Fill remaining batch slots with decode sequences
        decode_slots = self._max_batch_size - len(batch)
        batch.extend(self._decode_active[:decode_slots])

        return batch

    def complete(self, request_id: str) -> None:
        self._decode_active = [
            r for r in self._decode_active if r.request_id != request_id
        ]

    @property
    def pending_prefills(self) -> int:
        return self._prefill_queue.qsize()

    @property
    def active_decodes(self) -> int:
        return len(self._decode_active)
