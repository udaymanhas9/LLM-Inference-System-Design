import threading
import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TokenBucket:
    """
    Token-bucket denominated in LLM tokens/s, not RPS.
    Cost per request = prompt_tokens + max_response_tokens.
    This correctly charges expensive (long) requests more than cheap ones.
    """
    capacity: float        # max tokens in bucket (burst headroom)
    refill_rate: float     # tokens added per second
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._last_refill = time.monotonic()

    def consume(self, cost: float) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
            self._last_refill = now

            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False

    @property
    def available(self) -> float:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            return min(self.capacity, self._tokens + elapsed * self.refill_rate)


class RateLimiter:
    def __init__(
        self,
        default_capacity: float = 20_000,
        default_refill_rate: float = 10_000,
    ) -> None:
        self._default_capacity = default_capacity
        self._default_refill_rate = default_refill_rate
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _get_bucket(self, tenant_id: str) -> TokenBucket:
        with self._lock:
            if tenant_id not in self._buckets:
                self._buckets[tenant_id] = TokenBucket(
                    capacity=self._default_capacity,
                    refill_rate=self._default_refill_rate,
                )
            return self._buckets[tenant_id]

    def check_and_consume(
        self, tenant_id: str, prompt_tokens: int, max_response_tokens: int
    ) -> bool:
        """Pessimistic reservation: deduct full potential token cost upfront."""
        cost = float(prompt_tokens + max_response_tokens)
        return self._get_bucket(tenant_id).consume(cost)

    def configure_tenant(
        self, tenant_id: str, capacity: float, refill_rate: float
    ) -> None:
        with self._lock:
            self._buckets[tenant_id] = TokenBucket(
                capacity=capacity, refill_rate=refill_rate
            )
