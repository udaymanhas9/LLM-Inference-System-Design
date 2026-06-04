# Architecture — Component Deep Dive

## Ingress Tier

### Load Balancer
L7 proxy (Nginx or Envoy). Responsibilities:
- TLS termination
- Round-robin across gateway replicas
- Health-check-based failover
- Connection-level rate limiting (SYN flood protection)

Not custom. Commodity software. Replaced by a cloud LB (ALB, GCP LB) in managed deployments.

### Tenant-Aware Router (`src/gateway/router.py`)
First custom component. Two jobs:

**1. Tenant extraction**: Parse API key or JWT bearer token → extract `tenant_id`. This drives rate limiting and queue priority downstream.

**2. Cache-affinity routing**: Hash the first ~128 tokens of the prompt prefix. Route to the worker whose KV cache is most likely to hold matching prefix blocks (from earlier requests with the same system prompt). This is a soft hint — if the preferred worker is saturated, fall back to least-loaded.

Why prefix hashing instead of consistent hashing? Consistent hashing optimizes for even load distribution. We want the opposite for caching: maximize reuse on the same node. Prefix hashing with a saturation fallback gives us both — cache locality when possible, load balancing when necessary.

### Token-Cost Rate Limiter (`src/gateway/rate_limiter.py`)
Token-bucket algorithm, but denominated in **LLM tokens per second**, not HTTP requests per second.

Why tokens instead of RPS?
- A 4,000-token request consumes ~4× the GPU memory and compute of a 1,000-token request.
- An RPS bucket treats them identically and therefore undercharges large requests, enabling a single user to monopolize GPU capacity with a handful of long requests.
- Token-cost rate limiting correctly reflects actual GPU resource consumption.

Each tenant has an independent bucket. Bucket capacity = burst headroom (e.g., 20,000 tokens = 2 seconds of sustained throughput). Refill rate = sustained allowance (e.g., 10,000 tokens/s for a standard tier). Pessimistic reservation: full `prompt_tokens + max_response_tokens` is deducted upfront; unused response tokens are not refunded (acceptable for the burst-control use case; could be refined with actual token counts post-generation).

### Admission Queue (`src/gateway/admission.py`)
Asyncio priority queue. Decouples the ingress rate from the inference capacity.

Why is this important? Without a queue, any burst > (GPU throughput / avg latency) would either:
- Crash into the inference workers directly, inflating their queue depth and P99 latency
- Be immediately rejected, requiring clients to implement their own backoff

The admission queue absorbs short bursts gracefully. It returns 429 only when the queue itself is full — a signal that the burst is sustained, not momentary. The `Retry-After` header is set to a reasonable backoff estimate.

Priority levels allow tenants on higher tiers to pre-empt queue position (not GPU pre-emption, which is far too expensive). A high-priority tenant's requests sort to the front of the queue.

---

## Inference Tier

### PagedAttention (inside vLLM)
KV cache stored in non-contiguous virtual memory pages (like OS virtual memory). Benefits:
- No fragmentation: blocks are reclaimed immediately when sequences complete, without leaving stranded gaps
- Prefix sharing: sequences with identical prompt prefixes share the same KV blocks (copy-on-write), saving memory and prefill compute
- Higher concurrency: more requests fit in the same GPU memory footprint

### Continuous Batching (inside vLLM)
Traditional static batching waits for a full batch before running inference. Continuous batching processes a batch step at every iteration; as soon as one sequence finishes, a new one joins. GPU stays utilized without idle time between batches. This is the primary driver of high throughput at low latency.

### Chunked Prefill Scheduler (`src/inference/scheduler.py`)
Addresses prefill-decode interference. Each scheduler iteration:
1. Take at most `MAX_PREFILL_CHUNK_TOKENS` from the head prefill request
2. Run one decode step for all active decode sequences
3. Advance all sequences

Effect: no single prefill can monopolize more than `MAX_PREFILL_CHUNK_TOKENS` / `total_batch_tokens` of one iteration. Decode sequences see a bounded latency increase per iteration rather than a step-function stall. TTFT for the chunked request increases slightly (it takes N iterations instead of 1 to complete its prefill), but P95 decode latency drops across the system.

In production, this logic is implemented inside `vllm.core.scheduler.Scheduler`. The stub here exposes the same scheduling contract for illustration.

### Workers (`src/inference/worker.py` + `engine.py`)
Each worker is an independent process (one per GPU in production). Workers are stateless from the gateway's perspective — any worker can serve any request. The router's affinity hint increases KV cache hit rate but is never a hard constraint.

---

## Streaming Tier

### SSE Handler (`src/streaming/sse.py`)
Server-Sent Events over a persistent HTTP connection.

Why SSE over WebSockets?
- Token streaming is unidirectional (server → client). SSE is the right primitive for unidirectional push.
- SSE is HTTP/1.1 compatible, works through standard proxies, and has built-in reconnect semantics.
- WebSockets are bidirectional and require an upgrade handshake — unnecessary complexity for this use case.

Each request gets an `asyncio.Queue`. The worker pushes tokens to this queue as they are generated; the SSE handler drains it and flushes each token immediately with `\n\n`. Time-to-first-token (TTFT) is measured at the first flush — it captures the full end-to-end latency including queue wait, prefill, and network overhead.
