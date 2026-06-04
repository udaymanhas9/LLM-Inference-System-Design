# Trade-offs — Extended Analysis

## What I chose and what I deliberately left out

---

### 1. Chunked Prefill vs. Prefill-Decode Disaggregation

**Chosen**: Chunked prefill on shared worker nodes.

**Not built**: Full prefill-decode disaggregation (Splitwise / Mooncake pattern — separate GPU pools, KV block transfer over NVLink or InfiniBand).

**Why**:
- Disaggregation introduces a separate fleet to provision, monitor, and autoscale. Two failure domains instead of one.
- KV block transfer over the network adds ~1–5 ms per request (est.) at ~1 GB/s NVLink bandwidth for a 560 KB KV payload — marginal for P95 but real.
- Chunked prefill recovers 80–90% of the latency benefit (decode sequences no longer blocked for the full prefill duration) at a fraction of the operational complexity.
- **Revisit trigger**: above ~20k RPS or if the prefill workload becomes highly variable (mixture of 50-token and 2,000-token prompts in the same batch).

---

### 2. In-Process Priority Queue vs. Kafka / Redis Streams

**Chosen**: `asyncio.PriorityQueue` per gateway process.

**Not built**: Durable distributed queue (Kafka, Redis Streams).

**Why**:
- Kafka adds ~2–10 ms P99 latency per hop. For a P95 < 2 s SLA this is manageable, but it's wasted overhead when the queue doesn't need durability.
- At 5k RPS with ~80 concurrent requests per GPU, the queue depth is small enough to fit comfortably in process memory.
- An in-process queue fails over cleanly: when a gateway pod restarts, the client gets a 503 and retries. Queue state was ephemeral anyway.
- **Revisit trigger**: need cross-AZ queue survival, want centralized queue depth metrics across all gateway pods, or need queue replay on partial failures.

---

### 3. Routing-Based KV Locality vs. Distributed KV Cache

**Chosen**: Cache-affinity routing (SHA256 prefix hash → sticky worker).

**Not built**: Cross-node distributed KV cache (e.g., Redis cluster storing KV blocks, Mooncake's disaggregated memory pool).

**Why**:
- Transferring a 560 KB KV block from a remote node costs more than a cache miss (which triggers a fresh prefill in ~25 ms at 20k tok/s).
- Routing affinity achieves cache locality for free — no extra infrastructure, no serialization overhead.
- The main case where this breaks down: very long system prompts (>2k tokens) shared across all tenant users, where the recompute cost becomes significant. At that point, a separate KV cache tier (like vLLM's experimental prefix caching) is worth evaluating.

---

### 4. vLLM vs. TensorRT-LLM

**Chosen**: vLLM.

**Not evaluated**: TensorRT-LLM, TGI (Hugging Face).

**Why**:
- vLLM has the most complete PagedAttention + continuous batching implementation with active OSS maintenance and a Python-native API that matches this design.
- TRT-LLM has ~20–30% higher peak throughput on NVIDIA hardware but requires NVIDIA's toolchain (model compilation, plugin management, CUDA dependency pinning) which adds weeks of integration work and rebuild cycles on every model update.
- TGI is more deployment-friendly but lags vLLM on PagedAttention efficiency.
- **Revisit trigger**: if peak throughput becomes the bottleneck after scaling horizontally. TRT-LLM's gains are real at extreme scale.

---

### 5. Speculative Decoding

**Not built**: Draft model for speculative decoding.

**Why**:
- Speculative decoding can 2× decode throughput for tasks with predictable output (code completion, structured output).
- Requires a second draft model (~1–7B), acceptance-rate tuning, and additional memory. Adds operational complexity before the baseline is stable.
- **Revisit trigger**: after the baseline system is running and P50 throughput is the constraint, not P95 latency.

---

### 6. Request Cancellation / Partial Streaming

**Not built**: Explicit client disconnect → immediate GPU work cancellation.

**Why**:
- vLLM supports cancellation via `engine.abort(request_id)`. The SSE handler could hook into `asyncio` cancellation to propagate this.
- Kept out of scope to avoid complicating the illustration code. In production this is important for GPU efficiency — cancelled requests consuming KV cache memory and GPU cycles is wasteful.
