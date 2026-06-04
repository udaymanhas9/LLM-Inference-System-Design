# Fastino Labs — LLM Inference System Design

> **Submission for:** Fastino Labs Take-Home Assignment  
> **Scope:** Design document + illustrative Python skeleton (runs locally on CPU/M4 with a mock engine)

---

## Assumptions

*All numbers below are back-of-envelope estimates. Marked with (est.) where derived.*

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | 13B parameters, fp16 | Given |
| GPU (production) | NVIDIA H100 80GB SXM | Industry standard for this scale |
| Model weight footprint | ~26 GB fp16 (est.) | 13B × 2 bytes |
| KV cache per token | ~1.6 KB/token (est.) | 40 layers × 2 (K+V) × 128 head_dim × 2 bytes |
| KV memory per request | ~1.1 GB (est.) | (500 prompt + 200 response) × 1.6 KB |
| Available KV memory per H100 | ~50 GB (est.) | 80 GB − 26 GB weights − 4 GB overhead |
| Concurrent requests per H100 | ~45 (est.) | 50 GB ÷ 1.1 GB, PagedAttention |
| Decode throughput per H100 | ~2,000 tokens/s (est.) | Memory-bandwidth bound |
| Avg generation time per request | ~0.1 s (est.) | 200 tokens ÷ 2,000 tok/s |
| Requests in flight at steady state | ~500 (est.) | Little's Law: 5,000 RPS × 0.1 s |
| GPUs needed | ~12 (est.) | 500 in-flight ÷ 45 per GPU, round up |
| Target SLA | P95 TTFT + streaming < 2 s | Given |
| Multi-tenant | Yes — token-bucket per tenant | Given |

> GPU count is sensitive to batch efficiency. Full sizing math in `docs/sizing.md`.

---

## Architecture

```
  Clients
     │  HTTPS POST /v1/completions
     ▼
┌────────────────────────────────────────────────────────────────────┐
│                         INGRESS TIER                               │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Load Balancer (Nginx / Envoy)                              │  │
│  │  • TLS termination  • Round-robin across gateway replicas   │  │
│  └────────────────────────┬────────────────────────────────────┘  │
│                           │                                        │
│  ┌────────────────────────▼────────────────────────────────────┐  │
│  │  Tenant-Aware Router           [src/gateway/router.py]      │  │
│  │  • API key / JWT → tenant_id                                │  │
│  │  • SHA256(prompt prefix) → cache-affinity worker selection  │  │
│  │  • Falls back to least-loaded worker on saturation          │  │
│  └────────────────────────┬────────────────────────────────────┘  │
│                           │                                        │
│  ┌────────────────────────▼────────────────────────────────────┐  │
│  │  Token-Cost Rate Limiter       [src/gateway/rate_limiter.py]│  │
│  │  • Token-bucket per tenant, denominated in tokens/s         │  │
│  │  • Cost = prompt_tokens + max_response_tokens (pessimistic) │  │
│  │  • Expensive requests correctly cost more than cheap ones   │  │
│  └────────────────────────┬────────────────────────────────────┘  │
│                           │                                        │
│  ┌────────────────────────▼────────────────────────────────────┐  │
│  │  Admission Queue / Burst Buffer [src/gateway/admission.py]  │  │
│  │  • Asyncio priority queue, per-tenant priority tiers        │  │
│  │  • Decouples ingress spikes from inference capacity         │  │
│  │  • Returns 429 + Retry-After when depth > MAX_QUEUE_DEPTH   │  │
│  └────────────────────────┬────────────────────────────────────┘  │
└───────────────────────────┼────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────┐
│                        INFERENCE TIER                               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Cache-Aware Router                                          │  │
│  │  Routes by prefix hash → warm KV cache for shared prompts   │  │
│  └──────┬──────────────┬──────────────┬──────────────┬─────────┘  │
│         │              │              │              │             │
│  ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────┐ ┌───▼──────┐     │
│  │  Worker 0   │ │  Worker 1  │ │  Worker 2 │ │ Worker N │     │
│  │  vLLM +     │ │  vLLM +    │ │  vLLM +   │ │  ...     │     │
│  │ PagedAttn   │ │ PagedAttn  │ │ PagedAttn │ │          │     │
│  │ Cont. Batch │ │ Cont.Batch │ │Cont.Batch │ │          │     │
│  └──────┬──────┘ └─────┬──────┘ └────┬──────┘ └───┬──────┘     │
│         └──────────────┴──────────────┴─────────────┘            │
│                                │                                   │
│  ┌─────────────────────────────▼────────────────────────────────┐ │
│  │  Chunked Prefill Scheduler      [src/inference/scheduler.py] │ │
│  │  • Slices prefill into N-token chunks per iteration          │ │
│  │  • Interleaves decode steps between chunks                   │ │
│  │  • Prevents long prefills from stalling in-flight sequences  │ │
│  └──────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬────────────────────────────────────────┘
                            │ token stream
┌───────────────────────────▼────────────────────────────────────────┐
│                        STREAMING TIER                               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  SSE Handler                        [src/streaming/sse.py]   │  │
│  │  • Per-request asyncio queue                                 │  │
│  │  • Streams tokens as text/event-stream (TTFT measured here)  │  │
│  │  • EOS token → data: [DONE] → connection closed              │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Load Balancer
Standard L7 LB (Nginx/Envoy). Terminates TLS, distributes across gateway replicas. Commodity — no custom logic here.

### 2. Tenant-Aware Router (`src/gateway/router.py`)
Extracts `tenant_id` from API key or JWT. Routes using **cache-affinity hashing**: `SHA256(first ~128 tokens of prompt) % num_workers` selects the target worker. The goal is to reuse warm KV-cache blocks for repeated system prompts across a tenant's users (common in multi-tenant SaaS products where every user shares a long system prompt). If the affinity target is saturated, falls back to least-loaded.

### 3. Token-Cost Rate Limiter (`src/gateway/rate_limiter.py`)
Token-bucket per tenant, denominated in **tokens/second** — not RPS. A request costs `prompt_tokens + max_response_tokens` upfront (pessimistic reservation). This is the right unit: a 4,000-token request costs 4× a 1,000-token request, correctly reflecting its GPU impact. Refill rate is configurable per tier. Per-tenant isolation prevents one noisy tenant from exhausting the system.

### 4. Admission Queue / Burst Buffer (`src/gateway/admission.py`)
Asyncio priority queue that decouples ingress spikes from inference capacity. When a burst arrives, requests queue here rather than being immediately rejected or overloading workers. Queue depth is monitored — returns 429 with `Retry-After` when depth exceeds `MAX_QUEUE_DEPTH`. Provides natural backpressure.

### 5. Inference Workers (`src/inference/worker.py` + `engine.py`)
Each worker wraps vLLM (production) or a mock generator (local demo). Two key techniques:
- **PagedAttention**: KV cache stored in non-contiguous memory blocks (like virtual memory pages). Eliminates fragmentation — multiple requests can share prefix blocks, and blocks are reclaimed immediately on request completion.
- **Continuous batching**: New requests join the batch at any iteration boundary, not just when a full batch slots become free. GPU utilization stays high without waiting.

### 6. Chunked Prefill Scheduler (`src/inference/scheduler.py`)
Core production problem: a 500-token prefill monopolizes the GPU for one full forward pass, blocking all in-flight decode sequences for that iteration. For 5k RPS, this creates visible latency spikes for concurrent users.

Solution: prefill requests are split into chunks of `MAX_PREFILL_CHUNK_TOKENS`. Each scheduler iteration processes one chunk plus all active decode sequences. TTFT increases slightly for the chunked request, but P95 decode latency drops significantly across the system. In production, this logic lives inside vLLM's scheduler; the stub here illustrates the scheduling model.

### 7. SSE Streaming Handler (`src/streaming/sse.py`)
Each request gets an async generator. The worker pushes tokens to a per-request `asyncio.Queue`; the SSE handler drains it and flushes each token immediately as `data: {"token": "...", "index": N}\n\n`. TTFT is measured at the first token emit. Connection stays open until EOS or client disconnect. Using Server-Sent Events (not WebSockets) because the communication is unidirectional — tokens flow one way.

---

## Request Flow

1. **Client → Load Balancer**: `POST /v1/completions` with `Authorization: Bearer <api_key>`. TLS terminated.
2. **LB → Gateway Router**: Round-robin to a gateway replica. Router validates API key, extracts `tenant_id`.
3. **Cache-Affinity Routing**: `SHA256(prompt[:512])` selects preferred worker. If saturated, routes to least-loaded.
4. **Rate Limit Check**: Deducts `prompt_tokens + max_tokens` from tenant's token bucket. If insufficient → `429 Too Many Requests`.
5. **Admission Queue**: Enqueues with tenant priority. If queue depth ≥ `MAX_QUEUE_DEPTH` → `429` with `Retry-After: N`.
6. **Worker Receives Request**: Joins the continuous batch. Prefill is chunked if `prompt_tokens > MAX_PREFILL_CHUNK`.
7. **Chunked Prefill**: Scheduler interleaves prefill chunks with decode steps across iterations.
8. **Decode**: Autoregressive token generation. Each token pushed to per-request `asyncio.Queue`.
9. **SSE Stream**: Handler drains queue, flushes each token as `data: {…}` event. TTFT logged on first token.
10. **Completion**: Worker sends EOS sentinel. SSE handler emits `data: [DONE]`, closes connection.

---

## Scaling Challenges

### 1. Prefill-Decode Interference (Primary Bottleneck)

Prefill is **compute-bound** (processes all prompt tokens in a single parallel forward pass). Decode is **memory-bandwidth-bound** (one token per step). On a shared worker, a large prefill (500+ tokens) blocks all in-flight decode sequences for that full iteration — directly inflating their P95 latency.

**Mitigation ladder:**
| Level | Technique | Implemented |
|-------|-----------|-------------|
| 1 | Continuous batching — new requests enter mid-batch, GPU never idles | Yes (via vLLM) |
| 2 | Chunked prefill — caps prefill work per iteration, limits decode interference | Yes (scheduler stub) |
| 3 | Prefill-decode disaggregation — separate GPU pools for each phase | No — see Trade-offs |

### 2. KV Cache Memory Pressure

At 5k RPS with 700 avg tokens/request: steady-state ~500 concurrent requests × 1.1 GB/request ≈ **550 GB KV cache** needed across the fleet (est.). This drives the GPU count more than compute. PagedAttention's block allocator prevents fragmentation from stranded memory between prompt and response phases.

**Mitigation**: PagedAttention block management + prefix caching (shared KV blocks for identical system prompts across a tenant).

### 3. Multi-Tenant Fairness Under Bursts

Without per-tenant queuing, a single tenant bursting 1,000 RPS can flood the batch, causing P99 spikes for all other tenants — even if they're well within their quota.

**Mitigation**: Per-tenant token bucket prevents burst propagation. Per-tenant priority queue allows pre-agreed SLAs (high-tier tenants get lower queue priority numbers = dequeued first).

### 4. Head-of-Line Blocking

Long-running requests (high `max_tokens`) occupy batch slots far longer than average, reducing effective batch size for shorter requests.

**Mitigation**: Per-request `max_tokens` cap enforced at admission. Lower-tier tenants get stricter caps. Future: speculative execution or output-length prediction for smarter scheduling.

---

## Trade-offs

| Decision | Chosen | Not Built | Why |
|----------|--------|-----------|-----|
| **Prefill disaggregation** | Chunked prefill on shared workers | Splitwise/Mooncake pattern: separate prefill GPU pool | 3–5× infra complexity, separate fleet to manage, network transfer of KV blocks (~GB/req). Chunked prefill recovers most of the benefit at 1/10th the ops cost. Revisit above 20k RPS. |
| **Queue implementation** | Asyncio in-process priority queue | Kafka / Redis Streams for durable token routing | Kafka adds ~5ms P99 latency per hop and significant ops overhead. At 5k RPS, an in-process bounded queue is sufficient. Add durable queue when multi-AZ queue survival is required. |
| **KV cache sharing** | Prefix-hash affinity routing (soft locality) | Distributed KV cache (e.g., cross-node Redis/Mooncake) | Network round-trip for KV blocks (MBs) likely costs more than cache miss. Routing affinity achieves cache locality without network overhead. Revisit if tenant system prompts are very long (>2k tokens). |
| **Load balancer** | Commodity L7 (Nginx/Envoy) | Custom ML-aware LB (route by estimated decode length) | Estimated decode length is noisy and adds latency to the hot path. Commodity LB is battle-tested. Routing affinity already handles the main ML-specific routing concern. |
| **Serving framework** | vLLM (prod) / mock (local) | TensorRT-LLM, TGI | vLLM has the best PagedAttention + continuous batching OSS implementation with active maintenance. TRT-LLM has higher peak throughput but requires NVIDIA toolchain investment and rebuild cycles per model. |
| **Speculative decoding** | Not implemented | Draft model for speculative decoding | Meaningful throughput gain (~2×) at the cost of a second model, memory overhead, and acceptance-rate tuning. Worth adding after the baseline is stable. |

---

## Local Demo

This repo runs a **`MockEngine`** (deterministic token generator with simulated latency) on CPU. It illustrates the full flow — ingress → rate limit → schedule → stream — without claiming to benchmark production throughput.

**Production** would replace `MockEngine` with `VLLMEngine` wrapping `vllm.AsyncLLMEngine` on H100s.

```bash
# Install minimal dependencies (no GPU required)
make install

# Run 20-request mock simulation
make demo

# Run 50-request simulation
make sim
```

Expected local output:
```
=== Fastino Labs LLM Inference Demo ===
Requests: 20  Concurrency: 5
Engine: MockEngine (local demo — not a benchmark)

Results (18 completed, 2 rate-limited):
  TTFT p50: 201.3ms
  TTFT p95: 215.7ms
  Wall time: 4.21s
  Throughput: 4.3 req/s (mock, local)

Note: This is a local mock. Production targets 5,000 RPS on H100 GPUs.
```

---

## Repository Structure

```
fastino-labs/
├── README.md                    ← This file (primary deliverable)
├── docs/
│   ├── architecture.md          ← Deeper component descriptions
│   ├── sizing.md                ← Back-of-envelope GPU count math
│   └── tradeoffs.md             ← Extended trade-off analysis
├── src/
│   ├── config.py                ← Model, batch, rate-limit parameters
│   ├── gateway/
│   │   ├── router.py            ← Tenant routing + cache-affinity
│   │   ├── rate_limiter.py      ← Token-bucket (token-cost-based)
│   │   └── admission.py         ← Burst buffer / priority queue
│   ├── inference/
│   │   ├── engine.py            ← MockEngine (local) + VLLMEngine (prod)
│   │   ├── worker.py            ← Continuous-batching wrapper
│   │   └── scheduler.py         ← Chunked-prefill scheduler stub
│   └── streaming/
│       └── sse.py               ← Server-Sent Events handler
├── sim/
│   └── load_sim.py              ← Local load generator (illustrative)
├── requirements.txt
└── Makefile
```

See `docs/sizing.md` for the GPU count derivation and `docs/tradeoffs.md` for extended discussion.
