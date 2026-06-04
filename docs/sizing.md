# GPU Sizing — Back-of-Envelope

> All numbers are estimates. Assumptions stated inline. Model: 13B fp16 on H100 80GB SXM.

## 1. Model Weight Footprint

```
13B parameters × 2 bytes (fp16) = 26 GB
```

## 2. KV Cache per Token

```
KV cache / token = 2 (K+V) × num_layers × head_dim × num_heads × bytes_per_element
                 = 2 × 40 layers × 128 head_dim × (5120 / 128 heads = 40 heads) × 2 bytes
                 ≈ 2 × 40 × 128 × 40 × 2
                 ≈ 819,200 bytes ≈ 0.8 KB / token  (est.)
```

*Note: exact KV dimensions depend on the specific 13B architecture (Llama-2 used here as proxy).*

## 3. KV Cache per Request

```
avg prompt + response = 500 + 200 = 700 tokens
KV memory / request = 700 × 0.8 KB ≈ 560 KB  (est.)
```

## 4. Available KV Memory per H100

```
H100 80 GB
− 26 GB  model weights
−  4 GB  CUDA overhead, activations, misc
= 50 GB  available for KV cache  (est.)
```

## 5. Concurrent Requests per H100

```
50 GB ÷ 560 KB/request ≈ 90 concurrent requests  (est.)
```

PagedAttention's block allocator improves on this by ~10–20% vs. contiguous allocation (no fragmentation waste). Round down to **~80** for headroom.

## 6. Steady-State Requests in Flight (Little's Law)

```
L = λ × W
λ = 5,000 req/s
W = avg service time ≈ prefill_time + decode_time
  = (500 tokens / 20,000 tok/s prefill) + (200 tokens / 2,000 tok/s decode)
  = 0.025 s + 0.1 s = 0.125 s  (est.)

L = 5,000 × 0.125 = 625 requests in flight  (est.)
```

*Throughput estimates: prefill ~20k tok/s, decode ~2k tok/s per H100 at batch=64 (est.)*

## 7. GPU Count

```
GPUs = ceil(L / concurrent_per_GPU) = ceil(625 / 80) ≈ 8 GPUs  (est., lower bound)
```

Add ~50% headroom for burst, rolling restarts, and batch efficiency variance:

```
Production target: ~12–16 H100s  (est.)
```

## 8. P95 Latency Budget

```
Target: P95 < 2 s end-to-end

TTFT (prefill):          ~25–200 ms  (chunked prefill; depends on queue depth)
Decode (200 tokens):     ~100 ms     (200 / 2,000 tok/s per H100)
Network + gateway:       ~10 ms      (est.)
Queue wait at P95:       ~500 ms     (allowance for burst)
─────────────────────────────────────
Total P95 budget:        ~635 ms     (well within 2 s target if queue is managed)
```

The P95 target is achievable if admission control keeps queue depth bounded and chunked prefill keeps per-iteration TTFT under ~200 ms.

## 9. Sensitivity

| Assumption changed | Effect |
|--------------------|--------|
| avg response 400 tok (2×) | +50% KV memory → +6 GPUs |
| batch efficiency 60% | −20% concurrency → +3 GPUs |
| prefill throughput 10k tok/s | TTFT doubles; still < 500 ms |
| 10k RPS (2× traffic) | ~25–30 H100s |
