#!/usr/bin/env python3
"""
Local load simulator — illustrates the request flow end-to-end.

This is NOT a benchmark of 5,000 RPS. It runs MockEngine on CPU to show:
  ingress → rate limit check → worker routing → SSE token stream

Production would run vLLM on H100s with real throughput.
"""
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

# Allow running as `python sim/load_sim.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gateway.rate_limiter import RateLimiter
from src.inference.engine import OllamaEngine, create_engine
from src.inference.worker import InferenceRequest, InferenceWorker
from src.streaming.sse import stream_sse

_SAMPLE_PROMPTS = [
    "Explain the concept of PagedAttention in LLM inference systems.",
    "What are the trade-offs between continuous batching and static batching?",
    "Describe how token streaming works in a production inference API.",
    "How does chunked prefill reduce decode latency interference?",
    "What is the role of a KV cache in autoregressive language model decoding?",
    "Summarize the Splitwise prefill-decode disaggregation paper.",
    "What does Little's Law tell us about GPU sizing for inference?",
]

_TENANTS = ["tenant_a", "tenant_b", "tenant_c"]


async def simulate_one(
    tenant_id: str,
    prompt: str,
    rate_limiter: RateLimiter,
    worker: InferenceWorker,
    max_tokens: int = 80,
) -> dict:
    prompt_tokens = len(prompt.split()) * 2  # rough token estimate

    if not rate_limiter.check_and_consume(tenant_id, prompt_tokens, max_tokens):
        return {"status": "rate_limited", "tenant_id": tenant_id}

    request = InferenceRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        tenant_id=tenant_id,
    )

    start = time.monotonic()
    ttft_ms: float | None = None
    token_count = 0

    async for event in stream_sse(worker, request):
        if event.strip() == "data: [DONE]":
            break
        if event.startswith("data: "):
            try:
                data = json.loads(event[6:])
                if ttft_ms is None:
                    ttft_ms = data.get("ttft_ms")
                token_count += 1
            except (json.JSONDecodeError, KeyError):
                pass

    total_ms = (time.monotonic() - start) * 1000.0
    return {
        "status": "ok",
        "tenant_id": tenant_id,
        "ttft_ms": ttft_ms,
        "total_ms": round(total_ms, 1),
        "tokens": token_count,
    }


async def run_simulation(
    n_requests: int = 20,
    concurrency: int = 5,
    ollama_model: str | None = None,
) -> None:
    if ollama_model:
        engine = OllamaEngine(model=ollama_model)
        engine_label = f"OllamaEngine ({ollama_model}) — REAL inference"
        # Real models are slow; keep concurrency low to avoid overloading Ollama
        concurrency = min(concurrency, 2)
    else:
        engine = create_engine(use_mock=True)
        engine_label = "MockEngine (synthetic tokens — not real inference)"

    print(f"\n=== Fastino Labs LLM Inference Demo ===")
    print(f"Requests:    {n_requests}   Concurrency: {concurrency}")
    print(f"Engine:      {engine_label}\n")

    rate_limiter = RateLimiter()
    worker = InferenceWorker(worker_id="worker-0", engine=engine)

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(coro):
        async with semaphore:
            return await coro

    tasks = [
        bounded(
            simulate_one(
                tenant_id=_TENANTS[i % len(_TENANTS)],
                prompt=_SAMPLE_PROMPTS[i % len(_SAMPLE_PROMPTS)],
                rate_limiter=rate_limiter,
                worker=worker,
            )
        )
        for i in range(n_requests)
    ]

    wall_start = time.monotonic()
    results = await asyncio.gather(*tasks)
    wall_elapsed = time.monotonic() - wall_start

    ok = [r for r in results if r["status"] == "ok"]
    rate_limited = [r for r in results if r["status"] == "rate_limited"]

    print(f"Results: {len(ok)} completed, {len(rate_limited)} rate-limited\n")

    if ok:
        ttfts = sorted(r["ttft_ms"] for r in ok if r["ttft_ms"] is not None)
        if ttfts:
            p50 = statistics.median(ttfts)
            p95 = ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.95))]
            print(f"  TTFT  p50: {p50:.1f} ms")
            print(f"  TTFT  p95: {p95:.1f} ms")
            print(f"  TTFT  max: {max(ttfts):.1f} ms")

        print(f"  Wall time: {wall_elapsed:.2f} s")
        print(f"  Throughput: {len(ok) / wall_elapsed:.1f} req/s  (mock, local)\n")

    print("Note: production targets 5,000 RPS on H100 GPUs (see docs/sizing.md)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fastino LLM inference demo")
    parser.add_argument("n", nargs="?", type=int, default=5, help="Number of requests")
    parser.add_argument(
        "--model",
        metavar="NAME",
        help="Ollama model to use for real inference (e.g. gemma4:26b, qwen2.5-coder:14b). "
             "Omit to use MockEngine.",
    )
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(
        run_simulation(
            n_requests=args.n,
            concurrency=args.concurrency,
            ollama_model=args.model,
        )
    )
