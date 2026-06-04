import json
import time
from typing import AsyncGenerator

from src.inference.worker import InferenceRequest, InferenceWorker


async def stream_sse(
    worker: InferenceWorker,
    request: InferenceRequest,
) -> AsyncGenerator[str, None]:
    """
    Wraps worker.generate() as a Server-Sent Events stream.

    Each token yields:  data: {"token": "...", "index": N, "ttft_ms": F}\\n\\n
    Final event:        data: [DONE]\\n\\n

    SSE (not WebSockets) because token delivery is unidirectional.
    The client receives tokens; it never sends anything mid-stream.
    SSE is simpler, HTTP/1.1 compatible, and auto-reconnects.

    TTFT (time-to-first-token) is measured here — it captures the full
    round-trip through the gateway and scheduler, not just engine latency.
    """
    first_token = True
    ttft_ms = 0.0
    start = time.monotonic()
    index = 0

    async for token in worker.generate(request):
        if first_token:
            ttft_ms = (time.monotonic() - start) * 1000.0
            first_token = False

        payload = json.dumps(
            {"token": token, "index": index, "ttft_ms": round(ttft_ms, 1)}
        )
        yield f"data: {payload}\n\n"
        index += 1

    yield "data: [DONE]\n\n"


def sse_error(message: str, code: int = 500) -> str:
    payload = json.dumps({"error": message, "code": code})
    return f"data: {payload}\n\n"
