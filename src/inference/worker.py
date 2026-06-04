import asyncio
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict

from src.inference.engine import BaseEngine, create_engine


@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int
    tenant_id: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0  # lower = higher priority


class InferenceWorker:
    """
    Wraps an inference engine and exposes generate() for streaming tokens.

    Continuous batching model: each call to generate() joins the engine's
    running batch at the next available iteration boundary. The engine does
    not wait for a full batch to form before beginning inference — new
    requests slot in as soon as a sequence position frees up.

    In production, vLLM handles this internally. This wrapper illustrates
    the interface contract: submit a request, receive an async token stream.
    """

    def __init__(self, worker_id: str, engine: BaseEngine = None) -> None:
        self._worker_id = worker_id
        self._engine = engine or create_engine(use_mock=True)
        self._active_requests: Dict[str, asyncio.Queue] = {}

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def active_count(self) -> int:
        return len(self._active_requests)

    async def generate(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """Stream tokens for the given request."""
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._active_requests[request.request_id] = token_queue

        task = asyncio.create_task(self._run(request, token_queue))

        try:
            while True:
                token = await token_queue.get()
                if token is None:  # EOS sentinel
                    break
                yield token
        finally:
            self._active_requests.pop(request.request_id, None)
            task.cancel()

    async def _run(self, request: InferenceRequest, queue: asyncio.Queue) -> None:
        try:
            async for token in self._engine.generate(request.prompt, request.max_tokens):
                await queue.put(token)
        except Exception as exc:
            # Surface the error as a visible token so the UI shows it
            await queue.put(f"\n\n**[Engine error: {exc}]**")
        finally:
            await queue.put(None)  # always send EOS
