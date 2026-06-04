import asyncio
import json
import random
from abc import ABC, abstractmethod
from typing import AsyncGenerator

# Vocabulary used by the mock generator to produce plausible-looking tokens
_MOCK_VOCAB = [
    "The", "inference", "system", "processes", "tokens", "efficiently", "using",
    "PagedAttention", "and", "continuous", "batching", "to", "maximize", "GPU",
    "utilization", "while", "maintaining", "low", "latency", "for", "all", "tenants",
    ".", "The", "chunked", "prefill", "scheduler", "interleaves", "decode", "steps",
    "with", "prefill", "chunks", "to", "prevent", "starvation", "of", "active", "sequences",
]


class BaseEngine(ABC):
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
        ...


class MockEngine(BaseEngine):
    """
    Deterministic mock for local demo on CPU / M4.
    Simulates realistic TTFT and inter-token delay without requiring a GPU.

    Production: replace with VLLMEngine.
    """

    def __init__(
        self,
        tokens_per_second: float = 50.0,
        ttft_ms: float = 200.0,
        seed: int = 42,
    ) -> None:
        self._tokens_per_second = tokens_per_second
        self._ttft_ms = ttft_ms
        self._rng = random.Random(seed)

    async def generate(self, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
        await asyncio.sleep(self._ttft_ms / 1000.0)  # simulate prefill latency

        n_tokens = min(max_tokens, self._rng.randint(30, 80))
        delay = 1.0 / self._tokens_per_second

        for i in range(n_tokens):
            word = self._rng.choice(_MOCK_VOCAB)
            # Space-prefix like a real SentencePiece tokeniser (▁word → " word")
            yield (" " + word) if i > 0 else word
            await asyncio.sleep(delay)


class VLLMEngine(BaseEngine):
    """
    Production engine wrapping vllm.AsyncLLMEngine.

    Requires: pip install vllm  (CUDA environment, Linux/WSL2 only)
    Not usable on macOS M4 — use MockEngine locally.
    """

    def __init__(self, model: str, tensor_parallel_size: int = 1) -> None:
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            args = AsyncEngineArgs(model=model, tensor_parallel_size=tensor_parallel_size)
            self._engine = AsyncLLMEngine.from_engine_args(args)

            from vllm import SamplingParams
            self._SamplingParams = SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vllm not installed or not available on this platform. "
                "Use MockEngine for local development."
            ) from exc

    async def generate(self, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
        import uuid
        params = self._SamplingParams(max_tokens=max_tokens)
        request_id = str(uuid.uuid4())

        prev_len = 0
        async for output in self._engine.generate(prompt, params, request_id=request_id):
            if output.outputs:
                text = output.outputs[0].text
                delta = text[prev_len:]
                if delta:
                    yield delta
                prev_len = len(text)


class OllamaEngine(BaseEngine):
    """
    Streams tokens from a locally running Ollama instance via /api/chat.

    /api/chat (not /api/generate) is used because chat models like Gemma 4
    need the proper chat template applied — /api/generate sends raw text and
    skips template formatting, which causes garbled or empty responses.

    Start Ollama first:  ollama serve
    """

    OLLAMA_URL = "http://localhost:11434/api/chat"

    def __init__(self, model: str = "gemma4:26b", base_url: str = OLLAMA_URL) -> None:
        self._model = model
        self._base_url = base_url

    async def generate(self, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("pip install httpx  (required for OllamaEngine)") from exc

        # /api/chat expects a messages array; wrap the pre-formatted prompt as
        # a single user message so the chat template is applied correctly.
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"num_predict": max_tokens},
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", self._base_url, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise RuntimeError(
                        f"Ollama {resp.status_code}: {body.decode()[:200]}. "
                        "Is `ollama serve` running?"
                    )
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    # /api/chat token is in message.content, not response
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break


class LlamaCppEngine(BaseEngine):
    """
    Streams tokens through our own C++ inference library (src/cpp/inference_lib.cpp).

    This exercises the actual chunked-prefill and decode loop we wrote — NOT
    a black box.  Requires building the shared library first:

        make build-cpp

    Then point at any GGUF model file:

        python sim/load_sim.py 3 --model llama.cpp:/path/to/model.gguf

    The C library runs in a thread-pool thread; asyncio is never blocked.
    A semaphore serialises concurrent requests to the single llama_context
    (llama.cpp contexts are not thread-safe).
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,       # -1 = all layers on Metal
        n_ctx: int = 4096,
        prefill_chunk: int = 256,
        temperature: float = 0.7,
    ) -> None:
        import ctypes, pathlib, platform

        ext = ".dylib" if platform.system() == "Darwin" else ".so"
        lib_path = pathlib.Path(__file__).parent.parent.parent / f"libinference{ext}"
        if not lib_path.exists():
            raise RuntimeError(
                f"Shared library not found at {lib_path}.\n"
                "Run:  make build-cpp"
            )

        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_ffi()

        self._h = self._lib.inference_create(
            model_path.encode(),
            ctypes.c_int(n_gpu_layers),
            ctypes.c_int(n_ctx),
        )
        if not self._h:
            raise RuntimeError(f"Failed to load model: {model_path}")

        self._chunk = prefill_chunk
        self._temperature = temperature
        self._sem = asyncio.Semaphore(1)  # one generation at a time per context

    def _setup_ffi(self) -> None:
        import ctypes
        TOKEN_CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
        self._TOKEN_CB = TOKEN_CB

        self._lib.inference_create.restype  = ctypes.c_void_p
        self._lib.inference_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

        self._lib.inference_destroy.restype  = None
        self._lib.inference_destroy.argtypes = [ctypes.c_void_p]

        self._lib.inference_generate.restype  = ctypes.c_int
        self._lib.inference_generate.argtypes = [
            ctypes.c_void_p,  # ctx
            ctypes.c_char_p,  # prompt
            ctypes.c_int,     # max_tokens
            ctypes.c_int,     # prefill_chunk_size
            ctypes.c_float,   # temperature
            TOKEN_CB,         # callback
            ctypes.c_void_p,  # userdata
        ]

    def __del__(self) -> None:
        if hasattr(self, "_lib") and hasattr(self, "_h") and self._h:
            self._lib.inference_destroy(self._h)

    async def generate(self, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
        import ctypes

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def on_token(text_bytes: bytes, is_done: int, _: ctypes.c_void_p) -> None:
            if is_done:
                loop.call_soon_threadsafe(queue.put_nowait, None)
            else:
                token = text_bytes.decode("utf-8", errors="replace")
                loop.call_soon_threadsafe(queue.put_nowait, token)

        cb = self._TOKEN_CB(on_token)

        async with self._sem:
            fut = loop.run_in_executor(
                None,
                lambda: self._lib.inference_generate(
                    self._h,
                    prompt.encode(),
                    ctypes.c_int(max_tokens),
                    ctypes.c_int(self._chunk),
                    ctypes.c_float(self._temperature),
                    cb,
                    None,
                ),
            )

            while True:
                token = await queue.get()
                if token is None:
                    break
                yield token

            await fut  # surface any C-side errors


def create_engine(use_mock: bool = True, **kwargs) -> BaseEngine:
    if use_mock:
        return MockEngine(**kwargs)
    backend = kwargs.pop("backend", "vllm")
    if backend == "ollama":
        model = kwargs.pop("model", "gemma4:26b")
        return OllamaEngine(model=model, **kwargs)
    if backend == "llama.cpp":
        return LlamaCppEngine(**kwargs)
    model = kwargs.pop("model", "meta-llama/Llama-2-13b-chat-hf")
    return VLLMEngine(model=model, **kwargs)
