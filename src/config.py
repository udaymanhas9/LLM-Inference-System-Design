from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "mock"  # "meta-llama/Llama-2-13b" in prod
    max_tokens: int = 2048
    avg_prompt_tokens: int = 500
    avg_response_tokens: int = 200
    tensor_parallel_size: int = 1


@dataclass
class BatchConfig:
    max_batch_size: int = 64
    max_prefill_chunk_tokens: int = 256  # chunked prefill chunk size
    continuous_batching: bool = True


@dataclass
class RateLimitConfig:
    # Token-bucket denominated in tokens/s, not RPS
    default_capacity_tokens: float = 20_000
    default_refill_rate_tps: float = 10_000
    burst_multiplier: float = 2.0


@dataclass
class AdmissionConfig:
    max_queue_depth: int = 1_000
    request_timeout_s: float = 30.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)
    use_mock_engine: bool = True  # False in production (requires vllm + GPU)


DEFAULT_CONFIG = Config()
