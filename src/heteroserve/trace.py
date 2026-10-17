from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .schema import RequestSpec


@dataclass(frozen=True)
class TraceConfig:
    requests: int = 100
    seed: int = 17
    arrival_rate: float = 2.0
    long_prompt_fraction: float = 0.5
    long_output_fraction: float = 0.5
    shapes: tuple[tuple[int, int, float], ...] = ()
    arrival_pattern: str = "poisson"
    burst_factor: float = 4.0
    burst_period_s: float = 10.0

    def __post_init__(self) -> None:
        if self.requests < 1 or self.arrival_rate <= 0:
            raise ValueError("requests and arrival rate must be positive")
        if not 0 <= self.long_prompt_fraction <= 1 or not 0 <= self.long_output_fraction <= 1:
            raise ValueError("fractions must be in [0, 1]")
        if self.arrival_pattern not in {"poisson", "bursty"}:
            raise ValueError("arrival pattern must be poisson or bursty")
        if self.burst_factor < 1 or self.burst_period_s <= 0:
            raise ValueError("burst factor must be >= 1 and period must be positive")
        if self.shapes and (any(p < 1 or o < 1 or w <= 0 for p, o, w in self.shapes)):
            raise ValueError("shape tokens and weights must be positive")


def generate(config: TraceConfig) -> list[RequestSpec]:
    rng = random.Random(config.seed)
    arrival = 0.0
    trace: list[RequestSpec] = []
    for index in range(config.requests):
        rate = config.arrival_rate
        if config.arrival_pattern == "bursty":
            phase = int(arrival / config.burst_period_s) % 2
            rate *= config.burst_factor if phase == 0 else 1 / config.burst_factor
        arrival += -math.log1p(-rng.random()) / rate
        if config.shapes:
            prompt, output, _ = rng.choices(config.shapes, weights=[s[2] for s in config.shapes])[0]
        else:
            prompt = rng.randint(1024, 4096) if rng.random() < config.long_prompt_fraction else rng.randint(16, 256)
            output = rng.randint(128, 512) if rng.random() < config.long_output_fraction else rng.randint(8, 64)
        trace.append(RequestSpec(f"req-{index:06d}", arrival, prompt, output, config.seed + index))
    return trace


def write_jsonl(trace: list[RequestSpec], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in trace))
