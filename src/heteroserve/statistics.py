"""Run-level paired statistics; requests are never treated as independent trials."""

from __future__ import annotations

import random
from statistics import median


def paired_bootstrap_delta(
    baseline: list[float],
    policy: list[float],
    *,
    seed: int = 17,
    samples: int = 10_000,
) -> dict[str, float | int]:
    if len(baseline) != len(policy) or not baseline:
        raise ValueError("paired nonempty run vectors are required")
    if samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    deltas = [candidate - reference for reference, candidate in zip(baseline, policy)]
    rng = random.Random(seed)
    boot = sorted(
        median([deltas[rng.randrange(len(deltas))] for _ in deltas])
        for _ in range(samples)
    )
    return {
        "paired_runs": len(deltas),
        "median_delta": median(deltas),
        "ci95_low": boot[int(0.025 * samples)],
        "ci95_high": boot[min(samples - 1, int(0.975 * samples))],
    }
