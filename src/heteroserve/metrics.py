from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from .schema import RequestResult


@dataclass(frozen=True)
class SLO:
    max_ttft_s: float
    max_tpot_s: float
    max_e2e_s: float | None = None

    def __post_init__(self) -> None:
        if self.max_ttft_s <= 0 or self.max_tpot_s <= 0 or (
            self.max_e2e_s is not None and self.max_e2e_s <= 0
        ):
            raise ValueError("SLO bounds must be positive")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarize(results: list[RequestResult], slo: SLO) -> dict[str, float | int | None]:
    successes = [r for r in results if r.success]
    ttft = [value for r in successes if (value := r.ttft_s) is not None]
    tpot = [value for r in successes if (value := r.tpot_s) is not None]
    e2e = [r.e2e_s for r in successes]
    admission = [r.admission_wait_s for r in results]
    scheduling = [value for r in results if (value := r.scheduling_delay_s) is not None]
    if results:
        wall_s = max(r.completed_s for r in results) - min(r.submitted_s for r in results)
    else:
        wall_s = 0.0
    good = [
        r for r in successes
        if r.ttft_s is not None and r.ttft_s <= slo.max_ttft_s
        and r.tpot_s is not None and r.tpot_s <= slo.max_tpot_s
    ]
    output_tokens = sum(r.output_tokens for r in successes)
    good_tokens = sum(r.output_tokens for r in good)
    e2e_good = [r for r in successes if slo.max_e2e_s is not None
                and r.e2e_s <= slo.max_e2e_s]
    return {
        "requests": len(results), "successes": len(successes),
        "rejected": sum(r.outcome == "rejected" for r in results),
        "failed": sum(r.outcome == "failed" for r in results),
        "timed_out": sum(r.outcome == "timed_out" for r in results),
        "success_rate": len(successes) / len(results) if results else None,
        "tpot_eligible": len(tpot), "tpot_excluded": len(successes) - len(tpot),
        "p50_ttft_s": median(ttft) if ttft else None,
        "p95_ttft_s": percentile(ttft, 0.95), "p99_ttft_s": percentile(ttft, 0.99),
        "p50_tpot_s": median(tpot) if tpot else None,
        "p95_tpot_s": percentile(tpot, 0.95), "p99_tpot_s": percentile(tpot, 0.99),
        "p95_e2e_s": percentile(e2e, 0.95),
        "p95_admission_wait_s": percentile(admission, 0.95),
        "p95_scheduling_delay_s": percentile(scheduling, 0.95),
        "output_tokens": output_tokens,
        "token_throughput_per_s": output_tokens / wall_s if wall_s > 0 else None,
        "slo_requests": len(good) if tpot else None,
        "slo_fraction": len(good) / len(results) if results and tpot else None,
        "sla_goodput_tokens_per_s": good_tokens / wall_s if wall_s > 0 and tpot else None,
        "e2e_slo_requests": len(e2e_good) if slo.max_e2e_s is not None else None,
        "e2e_slo_fraction": len(e2e_good) / len(results) if results and slo.max_e2e_s is not None else None,
        "e2e_goodput_tokens_per_s": sum(r.output_tokens for r in e2e_good) / wall_s
        if wall_s > 0 and slo.max_e2e_s is not None else None,
    }
