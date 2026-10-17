"""Summarize isolated vLLM/NIXL Prometheus counter and histogram deltas."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METRICS = {
    "prefill_s": ("both", "vllm:request_prefill_time_seconds"),
    "decode_s": ("both", "vllm:request_decode_time_seconds"),
    "server_ttft_s": ("both", "vllm:time_to_first_token_seconds"),
    "server_inter_token_s": ("both", "vllm:inter_token_latency_seconds"),
    "kv_transfer_s": ("both", "vllm:nixl_xfer_time_seconds"),
    "kv_post_s": ("both", "vllm:nixl_post_time_seconds"),
    "kv_transfer_bytes": ("both", "vllm:nixl_bytes_transferred"),
}


def read_samples(path: Path) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if math.isfinite(value) and (name.endswith("_sum") or name.endswith("_count")
                                     or name.startswith("vllm:nixl_num_failed_")):
            samples[name] = samples.get(name, 0.0) + value
    return samples


def summarize(before: Path, after: Path, role: str) -> dict[str, object]:
    old, new = read_samples(before), read_samples(after)
    result: dict[str, object] = {}
    for label, (metric_role, base) in METRICS.items():
        if metric_role not in {role, "both"}:
            continue
        count_name, sum_name = base + "_count", base + "_sum"
        if count_name not in new or sum_name not in new:
            result[label] = None
            continue
        count = new[count_name] - old.get(count_name, 0.0)
        total = new[sum_name] - old.get(sum_name, 0.0)
        if count < 0 or total < 0:
            raise ValueError(f"metric counter reset: {base}")
        result[label] = {"count": int(count), "sum": total,
                         "mean": total / count if count else None}
    for name in ("vllm:nixl_num_failed_transfers_total",
                 "vllm:nixl_num_failed_notifications_total"):
        if name in new:
            delta = new[name] - old.get(name, 0.0)
            if delta < 0:
                raise ValueError(f"metric counter reset: {name}")
            result[name] = int(delta)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefill-before", "prefill-after", "decode-before", "decode-after"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "prefill": summarize(args.prefill_before, args.prefill_after, "prefill"),
        "decode": summarize(args.decode_before, args.decode_after, "decode"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
