"""Verify raw repeated trials and report observed baseline variability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import median


METRICS = ("success_rate", "p95_ttft_s", "p95_e2e_s",
           "token_throughput_per_s", "e2e_goodput_tokens_per_s")


def bootstrap_ci(values: list[float], *, seed: int = 17, samples: int = 10000) -> tuple[float, float]:
    if len(values) < 3:
        raise ValueError("three or more trials are required")
    rng = random.Random(seed)
    draws = sorted(median(rng.choices(values, k=len(values))) for _ in range(samples))
    return draws[int(samples * .025)], draws[min(samples - 1, int(samples * .975))]


def analyze(index_path: Path) -> dict:
    index = json.loads(index_path.read_text())
    if index["evidence_label"] != "real_gpu_final" or len(index["trials"]) < 3:
        raise ValueError("analysis requires at least three real_gpu_final trials")
    capability_path = Path(index["capabilities"])
    if hashlib.sha256(capability_path.read_bytes()).hexdigest() != index["capabilities_sha256"]:
        raise ValueError("capability snapshot hash mismatch")
    capabilities = json.loads(capability_path.read_text())
    server_config_path = Path(index["server_config"])
    if hashlib.sha256(server_config_path.read_bytes()).hexdigest() != index["server_config_sha256"]:
        raise ValueError("server config hash mismatch")
    trial_manifests = [json.loads(Path(path).read_text()) for path in index["trials"]]
    reference = trial_manifests[0]
    reference_outputs = None
    output_consistent = True
    router_hashes: set[str] = set()
    pool_counts: dict[str, int] = {}
    for manifest in trial_manifests:
        if manifest["evidence_label"] != "real_gpu_final":
            raise ValueError("nonfinal trial in final index")
        if manifest["trace_sha256"] != reference["trace_sha256"] or manifest["config_sha256"] != reference["config_sha256"]:
            raise ValueError("trace or configuration drift across trials")
        if manifest["environment"]["slurm_job_id"] != capabilities["slurm"]["SLURM_JOB_ID"]:
            raise ValueError("trial and GPU snapshot use different Slurm jobs")
        if manifest["environment"]["vllm"] != "0.11.2":
            raise ValueError("unexpected vLLM version")
        trace_path = Path(manifest["trace"])
        if hashlib.sha256(trace_path.read_bytes()).hexdigest() != manifest["trace_sha256"]:
            raise ValueError("measured trace hash mismatch")
        canonical_config = json.dumps(manifest["config"], sort_keys=True).encode()
        if hashlib.sha256(canonical_config).hexdigest() != manifest["config_sha256"]:
            raise ValueError("trial config hash mismatch")
        if manifest["environment"].get("source_sha256") != reference["environment"].get("source_sha256"):
            raise ValueError("source drift across trials")
        raw = Path(manifest["raw_results"])
        if hashlib.sha256(raw.read_bytes()).hexdigest() != manifest["raw_results_sha256"]:
            raise ValueError("raw results hash mismatch")
        rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
        if len(rows) != manifest["summary"]["requests"] or any(row["run_id"] != manifest["run_id"] for row in rows):
            raise ValueError("raw results do not match trial manifest")
        if sum(row["outcome"] == "success" for row in rows) != manifest["summary"]["successes"]:
            raise ValueError("success counts do not match raw results")
        for row in rows:
            if row.get("router_config_sha256"):
                router_hashes.add(row["router_config_sha256"])
            pool_counts[row["pool"]] = pool_counts.get(row["pool"], 0) + 1
            if "output_text" in row:
                output = row["output_text"]
                if len(output) != row["output_chars"] or (
                    hashlib.sha256(output.encode()).hexdigest() if output else None
                ) != row["output_sha256"]:
                    raise ValueError("stored output text does not match its checksum")
        outputs = {row["request_id"]: (row["output_tokens"], row["output_sha256"])
                   for row in rows if row["outcome"] == "success"}
        if reference_outputs is None:
            reference_outputs = outputs
        elif outputs != reference_outputs:
            output_consistent = False
    if len(router_hashes) > 1:
        raise ValueError("router configuration drift across requests or trials")
    metrics = {}
    for name in METRICS:
        values = [manifest["summary"][name] for manifest in trial_manifests]
        if any(value is None or not math.isfinite(value) for value in values):
            metrics[name] = None
        else:
            low, high = bootstrap_ci(values)
            metrics[name] = {"trial_values": values, "median": median(values),
                             "bootstrap_median_ci95": [low, high]}
    return {"evidence_label": "real_gpu_final", "run_id": index["run_id"],
            "gpu": capabilities["probes"]["nvidia_smi_query"]["stdout"].strip(),
            "trials": len(trial_manifests), "trace_sha256": reference["trace_sha256"],
            "config_sha256": reference["config_sha256"], "metrics": metrics,
            "router_config_sha256": next(iter(router_hashes), None), "pool_request_counts": pool_counts,
            "output_consistent_across_trials": output_consistent,
            "limitations": ["Client stream chunks do not establish exact token-level TPOT.",
                            "No paired policy or split comparison is implied by this baseline."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
