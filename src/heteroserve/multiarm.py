"""Matched regular-GPU replay of several deployed two-pool router policies."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

from .evaluation import validate_final_environment
from .paired import _sha, _write
from .statistics import paired_bootstrap_delta


METRICS = ("success_rate", "p95_ttft_s", "p95_e2e_s",
           "token_throughput_per_s", "e2e_goodput_tokens_per_s")


def _router_hash(path: Path) -> str:
    return hashlib.sha256(json.dumps(json.loads(path.read_text()),
                                     sort_keys=True).encode()).hexdigest()


async def run(args: argparse.Namespace) -> dict:
    from .replay import replay

    validate_final_environment()
    if args.repetitions < 3 or args.concurrency < 1:
        raise ValueError("at least three repetitions and positive concurrency required")
    capability = json.loads(args.capabilities.read_text())
    server = json.loads(args.server_config.read_text())
    secondary_path = getattr(args, "secondary_capabilities", None)
    deployment = ("two_independent_heterogeneous_pools" if secondary_path else
                  "two_independent_colocated_pools")
    primary_type = server.get("primary_gpu_type", "a40")
    if primary_type not in {"a40", "v100"}:
        raise ValueError("unsupported primary GPU type")
    if (server.get("model_id") != args.model_id
            or server.get("model_revision") != args.model_revision
            or server.get("tokenizer_revision") != args.model_revision
            or server.get("deployment") != deployment
            or server.get("ports") != [8000, 8001]):
        raise ValueError("server configuration does not match the measured model/topology")
    gpu_probe = capability["probes"]["nvidia_smi_query"]
    torch_probe = capability["probes"]["torch_runtime"]
    gpu_rows = [row for row in gpu_probe.get("stdout", "").splitlines() if row.strip()]
    torch_info = json.loads(torch_probe.get("stdout", "{}"))
    if (capability["evidence_label"] != "real_gpu_final"
            or capability["slurm"]["SLURM_JOB_ID"] != os.environ["SLURM_JOB_ID"]
            or capability["packages"].get("vllm") != "0.11.2"
            or gpu_probe.get("status") != "ok" or torch_probe.get("status") != "ok"):
        raise ValueError("primary regular vLLM job capability mismatch")
    secondary = None
    if secondary_path:
        secondary = json.loads(secondary_path.read_text())
        other_rows = [row for row in secondary["probes"]["nvidia_smi_query"].get("stdout", "").splitlines() if row.strip()]
        other_torch = json.loads(secondary["probes"]["torch_runtime"].get("stdout", "{}"))
        expected_primary = "A40" if primary_type == "a40" else "V100"
        if (capability["slurm"].get("SLURM_GPUS_ON_NODE") != "1"
                or len(gpu_rows) != 1 or expected_primary not in gpu_rows[0]
                or torch_info.get("device_count") != 1
                or secondary.get("evidence_label") != "real_gpu_final"
                or secondary["slurm"].get("SLURM_GPUS_ON_NODE") != "1"
                or secondary["slurm"].get("SLURM_JOB_ID") == os.environ["SLURM_JOB_ID"]
                or secondary["host"]["hostname"] == capability["host"]["hostname"]
                or secondary["packages"].get("vllm") != "0.11.2"
                or secondary["probes"]["nvidia_smi_query"].get("status") != "ok"
                or len(other_rows) != 1 or "NVIDIA A100" not in other_rows[0]
                or secondary["probes"]["torch_runtime"].get("status") != "ok"
                or other_torch.get("device_count") != 1):
            raise ValueError("heterogeneous run requires separate supported GPU and A100 regular jobs")
    elif (capability["slurm"].get("SLURM_GPUS_ON_NODE") != "2"
          or len(gpu_rows) != 2 or any("NVIDIA A40" not in row for row in gpu_rows)
          or torch_info.get("device_count") != 2):
        raise ValueError("two-A40 run requires both A40s in one job")
    arms = json.loads(args.arms_file.read_text())["arms"]
    if len(arms) < 2 or len({a["name"] for a in arms}) != len(arms):
        raise ValueError("configure at least two uniquely named arms")
    if arms[0]["name"] != "round_robin":
        raise ValueError("first arm must be the two-GPU round-robin baseline")
    metadata = json.loads(args.trace.with_suffix(".metadata.json").read_text())
    warmup_metadata = json.loads(args.warmup_trace.with_suffix(".metadata.json").read_text())
    for path, info in ((args.trace, metadata), (args.warmup_trace, warmup_metadata)):
        if (info["trace_sha256"] != _sha(path)
                or info["tokenizer_revision"] != args.model_revision
                or info["model"] != args.model_id
                or info["max_model_len"] != 2048):
            raise ValueError("trace/tokenizer mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    common = dict(model="heteroserve-model", model_revision=args.model_revision,
                  tokenizer_revision=args.model_revision, prompt_token_id=args.prompt_token_id,
                  concurrency=args.concurrency, timeout_s=args.timeout_s,
                  slo_ttft_s=args.slo_ttft_s, slo_tpot_s=args.slo_tpot_s,
                  slo_e2e_s=args.slo_e2e_s, save_output_text=True,
                  evidence_label="real_gpu_final")
    indexed_arms = []
    for arm in arms:
        path = Path(arm["router_config"])
        router = json.loads(path.read_text())
        if (router.get("mode") != arm["name"] or router.get("model") != "heteroserve-model"
                or len(router.get("pools", [])) != 2
                or {pool["name"] for pool in router["pools"]} != {"gpu0", "gpu1"}
                or any(pool.get("route_kind") != "colocated" or not pool.get("deployed")
                       for pool in router["pools"])):
            raise ValueError("every arm must route across the same two colocated GPU pools")
        indexed_arms.append({"name": arm["name"], "url": arm["url"],
                             "router_config": str(path),
                             "router_config_file_sha256": _sha(path),
                             "router_config_sha256": _router_hash(path)})
        target = args.output_dir / f"warmup_{arm['name']}.jsonl"
        warmup = argparse.Namespace(**{**common, "trace": args.warmup_trace,
                                      "base_url": arm["url"], "output": target,
                                      "run_id": f"{args.run_id}_warmup_{arm['name']}",
                                      "evidence_label": "gpu_dev_regular"})
        manifest = await replay(warmup)
        if manifest["summary"]["success_rate"] != 1:
            raise RuntimeError(f"{arm['name']} warmup failed")
    repetitions = []
    for number in range(1, args.repetitions + 1):
        order = arms[(number - 1) % len(arms):] + arms[:(number - 1) % len(arms)]
        repetition = {"number": number, "order": [a["name"] for a in order],
                      "manifests": {}}
        for arm in order:
            name = arm["name"]
            trial_id = f"{args.run_id}_rep{number:02d}_{name}"
            target = args.output_dir / f"{trial_id}.jsonl"
            trial = argparse.Namespace(**{**common, "trace": args.trace,
                                          "base_url": arm["url"],
                                          "output": target, "run_id": trial_id})
            await replay(trial)
            repetition["manifests"][name] = str(target.with_suffix(".manifest.json"))
        repetitions.append(repetition)
    index = {"schema_version": 1, "evidence_label": "real_gpu_final",
             "comparison": ("heterogeneous_cross_node_pool_routing" if secondary else
                            "two_gpu_colocated_pool_routing"), "run_id": args.run_id,
             "capabilities": str(args.capabilities),
             "capabilities_sha256": _sha(args.capabilities),
             "secondary_capabilities": str(secondary_path) if secondary_path else None,
             "secondary_capabilities_sha256": _sha(secondary_path) if secondary_path else None,
             "server_config": str(args.server_config),
             "server_config_sha256": _sha(args.server_config),
            "trace_sha256": _sha(args.trace), "arms": indexed_arms,
             "repetitions": repetitions}
    _write(args.output_dir / "multiarm_index.json", index)
    return index


def analyze(index_path: Path) -> dict:
    index = json.loads(index_path.read_text())
    if index["evidence_label"] != "real_gpu_final" or len(index["repetitions"]) < 3:
        raise ValueError("at least three real-GPU repetitions required")
    if _sha(Path(index["capabilities"])) != index["capabilities_sha256"]:
        raise ValueError("capability hash mismatch")
    if _sha(Path(index["server_config"])) != index["server_config_sha256"]:
        raise ValueError("server configuration hash mismatch")
    capability = json.loads(Path(index["capabilities"]).read_text())
    gpu_rows = [row for row in capability["probes"]["nvidia_smi_query"]["stdout"].splitlines()
                if row.strip()]
    secondary_path = index.get("secondary_capabilities")
    if secondary_path:
        server = json.loads(Path(index["server_config"]).read_text())
        primary_type = server.get("primary_gpu_type")
        expected_primary = {"a40": "A40", "v100": "V100"}.get(primary_type)
        if _sha(Path(secondary_path)) != index["secondary_capabilities_sha256"]:
            raise ValueError("secondary capability hash mismatch")
        secondary = json.loads(Path(secondary_path).read_text())
        other_rows = [row for row in secondary["probes"]["nvidia_smi_query"]["stdout"].splitlines() if row.strip()]
        if (index["comparison"] != "heterogeneous_cross_node_pool_routing"
                or capability["slurm"].get("SLURM_GPUS_ON_NODE") != "1"
                or secondary["slurm"].get("SLURM_GPUS_ON_NODE") != "1"
                or capability["host"]["hostname"] == secondary["host"]["hostname"]
                or len(gpu_rows) != 1 or not expected_primary or expected_primary not in gpu_rows[0]
                or len(other_rows) != 1 or "NVIDIA A100" not in other_rows[0]):
            raise ValueError("analysis requires separate actual A40 and A100 jobs")
        gpu_rows += other_rows
    elif (capability.get("evidence_label") != "real_gpu_final"
          or capability["slurm"].get("SLURM_GPUS_ON_NODE") != "2"
          or len(gpu_rows) != 2
          or any("NVIDIA A40" not in row for row in gpu_rows)):
        raise ValueError("analysis requires an actual two-A40 capability snapshot")
    job = capability["slurm"]["SLURM_JOB_ID"]
    arms = {arm["name"]: arm for arm in index["arms"]}
    if "round_robin" not in arms:
        raise ValueError("round-robin baseline missing")
    for arm in arms.values():
        path = Path(arm["router_config"])
        if (_sha(path) != arm["router_config_file_sha256"]
                or _router_hash(path) != arm["router_config_sha256"]):
            raise ValueError("router configuration hash mismatch")
    values = {name: {metric: [] for metric in METRICS} for name in arms}
    pool_counts = {name: {} for name in arms}
    text_mismatch = token_mismatch = failed_pairs = 0
    source = config = None
    for repetition in index["repetitions"]:
        observations = {}
        for name, arm in arms.items():
            manifest = json.loads(Path(repetition["manifests"][name]).read_text())
            raw = Path(manifest["raw_results"])
            if (manifest["evidence_label"] != "real_gpu_final"
                    or manifest["environment"]["slurm_job_id"] != job
                    or manifest["environment"]["vllm"] != "0.11.2"
                    or manifest["trace_sha256"] != index["trace_sha256"]
                    or _sha(Path(manifest["trace"])) != index["trace_sha256"]
                    or _sha(raw) != manifest["raw_results_sha256"]):
                raise ValueError("trial provenance or raw hash mismatch")
            trial_config = dict(manifest["config"])
            if trial_config.pop("base_url") != arm["url"]:
                raise ValueError("wrong arm endpoint")
            if config is None:
                config = trial_config
                source = manifest["environment"]["source_sha256"]
            if trial_config != config or manifest["environment"]["source_sha256"] != source:
                raise ValueError("configuration or source drift")
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            by_id = {row["request_id"]: row for row in rows}
            if len(rows) != len(by_id) or len(rows) != manifest["summary"]["requests"]:
                raise ValueError("request count/ID mismatch")
            if sum(r["outcome"] == "success" for r in rows) != manifest["summary"]["successes"]:
                raise ValueError("success count mismatch")
            for row in rows:
                output = row["output_text"]
                expected = hashlib.sha256(output.encode()).hexdigest() if output else None
                if (row["output_sha256"] != expected or row["output_chars"] != len(output)
                        or row.get("router_config_sha256") != arm["router_config_sha256"]):
                    raise ValueError("output or router hash mismatch")
                pool = row["pool"]
                pool_counts[name][pool] = pool_counts[name].get(pool, 0) + 1
            observations[name] = by_id
            for metric in METRICS:
                value = manifest["summary"].get(metric)
                if value is None or not math.isfinite(value):
                    raise ValueError(f"missing metric {metric}")
                values[name][metric].append(value)
        baseline = observations["round_robin"]
        for name, rows in observations.items():
            if rows.keys() != baseline.keys():
                raise ValueError("matched trace request IDs differ")
            if name == "round_robin":
                continue
            for request_id, row in rows.items():
                ref = baseline[request_id]
                text_mismatch += row["output_sha256"] != ref["output_sha256"]
                token_mismatch += row["output_tokens"] != ref["output_tokens"]
                failed_pairs += row["outcome"] != "success" or ref["outcome"] != "success"
    metrics = {}
    for name in arms:
        metrics[name] = {metric: {"trial_values": values[name][metric],
                                  "median": median(values[name][metric])}
                         for metric in METRICS}
        if name != "round_robin":
            for metric in METRICS:
                metrics[name][metric]["paired_delta_vs_round_robin"] = paired_bootstrap_delta(
                    values["round_robin"][metric], values[name][metric])
    return {"evidence_label": "real_gpu_final", "run_id": index["run_id"],
            "comparison": index["comparison"], "slurm_job_id": job,
            "gpu": "\n".join(gpu_rows),
            "repetitions": len(index["repetitions"]), "metrics": metrics,
            "pool_request_counts": pool_counts,
            "text_mismatched_requests_vs_round_robin": text_mismatch,
            "token_count_mismatched_requests_vs_round_robin": token_mismatch,
            "non_success_pairs": failed_pairs,
            "latency_comparison_eligible": token_mismatch == 0 and failed_pairs == 0,
            "metric_delta_convention": "policy minus round_robin",
            "limitations": ["Two independent colocated vLLM pools; no KV transfer or prefill/decode split.",
                            "Cross-node routing includes network effects." if secondary_path else
                            "Both GPUs must be identified from the same actual Slurm allocation.",
                            "Client text chunks do not establish exact token-level TPOT."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    for name in ("trace", "warmup-trace", "capabilities", "server-config", "arms-file", "output-dir"):
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    run_parser.add_argument("--secondary-capabilities", type=Path)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--model-id", required=True)
    run_parser.add_argument("--model-revision", required=True)
    run_parser.add_argument("--prompt-token-id", type=int, required=True)
    run_parser.add_argument("--concurrency", type=int, default=8)
    run_parser.add_argument("--repetitions", type=int, default=3)
    run_parser.add_argument("--timeout-s", type=float, default=120)
    run_parser.add_argument("--slo-ttft-s", type=float, required=True)
    run_parser.add_argument("--slo-tpot-s", type=float, required=True)
    run_parser.add_argument("--slo-e2e-s", type=float, required=True)
    analysis_parser = sub.add_parser("analyze")
    analysis_parser.add_argument("--index", type=Path, required=True)
    analysis_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args)) if args.command == "run" else analyze(args.index)
    if args.command == "analyze":
        _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
