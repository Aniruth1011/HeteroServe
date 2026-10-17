"""Run and audit paired direct-vLLM versus one-pool proxy trials on one GPU."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path

from .evaluation import validate_final_environment
from .statistics import paired_bootstrap_delta


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


async def run(args: argparse.Namespace) -> dict:
    from .replay import replay

    validate_final_environment()
    if args.pairs < 3 or args.concurrency < 1 or args.timeout_s <= 0:
        raise ValueError("at least three pairs, positive concurrency and timeout required")
    if args.direct_url == args.proxy_url:
        raise ValueError("direct and proxy endpoints must differ")
    capability = json.loads(args.capabilities.read_text())
    server = json.loads(args.server_config.read_text())
    metadata = json.loads(args.trace.with_suffix(".metadata.json").read_text())
    if (capability["evidence_label"] != "real_gpu_final"
            or capability["slurm"]["SLURM_JOB_ID"] != os.environ["SLURM_JOB_ID"]
            or capability["packages"].get("vllm") != "0.11.2"):
        raise ValueError("capability snapshot does not match this regular vLLM job")
    if (metadata["trace_sha256"] != _sha(args.trace)
            or metadata["model"] != server["model_id"]
            or metadata["tokenizer_revision"] != server["tokenizer_revision"]
            or metadata["max_model_len"] != server["max_model_len"]
            or server["model_revision"] != args.model_revision):
        raise ValueError("trace/model/server configuration mismatch")
    if args.warmup_trace is not None:
        warmup_metadata = json.loads(args.warmup_trace.with_suffix(".metadata.json").read_text())
        if (warmup_metadata["trace_sha256"] != _sha(args.warmup_trace)
                or warmup_metadata["model"] != server["model_id"]
                or warmup_metadata["tokenizer_revision"] != server["tokenizer_revision"]):
            raise ValueError("warmup trace/model mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    common = dict(model="heteroserve-model", model_revision=args.model_revision,
                  tokenizer_revision=server["tokenizer_revision"],
                  prompt_token_id=args.prompt_token_id, concurrency=args.concurrency,
                  timeout_s=args.timeout_s, slo_ttft_s=args.slo_ttft_s,
                  slo_tpot_s=args.slo_tpot_s, slo_e2e_s=args.slo_e2e_s,
                  save_output_text=True, evidence_label="real_gpu_final")
    endpoints = {"direct": args.direct_url, "proxy": args.proxy_url}
    warmups = {}
    if args.warmup_trace:
        for name, url in endpoints.items():
            target = args.output_dir / f"warmup_{name}.jsonl"
            trial = argparse.Namespace(**{**common, "trace": args.warmup_trace,
                                          "base_url": url, "output": target,
                                          "run_id": f"{args.run_id}_warmup_{name}",
                                          "evidence_label": "gpu_dev_regular"})
            manifest = await replay(trial)
            if manifest["summary"]["success_rate"] != 1:
                raise RuntimeError(f"{name} warmup failed")
            warmups[name] = str(target.with_suffix(".manifest.json"))
    pairs = []
    for number in range(1, args.pairs + 1):
        order = ("direct", "proxy") if number % 2 else ("proxy", "direct")
        pair = {"number": number, "order": list(order)}
        for name in order:
            trial_id = f"{args.run_id}_pair{number:02d}_{name}"
            target = args.output_dir / f"{trial_id}.jsonl"
            trial = argparse.Namespace(**{**common, "trace": args.trace,
                                          "base_url": endpoints[name],
                                          "output": target, "run_id": trial_id})
            manifest = await replay(trial)
            if manifest["environment"]["vllm"] != "0.11.2":
                raise RuntimeError("unexpected vLLM version")
            pair[name] = str(target.with_suffix(".manifest.json"))
        pairs.append(pair)
    index = {"schema_version": 1, "evidence_label": "real_gpu_final",
             "comparison": "direct_vllm_vs_single_pool_proxy_overhead",
             "run_id": args.run_id, "capabilities": str(args.capabilities),
             "capabilities_sha256": _sha(args.capabilities),
             "server_config": str(args.server_config),
             "server_config_sha256": _sha(args.server_config),
             "router_config": str(args.router_config),
             "router_config_file_sha256": _sha(args.router_config),
             "router_config_sha256": hashlib.sha256(json.dumps(
                 json.loads(args.router_config.read_text()), sort_keys=True).encode()).hexdigest(),
             "trace_sha256": _sha(args.trace), "warmups": warmups,
             "endpoints": endpoints, "pairs": pairs}
    _write(args.output_dir / "paired_index.json", index)
    return index


def analyze(index_path: Path) -> dict:
    index = json.loads(index_path.read_text())
    if index["evidence_label"] != "real_gpu_final" or len(index["pairs"]) < 3:
        raise ValueError("three or more final GPU pairs required")
    for key in ("capabilities", "server_config"):
        if _sha(Path(index[key])) != index[f"{key}_sha256"]:
            raise ValueError(f"{key} changed after the run")
    if _sha(Path(index["router_config"])) != index["router_config_file_sha256"]:
        raise ValueError("router configuration changed after the run")
    capability = json.loads(Path(index["capabilities"]).read_text())
    job = capability["slurm"]["SLURM_JOB_ID"]
    required_metrics = ("success_rate", "p95_ttft_s", "p95_e2e_s",
                        "token_throughput_per_s", "e2e_goodput_tokens_per_s")
    vectors = {name: {side: [] for side in ("direct", "proxy")}
               for name in required_metrics}
    text_changed = token_changed = failed = 0
    reference_source = reference_trace = reference_config = None
    pair_details = []
    for pair in index["pairs"]:
        observations = {}
        for side in ("direct", "proxy"):
            manifest = json.loads(Path(pair[side]).read_text())
            if (manifest["evidence_label"] != "real_gpu_final"
                    or manifest["environment"]["slurm_job_id"] != job
                    or manifest["environment"]["vllm"] != "0.11.2"
                    or manifest["trace_sha256"] != index["trace_sha256"]
                    or _sha(Path(manifest["trace"])) != manifest["trace_sha256"]
                    or _sha(Path(manifest["raw_results"])) != manifest["raw_results_sha256"]):
                raise ValueError("trial provenance or raw hash mismatch")
            config = dict(manifest["config"])
            if config.pop("base_url") != index["endpoints"][side]:
                raise ValueError("trial used the wrong endpoint")
            if reference_config is None:
                reference_config = config
                reference_source = manifest["environment"]["source_sha256"]
                reference_trace = manifest["trace_sha256"]
            if (config != reference_config
                    or manifest["environment"]["source_sha256"] != reference_source
                    or manifest["trace_sha256"] != reference_trace):
                raise ValueError("paired trial configuration/source drift")
            rows = [json.loads(line) for line in Path(manifest["raw_results"]).read_text().splitlines()]
            by_id = {row["request_id"]: row for row in rows}
            if len(rows) != len(by_id) or len(rows) != manifest["summary"]["requests"]:
                raise ValueError("trial request count or IDs invalid")
            if sum(row["outcome"] == "success" for row in rows) != manifest["summary"]["successes"]:
                raise ValueError("success count mismatch")
            for row in rows:
                output = row["output_text"]
                expected = hashlib.sha256(output.encode()).hexdigest() if output else None
                if expected != row["output_sha256"] or len(output) != row["output_chars"]:
                    raise ValueError("output text/hash mismatch")
                if side == "proxy" and row.get("router_config_sha256") != index["router_config_sha256"]:
                    raise ValueError("proxy route hash mismatch")
            observations[side] = by_id
            for name in required_metrics:
                value = manifest["summary"].get(name)
                if value is None or not math.isfinite(value):
                    raise ValueError(f"missing metric {name}")
                vectors[name][side].append(value)
        direct, proxy = observations["direct"], observations["proxy"]
        if direct.keys() != proxy.keys():
            raise ValueError("paired request IDs differ")
        pair_text = pair_tokens = pair_failed = 0
        for request_id in direct:
            a, b = direct[request_id], proxy[request_id]
            pair_text += a["output_sha256"] != b["output_sha256"]
            pair_tokens += a["output_tokens"] != b["output_tokens"]
            pair_failed += a["outcome"] != "success" or b["outcome"] != "success"
        text_changed += pair_text
        token_changed += pair_tokens
        failed += pair_failed
        pair_details.append({"number": pair["number"], "order": pair["order"],
                             "text_mismatched_requests": pair_text,
                             "token_count_mismatched_requests": pair_tokens,
                             "non_success_pairs": pair_failed})
    metrics = {name: paired_bootstrap_delta(values["direct"], values["proxy"])
               for name, values in vectors.items()}
    for name, values in vectors.items():
        metrics[name]["direct_trial_values"] = values["direct"]
        metrics[name]["proxy_trial_values"] = values["proxy"]
    return {"evidence_label": "real_gpu_final", "run_id": index["run_id"],
            "comparison": index["comparison"], "slurm_job_id": job,
            "gpu": capability["probes"]["nvidia_smi_query"]["stdout"].strip(),
            "pairs": len(pair_details), "pair_details": pair_details,
            "text_mismatched_requests": text_changed,
            "token_count_mismatched_requests": token_changed,
            "non_success_pairs": failed,
            "latency_comparison_eligible": token_changed == 0 and failed == 0,
            "metric_delta_convention": "proxy minus direct; positive latency is a regression",
            "metrics": metrics,
            "limitations": ["Single A40 and one colocated vLLM backend: this measures proxy overhead only.",
                            "No disaggregated KV transfer or adaptive routing is deployed.",
                            "Client text chunks do not establish exact token-level TPOT."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    for flag in ("trace", "capabilities", "server-config", "router-config", "output-dir"):
        run_parser.add_argument(f"--{flag}", type=Path, required=True)
    run_parser.add_argument("--warmup-trace", type=Path)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--model-revision", required=True)
    run_parser.add_argument("--prompt-token-id", type=int, required=True)
    run_parser.add_argument("--concurrency", type=int, default=8)
    run_parser.add_argument("--pairs", type=int, default=5)
    run_parser.add_argument("--timeout-s", type=float, default=120)
    run_parser.add_argument("--slo-ttft-s", type=float, required=True)
    run_parser.add_argument("--slo-tpot-s", type=float, required=True)
    run_parser.add_argument("--slo-e2e-s", type=float, required=True)
    run_parser.add_argument("--direct-url", default="http://127.0.0.1:8000")
    run_parser.add_argument("--proxy-url", default="http://127.0.0.1:8100")
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--index", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = asyncio.run(run(args))
    else:
        result = analyze(args.index)
        _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
