"""Run repeated, matched vLLM replay trials inside a regular Slurm GPU job."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

def validate_final_environment() -> None:
    if not os.getenv("SLURM_JOB_ID") or "debug" in os.getenv("SLURM_JOB_PARTITION", "").lower():
        raise RuntimeError("final evaluation requires a regular Slurm allocation")
    if "debug" in os.getenv("SLURM_JOB_QOS", "").lower():
        raise RuntimeError("final evaluation cannot use debug QoS")


async def evaluate(args: argparse.Namespace) -> dict:
    from .replay import replay
    if args.trials < 3:
        raise ValueError("at least three measured trials are required")
    validate_final_environment()
    if not args.capabilities.is_file():
        raise FileNotFoundError(args.capabilities)
    capabilities = json.loads(args.capabilities.read_text())
    if capabilities["evidence_label"] != "real_gpu_final":
        raise ValueError("capability snapshot is not labeled real_gpu_final")
    if capabilities["slurm"]["SLURM_JOB_ID"] != os.environ["SLURM_JOB_ID"]:
        raise ValueError("capability snapshot is from a different Slurm job")
    if capabilities["probes"]["nvidia_smi_query"]["status"] != "ok":
        raise ValueError("GPU capability probe did not succeed")
    if capabilities["packages"].get("vllm") != "0.11.2":
        raise ValueError("capability probe did not observe vLLM 0.11.2")
    metadata_path = args.trace.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise ValueError("final trace requires tokenizer metadata from heteroserve.text_trace")
    metadata = json.loads(metadata_path.read_text())
    if metadata["tokenizer_revision"] != args.tokenizer_revision or metadata["model"] != args.model_id:
        raise ValueError("trace tokenizer does not match served model")
    if metadata["trace_sha256"] != hashlib.sha256(args.trace.read_bytes()).hexdigest():
        raise ValueError("trace differs from tokenization metadata")
    if metadata["max_model_len"] != 2048:
        raise ValueError("trace context limit differs from server configuration")
    if not args.server_config.is_file():
        raise FileNotFoundError(args.server_config)
    server_config = json.loads(args.server_config.read_text())
    if server_config["model_id"] != args.model_id or server_config["model_revision"] != args.model_revision:
        raise ValueError("server config differs from measured model")
    if server_config["tokenizer_revision"] != args.tokenizer_revision:
        raise ValueError("server config differs from trace tokenizer")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    common = dict(trace=args.trace, base_url=args.base_url, model=args.model,
                  model_revision=args.model_revision, tokenizer_revision=args.tokenizer_revision,
                  prompt_token_id=args.prompt_token_id, concurrency=args.concurrency,
                  timeout_s=args.timeout_s, slo_ttft_s=args.slo_ttft_s,
                  slo_tpot_s=args.slo_tpot_s, slo_e2e_s=args.slo_e2e_s,
                  save_output_text=True,
                  evidence_label="real_gpu_final")
    runs = []
    if args.warmup_trace:
        warmup = argparse.Namespace(**{**common, "trace": args.warmup_trace,
                                      "output": args.output_dir / "warmup.jsonl",
                                      "run_id": f"{args.run_id}_warmup",
                                      "evidence_label": "gpu_dev_regular"})
        warmup_manifest = await replay(warmup)
        if warmup_manifest["summary"]["success_rate"] != 1:
            raise RuntimeError("warmup failed; measured trials were not started")
    for trial in range(args.trials):
        run_id = f"{args.run_id}_trial{trial + 1:02d}"
        trial_args = argparse.Namespace(**{**common, "run_id": run_id,
                                           "output": args.output_dir / f"{run_id}.jsonl"})
        manifest = await replay(trial_args)
        if manifest["environment"]["vllm"] != "0.11.2":
            raise RuntimeError("measured run used a different vLLM version")
        runs.append(str(trial_args.output.with_suffix(".manifest.json")))
    index = {"schema_version": 1, "evidence_label": "real_gpu_final",
             "run_id": args.run_id, "capabilities": str(args.capabilities),
             "capabilities_sha256": hashlib.sha256(args.capabilities.read_bytes()).hexdigest(),
             "server_config": str(args.server_config),
             "server_config_sha256": hashlib.sha256(args.server_config.read_bytes()).hexdigest(),
             "warmup": str(args.output_dir / "warmup.manifest.json") if args.warmup_trace else None,
             "trials": runs, "trial_count": len(runs)}
    (args.output_dir / "evaluation.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--warmup-trace", type=Path)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="heteroserve-model")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--prompt-token-id", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--slo-ttft-s", type=float, required=True)
    parser.add_argument("--slo-tpot-s", type=float, required=True)
    parser.add_argument("--slo-e2e-s", type=float, required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.timeout_s <= 0:
        parser.error("concurrency and timeout must be positive")
    print(json.dumps(asyncio.run(evaluate(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
