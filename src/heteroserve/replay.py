"""Replay exact-length token-ID traces against vLLM's streaming Completions API."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx

from .metrics import SLO, summarize
from .schema import RequestResult, RequestSpec


def read_trace(path: Path) -> list[RequestSpec]:
    rows = [RequestSpec(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row.request_id for row in rows}) != len(rows):
        raise ValueError("trace must contain requests with unique IDs")
    if any(a.arrival_s > b.arrival_s for a, b in zip(rows, rows[1:])):
        raise ValueError("trace arrivals must be ordered")
    return rows


def git_sha() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def source_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parents[2]
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths.extend(root / name for name in (
        "pyproject.toml", "requirements-gpu.txt", "requirements-client.txt",
        "scripts/serve_vllm.sh", "scripts/slurm/final_colocated.sbatch",
        "scripts/slurm/final_paired_proxy.sbatch",
        "scripts/slurm/final_two_pool.sbatch",
        "scripts/slurm/hetero_a100_server.sh",
        "scripts/slurm/final_hetero_a40_client.sh",
        "scripts/slurm/final_hetero_v100_client.sh",
        "scripts/slurm/split_pd.sh",
    ))
    for path in paths:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


async def send_one(
    client: httpx.AsyncClient, request: RequestSpec, *, base_url: str, model: str,
    prompt_token_id: int, semaphore: asyncio.Semaphore, origin: float,
    timeout_s: float, save_output_text: bool = False,
) -> tuple[RequestResult, dict]:
    scheduled = origin + request.arrival_s
    await asyncio.sleep(max(0, scheduled - time.perf_counter()))
    arrived = time.perf_counter()
    submitted = arrived
    server_submission = None
    first_token = None
    last_token = None
    output_tokens = 0
    chunk_count = 0
    output_parts: list[str] = []
    finish_reason = None
    served_pool = "colocated"
    router_admission_wait_s = None
    router_config_sha256 = None
    stream_done = False
    error = None
    outcome = "success"
    admitted = None
    completed = submitted
    try:
        async with asyncio.timeout_at(arrived + timeout_s):
            async with semaphore:
                admitted = time.perf_counter()
                payload = {
                    "model": model,
                    "prompt": request.prompt_token_ids if request.prompt_token_ids is not None
                    else [prompt_token_id] * request.prompt_tokens,
                    "max_tokens": request.max_output_tokens,
                    "temperature": 0,
                    "seed": request.seed,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                server_submission = time.perf_counter()
                async with client.stream("POST", f"{base_url}/v1/completions", json=payload) as response:
                    response.raise_for_status()
                    served_pool = response.headers.get("x-heteroserve-pool", "colocated")
                    router_config_sha256 = response.headers.get("x-heteroserve-config-sha256")
                    if "x-heteroserve-admission-wait-s" in response.headers:
                        router_admission_wait_s = float(response.headers["x-heteroserve-admission-wait-s"])
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            stream_done = True
                            break
                        event = json.loads(data)
                        usage = event.get("usage")
                        if usage and usage.get("prompt_tokens") is not None:
                            if int(usage["prompt_tokens"]) != request.prompt_tokens:
                                raise ValueError("server prompt token count differs from trace")
                        if usage and usage.get("completion_tokens") is not None:
                            output_tokens = int(usage["completion_tokens"])
                            if output_tokens > request.max_output_tokens:
                                raise ValueError("server exceeded requested completion token limit")
                        for choice in event.get("choices", []):
                            if choice.get("finish_reason") is not None:
                                finish_reason = choice["finish_reason"]
                            if choice.get("text"):
                                output_parts.append(choice["text"])
                                now = time.perf_counter()
                                first_token = first_token or now
                                last_token = now
                                chunk_count += 1
            if not stream_done:
                raise ValueError("stream ended without [DONE]")
            if output_tokens == 0:
                raise ValueError("stream ended without completion token usage")
            if first_token is None:
                raise ValueError("stream reported output tokens without text chunks")
    except (TimeoutError, httpx.TimeoutException) as exc:
        outcome = "timed_out"
        error = f"{type(exc).__name__}: {exc}"
    except httpx.HTTPStatusError as exc:
        outcome = "rejected" if exc.response.status_code in {429, 503} else "failed"
        error = f"{type(exc).__name__}: {exc}"
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        outcome = "failed"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        completed = time.perf_counter()
    wait = (admitted - submitted) if admitted is not None else completed - submitted
    result = RequestResult(
        request.request_id, submitted - origin,
        None if first_token is None else first_token - origin,
        completed - origin, output_tokens, error is None,
        wait, served_pool, error,
        None if last_token is None else last_token - origin,
        False,
        request.arrival_s, arrived - origin,
        None if server_submission is None else server_submission - origin,
        outcome,
    )
    output_text = "".join(output_parts)
    record = {**asdict(result),
              "prompt_tokens": request.prompt_tokens, "max_output_tokens": request.max_output_tokens,
              "seed": request.seed, "ttft_s": result.ttft_s, "tpot_s": result.tpot_s,
              "approximate_tpot_s": result.approximate_tpot_s,
              "tpot_method": "text_chunk_endpoints_approximate" if result.approximate_tpot_s is not None else None,
              "scheduling_delay_s": result.scheduling_delay_s,
              "e2e_s": result.e2e_s, "stream_text_chunks": chunk_count,
              "output_sha256": hashlib.sha256(output_text.encode()).hexdigest() if output_text else None,
              "output_chars": len(output_text), "finish_reason": finish_reason,
              "router_admission_wait_s": router_admission_wait_s,
              "router_config_sha256": router_config_sha256,
              "prompt_mode": "trace_token_ids" if request.prompt_token_ids is not None else "repeated_token"}
    if save_output_text:
        record["output_text"] = output_text
    return result, record


async def replay(args: argparse.Namespace, client: httpx.AsyncClient | None = None) -> dict:
    trace = read_trace(args.trace)
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=args.concurrency))
    assert client is not None
    try:
        model_check_url = getattr(args, "model_check_url", None) or args.base_url
        model_response = await client.get(f"{model_check_url}/v1/models", timeout=args.timeout_s)
        model_response.raise_for_status()
        names = {item["id"] for item in model_response.json()["data"]}
        if args.model not in names:
            raise ValueError(f"model {args.model!r} not served; available: {sorted(names)}")
        origin = time.perf_counter()
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [asyncio.create_task(send_one(
            client, request, base_url=args.base_url, model=args.model,
            prompt_token_id=args.prompt_token_id, semaphore=semaphore,
            origin=origin, timeout_s=args.timeout_s,
            save_output_text=getattr(args, "save_output_text", False),
        )) for request in trace]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results = []
        with args.output.open("w") as handle:
            for task in asyncio.as_completed(tasks):
                result, record = await task
                results.append(result)
                record["run_id"] = args.run_id
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
        summary = summarize(results, SLO(args.slo_ttft_s, args.slo_tpot_s,
                                         getattr(args, "slo_e2e_s", None)))
        trace_hash = hashlib.sha256(args.trace.read_bytes()).hexdigest()
        config = {"base_url": args.base_url, "model_check_url": model_check_url,
                  "model": args.model,
                  "model_revision": args.model_revision,
                  "tokenizer_revision": args.tokenizer_revision,
                  "prompt_token_id": args.prompt_token_id,
                  "concurrency": args.concurrency, "timeout_s": args.timeout_s,
                  "slo_ttft_s": args.slo_ttft_s, "slo_tpot_s": args.slo_tpot_s,
                  "slo_e2e_s": getattr(args, "slo_e2e_s", None),
                  "save_output_text": getattr(args, "save_output_text", False)}
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        manifest = {
            "evidence_label": args.evidence_label, "run_id": args.run_id,
            "trace": str(args.trace), "trace_sha256": trace_hash,
            "raw_results": str(args.output), "summary": summary,
            "raw_results_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "config": config, "config_sha256": config_hash,
            "prompt_mode": "trace_token_ids" if all(r.prompt_token_ids is not None for r in trace)
            else "repeated_or_mixed_token_ids",
            "output_check": "SHA-256 of concatenated streamed text; optional text stored in raw JSONL",
            "timing_definition": {
                "scheduled_s": "trace offset from replay origin",
                "actual_arrival_s": "client wake time from replay origin",
                "server_submission_s": "client HTTP submission time from replay origin",
                "submitted_s": "legacy alias for actual_arrival_s",
                "ttft_s": "first nonempty streamed text minus actual client arrival; includes admission wait",
                "e2e_s": "completion minus actual client arrival; includes admission wait",
                "tpot_s": "unavailable without validated per-token timestamps",
                "approximate_tpot_s": "last minus first nonempty text chunk divided by completion_tokens minus one; not exact TPOT",
            },
            "environment": {"python": platform.python_version(), "git_sha": git_sha(),
                            "source_sha256": source_sha256(),
                            "vllm": installed_version("vllm"),
                            "httpx": importlib.metadata.version("httpx"),
                            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                            "slurm_job_partition": os.getenv("SLURM_JOB_PARTITION"),
                            "slurm_job_account": os.getenv("SLURM_JOB_ACCOUNT")},
        }
        manifest_path = args.output.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest
    finally:
        if owns_client:
            await client.aclose()


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-check-url", help="vLLM endpoint for model validation when a proxy lacks /v1/models")
    parser.add_argument("--model", default="heteroserve-model")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--prompt-token-id", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--slo-ttft-s", type=float, required=True)
    parser.add_argument("--slo-tpot-s", type=float, required=True)
    parser.add_argument("--slo-e2e-s", type=float)
    parser.add_argument("--save-output-text", action="store_true")
    parser.add_argument(
        "--evidence-label",
        choices=["gpu_debug", "gpu_dev_regular", "real_gpu_final"],
        required=True,
    )
    args = parser.parse_args()
    if args.prompt_token_id < 0 or args.concurrency < 1 or args.timeout_s <= 0:
        parser.error("token ID must be nonnegative; concurrency and timeout must be positive")
    print(json.dumps(asyncio.run(replay(args))["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
