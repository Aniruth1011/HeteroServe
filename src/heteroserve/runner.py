"""Trace replay and external server-pool routing."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .client import Endpoint, stream_completion
from .metrics import SLO, summarize
from .policy import AdaptivePolicy, CostModel, PoolState, ThresholdPolicy
from .provenance import build_manifest, write_manifest
from .schema import RequestResult, WorkloadRequest
from .workload import read_workload


@dataclass
class ResultWriter:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def append(self, result: RequestResult) -> None:
        line = json.dumps(result.to_dict(), sort_keys=True) + "\n"
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()


def _pool_states(config: dict[str, Any]) -> list[PoolState]:
    return [PoolState(**pool["state"]) for pool in config["pools"]]


def _select_policy(config: dict[str, Any]):
    name = config["policy"]["name"]
    if name == "colocated":
        target = config["policy"]["pool"]
        return lambda _request, _pools: target
    if name == "threshold":
        policy = ThresholdPolicy(
            config["policy"]["prompt_tokens"],
            config["policy"]["short_pool"],
            config["policy"]["long_pool"],
        )
        return policy.choose
    if name == "adaptive":
        policy = AdaptivePolicy(CostModel(), config["policy"].get("hysteresis_fraction", 0.05))
        return policy.choose
    raise ValueError(f"unknown policy: {name}")


async def replay(config: dict[str, Any], requests: list[WorkloadRequest], writer: ResultWriter) -> list[RequestResult]:
    endpoints = {
        pool["state"]["name"]: Endpoint(
            pool["state"]["name"], pool["base_url"], pool["model"], pool.get("timeout_s", 120)
        )
        for pool in config["pools"]
    }
    states = _pool_states(config)
    choose = _select_policy(config)
    semaphore = asyncio.Semaphore(int(config["max_concurrency"]))
    max_queue = int(config.get("max_queue", config["max_concurrency"]))
    admission_timeout_s = float(config.get("admission_timeout_s", 60))
    queue_lock = asyncio.Lock()
    waiting = 0
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for event in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(event, stop.set)
        except NotImplementedError:
            pass
    origin = time.perf_counter()

    async def execute(request: WorkloadRequest) -> RequestResult | None:
        nonlocal waiting
        delay = origin + request.arrival_s - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        if stop.is_set():
            return None
        submitted = time.perf_counter()
        queued = False
        async with queue_lock:
            if semaphore.locked():
                if waiting >= max_queue:
                    completed = time.perf_counter()
                    result = RequestResult(
                        request.request_id, submitted, None, completed, 0, False,
                        completed - submitted, "unassigned", "admission queue full",
                    )
                    await writer.append(result)
                    return result
                waiting += 1
                queued = True
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=admission_timeout_s)
        except TimeoutError:
            if queued:
                async with queue_lock:
                    waiting -= 1
            completed = time.perf_counter()
            result = RequestResult(
                request.request_id, submitted, None, completed, 0, False,
                completed - submitted, "unassigned", "admission timeout",
            )
            await writer.append(result)
            return result
        if queued:
            async with queue_lock:
                waiting -= 1
        try:
            admitted = time.perf_counter()
            decision_start = time.perf_counter()
            pool_name = choose(request, states)
            controller_overhead = time.perf_counter() - decision_start
            result = await asyncio.to_thread(
                stream_completion,
                endpoints[pool_name],
                request,
                submitted,
                admitted - submitted,
                controller_overhead,
            )
            if not result.success and config.get("fallback_on_error", True):
                fallback = next(
                    (state.name for state in states if state.available and state.name != pool_name),
                    None,
                )
                if fallback is not None:
                    first_error = result.error or f"HTTP {result.http_status}"
                    result = await asyncio.to_thread(
                        stream_completion,
                        endpoints[fallback],
                        request,
                        submitted,
                        admitted - submitted,
                        controller_overhead,
                    )
                    result = replace(
                        result,
                        attempts=2,
                        fallback_from=pool_name,
                        attempt_errors=(first_error,) + ((result.error,) if result.error else ()),
                    )
            await writer.append(result)
            return result
        finally:
            semaphore.release()

    completed = await asyncio.gather(*(execute(request) for request in requests))
    return [result for result in completed if result is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.run_dir.mkdir(parents=True, exist_ok=False)
    write_manifest(
        build_manifest(args.config, args.trace, args.run_id),
        args.run_dir / "manifest.json",
    )
    results = asyncio.run(replay(config, read_workload(args.trace), ResultWriter(args.run_dir / "requests.jsonl")))
    slo = SLO(**config["slo"])
    (args.run_dir / "summary.json").write_text(
        json.dumps(summarize(results, slo), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if results and all(result.success for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
