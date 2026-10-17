"""External, server-pool Completions router for already deployed vLLM endpoints."""

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from .policy import (AdaptivePolicy, CostModel, DQAPolicy, GPUProfile,
                     KVReservations, LeastActivePolicy, PoolState,
                     RoundRobinPolicy, ThresholdPolicy)
from .schema import RequestSpec


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    pools = config.get("pools", [])
    if not pools or len({pool["name"] for pool in pools}) != len(pools):
        raise ValueError("configure at least one pool with unique names")
    if config["mode"] not in {"static", "round_robin", "least_active",
                              "threshold", "adaptive", "dqa"}:
        raise ValueError("unknown routing mode")
    if config["mode"] != "static" and len(pools) < 2:
        raise ValueError("multi-pool policy requires two deployed endpoints")
    if any(not pool.get("url", "").startswith(("http://", "https://")) for pool in pools):
        raise ValueError("each pool needs an HTTP URL")
    if any(pool.get("route_kind", "colocated") not in {"colocated", "disaggregated"} for pool in pools):
        raise ValueError("unknown pool route_kind")
    split_names = {pool["name"] for pool in pools if pool.get("route_kind") == "disaggregated"}
    if split_names:
        if not config.get("connector_evidence"):
            raise ValueError("disaggregated routing requires connector evidence path")
        evidence_path = Path(config["connector_evidence"])
        if not evidence_path.is_file():
            raise ValueError("connector evidence path is missing")
        evidence = json.loads(evidence_path.read_text())
        if not evidence.get("validated") or evidence.get("vllm_version") != "0.11.2" or not split_names.issubset(set(evidence.get("pool_names", []))):
            raise ValueError("connector evidence does not validate the configured split pools")
    if config["mode"] == "static" and config.get("static_pool") not in {p["name"] for p in pools}:
        raise ValueError("static_pool must name a configured pool")
    if config["mode"] == "threshold" and (
        config.get("short_pool") not in {p["name"] for p in pools}
        or config.get("long_pool") not in {p["name"] for p in pools}
        or config.get("short_pool") == config.get("long_pool")
        or config.get("threshold", 0) < 1
    ):
        raise ValueError("threshold mode requires valid pool names and token threshold")
    if config.get("max_inflight_per_pool", 1) < 1:
        raise ValueError("max_inflight_per_pool must be positive")
    if config.get("model") != "heteroserve-model":
        raise ValueError("router currently supports the pinned served model name")
    if config["mode"] in {"adaptive", "dqa"} and any(
        pool.get("prefill_tokens_per_s", 0) <= 0 or pool.get("decode_tokens_per_s", 0) <= 0
        for pool in pools
    ):
        raise ValueError("cost-based routing requires measured positive pool rates")
    return config


def make_states(config: dict[str, Any], health: dict[str, bool], active: dict[str, int]) -> list[PoolState]:
    names = {field.name for field in fields(PoolState)}
    limit = config.get("max_inflight_per_pool", 1)
    result = []
    for entry in config["pools"]:
        values = {"name": entry["name"], "available": True, "queue_wait_s": 0,
                  "prefill_tokens_per_s": 1, "decode_tokens_per_s": 1}
        values.update({key: value for key, value in entry.items() if key in names})
        state = PoolState(**values)
        result.append(replace(state, available=health[state.name] and active[state.name] < limit,
                              active_decode_requests=active[state.name],
                              queue_wait_s=active[state.name] * entry.get("estimated_tpot_s", 0)))
    return result


def make_policy(config: dict[str, Any]):
    mode = config["mode"]
    if mode == "static":
        return None
    if mode == "round_robin":
        return RoundRobinPolicy()
    if mode == "least_active":
        return LeastActivePolicy()
    if mode == "threshold":
        return ThresholdPolicy(config["threshold"], config["short_pool"], config["long_pool"])
    profiles = {name: GPUProfile(**values) for name, values in config.get("gpu_profiles", {}).items()}
    model = CostModel(profiles)
    if mode == "adaptive":
        return AdaptivePolicy(model, config.get("hysteresis_fraction", .05))
    return DQAPolicy(model, KVReservations(),
                     config.get("min_benefit_fraction", .05),
                     config.get("hysteresis_fraction", .05),
                     config.get("max_queue_wait_s"))


def create_app(config: dict[str, Any]):
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    policy = make_policy(config)
    config_sha256 = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    pools = {entry["name"]: entry for entry in config["pools"]}
    health = {name: True for name in pools}
    active = {name: 0 for name in pools}
    lock = asyncio.Lock()
    client: httpx.AsyncClient | None = None
    health_task: asyncio.Task | None = None

    async def refresh_health() -> None:
        assert client is not None
        while True:
            for name, entry in pools.items():
                try:
                    response = await client.get(entry["url"].rstrip("/") + "/v1/models", timeout=5)
                    response.raise_for_status()
                    found = any(item.get("id") == config["model"] for item in response.json()["data"])
                except (httpx.HTTPError, KeyError, ValueError):
                    found = False
                async with lock:
                    health[name] = found
            await asyncio.sleep(config.get("health_interval_s", 5))

    @asynccontextmanager
    async def lifespan(_app):
        nonlocal client, health_task
        client = httpx.AsyncClient(timeout=config.get("timeout_s", 120))
        health_task = asyncio.create_task(refresh_health())
        try:
            yield
        finally:
            health_task.cancel()
            with suppress(asyncio.CancelledError):
                await health_task
            await client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def get_health():
        async with lock:
            return {"pools": dict(health), "active": dict(active), "mode": config["mode"]}

    @app.get("/v1/models")
    async def get_models():
        return {"object": "list", "data": [{"id": config["model"], "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(request: Request):
        nonlocal client
        try:
            payload = await request.json()
            prompt = payload["prompt"]
            if not isinstance(prompt, list) or not prompt or any(
                not isinstance(item, int) or item < 0 for item in prompt
            ) or payload.get("model") != config["model"] or not payload.get("stream"):
                raise ValueError("router requires one token-ID prompt and stream=true")
            spec = RequestSpec(str(uuid.uuid4()), 0, len(prompt), int(payload["max_tokens"]),
                               int(payload.get("seed", 0)), prompt)
        except (ValueError, TypeError, KeyError):
            return JSONResponse({"error": "invalid streaming Completions request"}, status_code=400)
        assert client is not None
        arrival = time.perf_counter()
        attempted: set[str] = set()
        for _ in pools:
            async with lock:
                states = make_states(config, health, active)
                states = [replace(state, available=False) if state.name in attempted else state
                          for state in states]
                try:
                    if config["mode"] == "static":
                        name = config["static_pool"]
                        if not any(state.name == name and state.available and state.deployed for state in states):
                            raise RuntimeError("static pool unavailable")
                    elif isinstance(policy, DQAPolicy):
                        name = policy.admit(spec, states)
                    else:
                        name = policy.choose(spec, states)
                    active[name] += 1
                except (RuntimeError, ValueError):
                    return JSONResponse({"error": "no serving pool has capacity"}, status_code=503)
            attempted.add(name)
            upstream = None
            try:
                upstream_request = client.build_request(
                    "POST", pools[name]["url"].rstrip("/") + "/v1/completions", json=payload)
                admission_wait = time.perf_counter() - arrival
                upstream = await client.send(upstream_request, stream=True)
                if upstream.status_code in {429, 503}:
                    await upstream.aclose()
                    async with lock:
                        health[name] = False
                    continue
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    await upstream.aclose()
                    return Response(content=body, status_code=upstream.status_code,
                                    media_type=upstream.headers.get("content-type"))
                async def body_stream():
                    try:
                        async for chunk in upstream.aiter_raw():
                            yield chunk
                    finally:
                        await upstream.aclose()
                        async with lock:
                            active[name] -= 1
                            if isinstance(policy, DQAPolicy):
                                policy.finish(spec.request_id)

                return StreamingResponse(body_stream(), media_type="text/event-stream",
                                         headers={"x-heteroserve-pool": name,
                                                  "x-heteroserve-config-sha256": config_sha256,
                                                  "x-heteroserve-admission-wait-s": str(admission_wait)})
            except httpx.HTTPError:
                async with lock:
                    health[name] = False
            finally:
                if upstream is None or upstream.is_closed:
                    async with lock:
                        active[name] -= 1
                        if isinstance(policy, DQAPolicy):
                            policy.finish(spec.request_id)
        return JSONResponse({"error": "all serving pools failed"}, status_code=503)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(load_config(args.config)), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
