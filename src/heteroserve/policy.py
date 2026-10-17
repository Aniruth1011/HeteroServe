from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .schema import RequestSpec


@dataclass(frozen=True)
class PoolState:
    name: str
    available: bool
    queue_wait_s: float
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    fixed_overhead_s: float = 0.0
    transfer_bandwidth_bytes_s: float | None = None
    transfer_fixed_s: float = 0.0
    bytes_per_prompt_token: int = 0
    gpu_type: str | None = None
    route_kind: str = "colocated"
    deployed: bool = True
    active_decode_requests: int = 0
    remaining_decode_tokens: int = 0
    recent_tpot_s: float | None = None
    kv_capacity_bytes: int | None = None
    kv_occupied_bytes: int = 0
    kv_reserved_bytes: int = 0
    prefill_batch_penalty: float = 0.0
    decode_batch_penalty: float = 0.0
    kv_bytes_per_token: int = 0


class Policy(Protocol):
    def choose(self, request: RequestSpec, pools: list[PoolState]) -> str: ...


@dataclass(frozen=True)
class GPUProfile:
    """Measured per-GPU rates and batching slopes; provenance belongs in the run manifest."""

    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    prefill_batch_penalty: float = 0.0
    decode_batch_penalty: float = 0.0


class CostModel:
    def __init__(self, profiles: dict[str, GPUProfile] | None = None) -> None:
        self.profiles = profiles or {}

    def predict(self, request: RequestSpec, pool: PoolState) -> float:
        profile = self.profiles.get(pool.gpu_type) if pool.gpu_type is not None else None
        prefill_rate = profile.prefill_tokens_per_s if profile else pool.prefill_tokens_per_s
        decode_rate = profile.decode_tokens_per_s if profile else pool.decode_tokens_per_s
        prefill_penalty = profile.prefill_batch_penalty if profile else pool.prefill_batch_penalty
        decode_penalty = profile.decode_batch_penalty if profile else pool.decode_batch_penalty
        if prefill_rate <= 0 or decode_rate <= 0:
            raise ValueError("pool rates must be positive")
        transfer = 0.0
        if pool.bytes_per_prompt_token:
            if not pool.transfer_bandwidth_bytes_s or pool.transfer_bandwidth_bytes_s <= 0:
                raise ValueError("transfer bandwidth required when transfer bytes are nonzero")
            transfer = (
                request.prompt_tokens * pool.bytes_per_prompt_token / pool.transfer_bandwidth_bytes_s
                + pool.transfer_fixed_s
            )
        prefill = request.prompt_tokens / prefill_rate
        prefill *= 1 + prefill_penalty * pool.active_decode_requests
        tpot = (1 + decode_penalty * pool.active_decode_requests) / decode_rate
        if pool.recent_tpot_s is not None:
            tpot = max(tpot, pool.recent_tpot_s)
        decode_queue = pool.remaining_decode_tokens * tpot / max(1, pool.active_decode_requests)
        return (pool.queue_wait_s + pool.fixed_overhead_s + prefill
                + request.max_output_tokens * tpot + decode_queue + transfer)


class KVReservations:
    """Client-side estimated decode KV admission; server occupancy must be refreshed externally."""

    def __init__(self) -> None:
        self._active: dict[str, tuple[str, int]] = {}

    def estimate_bytes(self, request: RequestSpec, pool: PoolState) -> int:
        return (request.prompt_tokens + request.max_output_tokens) * pool.kv_bytes_per_token

    def can_reserve(self, request: RequestSpec, pool: PoolState) -> bool:
        if pool.kv_capacity_bytes is None or pool.kv_bytes_per_token <= 0:
            return pool.route_kind != "disaggregated"
        return (pool.kv_occupied_bytes + pool.kv_reserved_bytes
                + self.reserved_bytes(pool.name) + self.estimate_bytes(request, pool)
                <= pool.kv_capacity_bytes)

    def reserve(self, request: RequestSpec, pool: PoolState) -> int:
        if request.request_id in self._active:
            raise ValueError("request already has a KV reservation")
        if not self.can_reserve(request, pool):
            raise RuntimeError("insufficient decode KV capacity")
        size = self.estimate_bytes(request, pool)
        self._active[request.request_id] = (pool.name, size)
        return size

    def release(self, request_id: str) -> int:
        entry = self._active.pop(request_id, None)
        return 0 if entry is None else entry[1]

    def reserved_bytes(self, pool_name: str) -> int:
        return sum(size for name, size in self._active.values() if name == pool_name)


class DQAPolicy:
    """Choose among deployed server pools; no in-flight request migration."""

    def __init__(self, model: CostModel, reservations: KVReservations,
                 min_benefit_fraction: float = 0.05, hysteresis_fraction: float = 0.05,
                 max_queue_wait_s: float | None = None) -> None:
        if min_benefit_fraction < 0 or hysteresis_fraction < 0:
            raise ValueError("benefit and hysteresis must be nonnegative")
        self.model = model
        self.reservations = reservations
        self.min_benefit_fraction = min_benefit_fraction
        self.hysteresis_fraction = hysteresis_fraction
        self.max_queue_wait_s = max_queue_wait_s
        self.previous: str | None = None

    def choose(self, request: RequestSpec, pools: list[PoolState]) -> str:
        eligible = [pool for pool in pools if pool.available and pool.deployed
                    and (self.max_queue_wait_s is None or pool.queue_wait_s <= self.max_queue_wait_s)
                    and (pool.route_kind != "disaggregated" or
                         (pool.transfer_bandwidth_bytes_s is not None
                          and pool.transfer_bandwidth_bytes_s > 0
                          and pool.bytes_per_prompt_token > 0))
                    and self.reservations.can_reserve(request, pool)]
        if not eligible:
            raise RuntimeError("no serving pool has available capacity")
        costs = {pool.name: self.model.predict(request, pool) for pool in eligible}
        colocated = [pool for pool in eligible if pool.route_kind == "colocated"]
        split = [pool for pool in eligible if pool.route_kind == "disaggregated"]
        if colocated and split:
            base = min(colocated, key=lambda pool: costs[pool.name])
            alternative = min(split, key=lambda pool: costs[pool.name])
            chosen = (alternative if costs[alternative.name] <
                      costs[base.name] * (1 - self.min_benefit_fraction) else base)
        else:
            chosen = min(eligible, key=lambda pool: costs[pool.name])
        if self.previous in costs and costs[self.previous] <= costs[chosen.name] * (1 + self.hysteresis_fraction):
            previous = next(pool for pool in eligible if pool.name == self.previous)
            if previous.route_kind != "disaggregated" or not colocated or (
                costs[previous.name] < min(costs[pool.name] for pool in colocated)
                * (1 - self.min_benefit_fraction)
            ):
                chosen = previous
        self.previous = chosen.name
        return chosen.name

    def admit(self, request: RequestSpec, pools: list[PoolState]) -> str:
        name = self.choose(request, pools)
        pool = next(pool for pool in pools if pool.name == name)
        self.reservations.reserve(request, pool)
        return name

    def finish(self, request_id: str) -> int:
        return self.reservations.release(request_id)


class ThresholdPolicy:
    def __init__(self, threshold: int, short_pool: str, long_pool: str) -> None:
        self.threshold, self.short_pool, self.long_pool = threshold, short_pool, long_pool

    def choose(self, request: RequestSpec, pools: list[PoolState]) -> str:
        available = {pool.name for pool in pools if pool.available and pool.deployed}
        preferred = self.long_pool if request.prompt_tokens >= self.threshold else self.short_pool
        fallback = self.short_pool if preferred == self.long_pool else self.long_pool
        if preferred in available:
            return preferred
        if fallback in available:
            return fallback
        raise RuntimeError("no serving pool is available")


class RoundRobinPolicy:
    """Use every deployed pool in turn; router admission lock serializes choose."""

    def __init__(self) -> None:
        self.next_index = 0

    def choose(self, request: RequestSpec, pools: list[PoolState]) -> str:
        del request
        if not pools:
            raise RuntimeError("no serving pool is available")
        for offset in range(len(pools)):
            index = (self.next_index + offset) % len(pools)
            if pools[index].available and pools[index].deployed:
                self.next_index = (index + 1) % len(pools)
                return pools[index].name
        raise RuntimeError("no serving pool is available")


class LeastActivePolicy:
    """Queue-aware two-pool heuristic; no GPU-rate calibration implied."""

    def __init__(self) -> None:
        self.tie_breaker = RoundRobinPolicy()

    def choose(self, request: RequestSpec, pools: list[PoolState]) -> str:
        available = [pool for pool in pools if pool.available and pool.deployed]
        if not available:
            raise RuntimeError("no serving pool is available")
        minimum = min(pool.active_decode_requests for pool in available)
        return self.tie_breaker.choose(
            request, [pool for pool in pools if pool.active_decode_requests == minimum]
        )


class AdaptivePolicy:
    def __init__(self, model: CostModel, hysteresis_fraction: float = 0.05) -> None:
        if hysteresis_fraction < 0:
            raise ValueError("hysteresis must be nonnegative")
        self.model, self.hysteresis_fraction, self.previous = model, hysteresis_fraction, None

    def choose(self, request: RequestSpec, pools: list[PoolState]) -> str:
        eligible = [pool for pool in pools if pool.available and pool.deployed]
        if not eligible:
            raise RuntimeError("no serving pool is available")
        costs = {pool.name: self.model.predict(request, pool) for pool in eligible}
        best = min(costs, key=costs.get)
        if self.previous in costs and costs[self.previous] <= costs[best] * (1 + self.hysteresis_fraction):
            best = self.previous
        self.previous = best
        return best
