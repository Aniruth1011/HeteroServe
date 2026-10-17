from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_s: float
    prompt_tokens: int
    max_output_tokens: int
    seed: int
    prompt_token_ids: list[int] | None = None

    def __post_init__(self) -> None:
        if self.arrival_s < 0 or self.prompt_tokens < 1 or self.max_output_tokens < 1:
            raise ValueError("arrival must be nonnegative and token counts must be positive")
        if self.prompt_token_ids is not None and (
            len(self.prompt_token_ids) != self.prompt_tokens
            or any(not isinstance(token, int) or token < 0 for token in self.prompt_token_ids)
        ):
            raise ValueError("prompt_token_ids must match prompt_tokens and contain nonnegative IDs")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadRequest:
    """Text-prompt request used by the legacy public-trace runner."""

    request_id: str
    arrival_s: float
    prompt_tokens: int
    max_output_tokens: int
    seed: int
    prompt: str
    source: str
    source_id: str
    reference_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if (not self.request_id or self.arrival_s < 0 or self.prompt_tokens < 1
                or self.max_output_tokens < 1 or not self.prompt or not self.source):
            raise ValueError("invalid text workload request")
        if self.reference_output_tokens is not None and self.reference_output_tokens < 1:
            raise ValueError("reference output tokens must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    submitted_s: float
    first_token_s: float | None
    completed_s: float
    output_tokens: int
    success: bool
    admission_wait_s: float
    pool: str
    error: str | None = None
    last_token_s: float | None = None
    exact_token_timing: bool = False
    scheduled_s: float | None = None
    actual_arrival_s: float | None = None
    server_submission_s: float | None = None
    outcome: str = "success"
    finish_reason: str | None = None
    response_text_sha256: str | None = None
    controller_overhead_s: float = 0.0
    http_status: int | None = None
    attempts: int = 1
    fallback_from: str | None = None
    attempt_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.completed_s < self.submitted_s or self.output_tokens < 0:
            raise ValueError("invalid completion time or output token count")
        if self.first_token_s is not None and not (
            self.submitted_s <= self.first_token_s <= self.completed_s
        ):
            raise ValueError("first-token timestamp is outside request lifetime")
        if self.admission_wait_s < 0:
            raise ValueError("admission wait must be nonnegative")
        if self.last_token_s is not None and (
            self.first_token_s is None or not self.first_token_s <= self.last_token_s <= self.completed_s
        ):
            raise ValueError("last-token timestamp is outside request lifetime")
        if self.outcome not in {"success", "rejected", "failed", "timed_out"}:
            raise ValueError("invalid request outcome")
        if self.controller_overhead_s < 0 or self.attempts < 1:
            raise ValueError("controller overhead and attempts must be nonnegative")
        if not self.success and self.outcome == "success":
            if self.http_status in {429, 503} or self.error == "admission queue full":
                inferred = "rejected"
            elif self.error and "timeout" in self.error.lower():
                inferred = "timed_out"
            else:
                inferred = "failed"
            object.__setattr__(self, "outcome", inferred)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ttft_s": self.ttft_s, "tpot_s": self.tpot_s,
                "approximate_tpot_s": self.approximate_tpot_s,
                "scheduling_delay_s": self.scheduling_delay_s, "e2e_s": self.e2e_s}

    @property
    def ttft_s(self) -> float | None:
        return None if self.first_token_s is None else self.first_token_s - self.submitted_s

    @property
    def tpot_s(self) -> float | None:
        if self.first_token_s is None or self.output_tokens < 2 or not self.exact_token_timing:
            return None
        if self.last_token_s is None:
            return None
        return (self.last_token_s - self.first_token_s) / (self.output_tokens - 1)

    @property
    def approximate_tpot_s(self) -> float | None:
        if self.first_token_s is None or self.last_token_s is None or self.output_tokens < 2:
            return None
        return (self.last_token_s - self.first_token_s) / (self.output_tokens - 1)

    @property
    def scheduling_delay_s(self) -> float | None:
        if self.scheduled_s is None or self.actual_arrival_s is None:
            return None
        return max(0.0, self.actual_arrival_s - self.scheduled_s)

    @property
    def e2e_s(self) -> float:
        return self.completed_s - self.submitted_s
