"""Build deterministic executable workloads from public trace artifacts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, TextIO

from .schema import WorkloadRequest

TokenCounter = Callable[[str], int]
PromptFormatter = Callable[[str], str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def whitespace_token_count(text: str) -> int:
    """CPU-test fallback only; final traces must use the pinned model tokenizer."""
    return max(1, len(text.split()))


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_oasst_pairs(path: Path, language: str = "en") -> list[tuple[str, str, str]]:
    """Return (source_id, prompt, best-ranked direct answer) from OASST messages."""
    messages: dict[str, dict[str, object]] = {}
    children: dict[str, list[dict[str, object]]] = {}
    with _open_text(path) as handle:
        for line in handle:
            row = json.loads(line)
            message_id = str(row["message_id"])
            messages[message_id] = row
            parent_id = row.get("parent_id")
            if parent_id:
                children.setdefault(str(parent_id), []).append(row)
    pairs: list[tuple[str, str, str]] = []
    for message_id, prompt in messages.items():
        if prompt.get("role") != "prompter" or prompt.get("lang") != language:
            continue
        candidates = [
            row for row in children.get(message_id, [])
            if row.get("role") == "assistant" and row.get("lang") == language
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda row: (
            int(row["rank"]) if row.get("rank") is not None else 10**9,
            str(row["message_id"]),
        ))
        answer = candidates[0]
        pairs.append((message_id, str(prompt["text"]).strip(), str(answer["text"]).strip()))
    return sorted((item for item in pairs if item[1] and item[2]), key=lambda item: item[0])


def load_burst_schedule(path: Path, limit: int | None = None) -> list[tuple[float, int, int, str]]:
    """Load relative arrival, input/output lengths, and row ID from BurstGPT CSV."""
    rows: list[tuple[float, int, int, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("BurstGPT CSV has no header")
        normalized = {name.strip().lower(): name for name in reader.fieldnames}
        required = {"timestamp", "request tokens", "response tokens"}
        missing = required - normalized.keys()
        if missing:
            raise ValueError(f"BurstGPT CSV missing columns: {sorted(missing)}")
        first_timestamp: float | None = None
        for index, row in enumerate(reader):
            output_tokens = int(float(row[normalized["response tokens"]]))
            if output_tokens < 1:
                continue
            timestamp = float(row[normalized["timestamp"]])
            first_timestamp = timestamp if first_timestamp is None else first_timestamp
            rows.append((
                timestamp - first_timestamp,
                int(float(row[normalized["request tokens"]])),
                output_tokens,
                str(index),
            ))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("BurstGPT CSV produced no successful requests")
    return rows


def build_hybrid_workload(
    pairs: Iterable[tuple[str, str, str]],
    schedule: list[tuple[float, int, int, str]],
    *,
    seed: int,
    token_counter: TokenCounter = whitespace_token_count,
    prompt_formatter: PromptFormatter = lambda value: value,
) -> list[WorkloadRequest]:
    """Pair real prompt text with a real arrival/length schedule.

    This is deliberately labeled ``hybrid_oasst1_burstgpt``: BurstGPT does not
    publish prompt text, so its requested input length is metadata rather than a
    guarantee that OASST text has the same token count.
    """
    available = list(pairs)
    if not available:
        raise ValueError("no OASST prompt/answer pairs")
    rng = random.Random(seed)
    rng.shuffle(available)
    requests: list[WorkloadRequest] = []
    for index, (arrival, _trace_prompt_tokens, output_tokens, burst_id) in enumerate(schedule):
        source_id, prompt, reference = available[index % len(available)]
        formatted_prompt = prompt_formatter(prompt)
        requests.append(WorkloadRequest(
            request_id=f"hybrid-{index:06d}",
            arrival_s=arrival,
            prompt_tokens=token_counter(formatted_prompt),
            max_output_tokens=max(1, output_tokens),
            seed=seed + index,
            prompt=formatted_prompt,
            source="hybrid_oasst1_burstgpt",
            source_id=f"oasst:{source_id};burst:{burst_id}",
            reference_output_tokens=token_counter(reference),
        ))
    return requests


def write_workload(requests: Iterable[WorkloadRequest], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(asdict(request), sort_keys=True) + "\n")


def read_workload(path: Path) -> list[WorkloadRequest]:
    with path.open(encoding="utf-8") as handle:
        return [WorkloadRequest(**json.loads(line)) for line in handle if line.strip()]
