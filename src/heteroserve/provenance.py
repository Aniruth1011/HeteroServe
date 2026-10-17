"""Run manifests and deterministic artifact hashing."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(command: list[str]) -> str | None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_manifest(config_path: Path, trace_path: Path, run_id: str) -> dict[str, Any]:
    slurm = {key: value for key, value in os.environ.items() if key.startswith("SLURM_")}
    declared_config = json.loads(config_path.read_text(encoding="utf-8"))
    evidence_label = os.getenv("HETEROSERVE_EVIDENCE_LABEL")
    if not evidence_label:
        evidence_label = "gpu_debug" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu_test"
    return {
        "schema_version": 1,
        "evidence_label": evidence_label,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": sha256_path(config_path)},
        "declared_config": declared_config,
        "trace": {"path": str(trace_path), "sha256": sha256_path(trace_path)},
        "source": {
            "git_sha": _command(["git", "rev-parse", "HEAD"]),
            "git_status": _command(["git", "status", "--short"]),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "vllm": package_version("vllm"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
        "slurm": slurm,
    }


def write_manifest(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
