#!/usr/bin/env python3
"""Collect host, Slurm, GPU, and Python capability evidence as JSON."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

SLURM_KEYS = (
    "SLURM_CLUSTER_NAME", "SLURM_JOB_ID", "SLURM_JOB_NAME",
    "SLURM_JOB_ACCOUNT", "SLURM_JOB_PARTITION", "SLURM_JOB_QOS",
    "SLURM_JOB_NODELIST", "SLURM_JOB_NUM_NODES", "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_NTASKS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS",
    "CUDA_VISIBLE_DEVICES",
)
PACKAGE_NAMES = ("torch", "vllm", "transformers", "tokenizers", "numpy")


def run(command: list[str], timeout_seconds: int = 20) -> dict[str, Any]:
    result: dict[str, Any] = {"command": shlex.join(command), "timeout_seconds": timeout_seconds}
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=timeout_seconds
        )
    except FileNotFoundError as exc:
        result.update(status="not_found", error=str(exc))
    except subprocess.TimeoutExpired as exc:
        result.update(status="timeout", stdout=exc.stdout or "", stderr=exc.stderr or "")
    else:
        result.update(
            status="ok" if completed.returncode == 0 else "error",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    return result


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def has_gpu_allocation() -> bool:
    return any(
        os.getenv(key)
        for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES")
    )


def evidence_label() -> str:
    if not has_gpu_allocation():
        return "cpu_context"
    partition = os.getenv("SLURM_JOB_PARTITION", "").lower()
    qos = os.getenv("SLURM_JOB_QOS", "").lower()
    if "debug" in partition or "debug" in qos:
        return "gpu_debug"
    return "gpu_dev_regular"


def collect(label: str | None = None) -> dict[str, Any]:
    detected = evidence_label()
    if label == "real_gpu_final" and detected != "gpu_dev_regular":
        raise ValueError("real_gpu_final requires a regular GPU allocation")
    torch_probe = (
        "import json, torch; print(json.dumps({"
        "'version': torch.__version__, 'cuda_build': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count(), "
        "'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"
    )
    return {
        "schema_version": 1,
        "evidence_label": label or detected,
        "collected_at": datetime.now().astimezone().isoformat(),
        "host": {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version, "executable": sys.executable,
        },
        "slurm": {key: os.getenv(key) for key in SLURM_KEYS},
        "packages": package_versions(),
        "probes": {
            "nvidia_smi_query": run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"]),
            "nvidia_smi_topology": run(["nvidia-smi", "topo", "-m"]),
            "torch_runtime": run([sys.executable, "-c", torch_probe], timeout_seconds=60),
            "cpu": run(["lscpu"]), "memory": run(["free", "-b"]),
            "network": run(["ip", "-brief", "address"]),
        },
    }


def write_json(payload: dict[str, Any], output: str) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output == "-":
        sys.stdout.write(serialized)
        return
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="-", help="JSON destination, or '-' for stdout")
    parser.add_argument("--evidence-label", choices=["real_gpu_final"])
    args = parser.parse_args()
    write_json(collect(args.evidence_label), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
