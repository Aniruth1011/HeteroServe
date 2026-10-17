"""OpenAI-compatible streaming client with client-clock latency timestamps."""

from __future__ import annotations

import hashlib
import http.client
import json
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from .schema import RequestResult, WorkloadRequest


@dataclass(frozen=True)
class Endpoint:
    name: str
    base_url: str
    model: str
    timeout_s: float = 120.0


def _completion_path(base_url: str) -> tuple[str, str, int, str]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid endpoint URL: {base_url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    prefix = parsed.path.rstrip("/")
    return parsed.scheme, parsed.hostname, port, f"{prefix}/v1/completions"


def stream_completion(
    endpoint: Endpoint,
    request: WorkloadRequest,
    submitted_s: float,
    admission_wait_s: float,
    controller_overhead_s: float,
) -> RequestResult:
    """Execute one request. Timestamps use ``time.perf_counter`` on the client."""
    scheme, host, port, path = _completion_path(endpoint.base_url)
    connection_type = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    connection = connection_type(host, port, timeout=endpoint.timeout_s)
    first_token_s: float | None = None
    output_parts: list[str] = []
    output_tokens = 0
    finish_reason: str | None = None
    status: int | None = None
    error: str | None = None
    try:
        body = json.dumps({
            "model": endpoint.model,
            "prompt": request.prompt,
            "max_tokens": request.max_output_tokens,
            "temperature": 0,
            "seed": request.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        status = response.status
        if status != 200:
            error = response.read().decode("utf-8", errors="replace")[:2000]
        else:
            while True:
                raw = response.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    output_tokens = int(usage["completion_tokens"])
                for choice in event.get("choices") or []:
                    text = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                    if text:
                        first_token_s = first_token_s or time.perf_counter()
                        output_parts.append(text)
                    finish_reason = choice.get("finish_reason") or finish_reason
    except Exception as exc:  # preserved as per-request failure evidence
        error = f"{type(exc).__name__}: {exc}"
    finally:
        completed_s = time.perf_counter()
        connection.close()
    text = "".join(output_parts)
    if output_tokens < 1 and text:
        # Diagnostic fallback only. Final manifests should show usage support.
        output_tokens = max(1, len(text.split()))
    success = status == 200 and first_token_s is not None and output_tokens > 0 and error is None
    return RequestResult(
        request_id=request.request_id,
        submitted_s=submitted_s,
        first_token_s=first_token_s,
        completed_s=completed_s,
        output_tokens=output_tokens,
        success=success,
        admission_wait_s=admission_wait_s,
        pool=endpoint.name,
        error=error,
        finish_reason=finish_reason,
        response_text_sha256=hashlib.sha256(text.encode()).hexdigest() if text else None,
        controller_overhead_s=controller_overhead_s,
        http_status=status,
    )
