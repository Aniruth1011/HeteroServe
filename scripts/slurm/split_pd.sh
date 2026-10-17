#!/usr/bin/env bash
# Real vLLM 0.11.2 A100-prefill / V100-decode experiment on two Slurm jobs.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODE="${1:-help}"
if [[ "$MODE" == help || "$MODE" == -h || "$MODE" == --help ]]; then
  echo "Usage: bash scripts/slurm/split_pd.sh prepare|prefill|decode A100_JOB_ID"
  echo "prepare: once in shared project; prefill: A100 allocation; decode: V100 allocation"
  exit 0
fi
VENV_PATH="${VENV_PATH:-$ROOT/.venv}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Missing virtual environment: $VENV_PATH" >&2
  exit 2
fi
source "$VENV_PATH/bin/activate"
mkdir -p results/raw/split_pd results/raw/capabilities results/processed scripts/vendor

MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
TOKENIZER_REVISION="$MODEL_REVISION"
export MODEL_ID MODEL_REVISION TOKENIZER_REVISION
PROXY=scripts/vendor/vllm_0_11_2_toy_proxy_server.py
PROXY_URL=https://raw.githubusercontent.com/vllm-project/vllm/v0.11.2/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

die() { echo "ERROR: $*" >&2; exit 1; }

check_versions() {
  python - <<'PY'
import importlib.metadata as m
import importlib.util
from packaging.version import Version
vllm = m.version("vllm")
nixl = m.version("nixl")
if vllm != "0.11.2" or Version(nixl) < Version("0.6.0"):
    raise SystemExit(f"Need vllm==0.11.2 and nixl>=0.6.0; got {vllm}, {nixl}")
if importlib.util.find_spec("nixl") is None:
    raise SystemExit("nixl package metadata exists but its Python module is missing")
print(f"vllm={vllm} nixl={nixl}")
PY
}

check_allocation() {
  local expected="$1" gpu
  [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_GPUS_ON_NODE:-}" == 1 ]] ||
    die "Run this role inside one regular Slurm GPU allocation"
  [[ "${SLURM_JOB_PARTITION:-}" != debug && "${SLURM_JOB_QOS:-}" != debug ]] ||
    die "Final split measurements require a regular GPU allocation"
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  [[ "$gpu" == *"$expected"* ]] || die "Expected one $expected, got: $gpu"
  echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$gpu"
}

eth0_host() {
  ip -4 -o addr show dev eth0 | awk '{split($4,a,"/");print a[1];exit}'
}

wait_models() {
  local url="$1" pid="$2" log="$3" attempt
  for attempt in $(seq 1 180); do
    if curl -fsS --max-time 5 "$url/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -n 80 "$log" >&2
      die "vLLM server exited before ready: $url"
    fi
    sleep 2
  done
  tail -n 80 "$log" >&2
  die "Timed out waiting for $url"
}

prepare() {
  python -m pip install 'nixl>=0.6.0'
  check_versions
  python -c 'import httpx, fastapi, uvicorn; print("proxy and replay dependencies present")'
  curl -fsSL --retry 3 "$PROXY_URL" -o "$PROXY"
  python -m py_compile "$PROXY"
  rg -q 'kv_transfer_params' "$PROXY" || grep -q 'kv_transfer_params' "$PROXY" ||
    die "Downloaded proxy is missing the KV handoff"
  python -m pip freeze > results/raw/split_pd/environment.txt
  sha256sum "$PROXY" scripts/slurm/split_pd.sh src/heteroserve/replay.py \
    src/heteroserve/split_metrics.py > results/raw/split_pd/source_sha256.txt
  echo "PREPARED: results/raw/split_pd/environment.txt"
}

prefill() {
  check_allocation A100
  check_versions
  local host url log cap ready server_pid telemetry_pid
  host="$(eth0_host)"
  [[ -n "$host" ]] || die "No eth0 IPv4 address"
  url="http://$host:8100"
  log="results/raw/split_pd/a100_${SLURM_JOB_ID}.server.log"
  cap="results/raw/capabilities/real_gpu_final_${SLURM_JOB_ID}_split_a100.json"
  ready="results/raw/split_pd/a100_${SLURM_JOB_ID}.ready.json"
  python scripts/probes/collect_capabilities.py --output "$cap" --evidence-label real_gpu_final
  python -m pip freeze > "results/raw/split_pd/a100_${SLURM_JOB_ID}.environment.txt"
  nvidia-smi --query-gpu=timestamp,index,name,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv --loop-ms=1000 > "results/raw/split_pd/a100_${SLURM_JOB_ID}.telemetry.csv" 2>&1 &
  telemetry_pid=$!
  PREFILL_READY_FILE="$ready"
  PREFILL_TEL_PID="$telemetry_pid"
  PREFILL_SERVER_PID=""
  cleanup_prefill() {
    rm -f "$PREFILL_READY_FILE"
    [[ -z "$PREFILL_SERVER_PID" ]] || { kill "$PREFILL_SERVER_PID" 2>/dev/null || true; wait "$PREFILL_SERVER_PID" 2>/dev/null || true; }
    kill "$PREFILL_TEL_PID" 2>/dev/null || true
    wait "$PREFILL_TEL_PID" 2>/dev/null || true
  }
  trap cleanup_prefill EXIT
  trap 'exit 143' TERM INT
  export VLLM_NIXL_SIDE_CHANNEL_HOST="$host" VLLM_NIXL_SIDE_CHANNEL_PORT=5600
  export UCX_NET_DEVICES=all VLLM_KV_CACHE_LAYOUT=HND
  export HOST="$host" PORT=8100 TENSOR_PARALLEL_SIZE=1
  bash scripts/serve_vllm.sh --dtype float16 --max-model-len 2048 --block-size 16 \
    --gpu-memory-utilization 0.5 --no-enable-prefix-caching --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
    > "$log" 2>&1 &
  server_pid=$!
  PREFILL_SERVER_PID="$server_pid"
  wait_models "$url" "$server_pid" "$log"
  python - "$ready.tmp" "$url" "$cap" "$log" <<'PY'
import json, os, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "job_id": os.environ["SLURM_JOB_ID"], "hostname": os.uname().nodename,
    "url": sys.argv[2], "capability": sys.argv[3], "server_log": sys.argv[4],
    "model_id": os.environ["MODEL_ID"],
    "model_revision": os.environ["MODEL_REVISION"],
    "tokenizer_revision": os.environ["TOKENIZER_REVISION"],
    "connector": "NixlConnector", "role": "prefill_producer",
    "kv_cache_layout": "HND", "block_size": 16, "dtype": "float16",
}, indent=2, sort_keys=True) + "\n")
PY
  mv "$ready.tmp" "$ready"
  echo "PREFILL_READY=$ready"
  echo "A100_JOB_ID=$SLURM_JOB_ID"
  wait "$server_pid"
}

snapshot() {
  local stem="$1" phase="$2"
  curl -fsS --max-time 15 "$PREFILL_URL/metrics" > "${stem}_prefill_${phase}.prom"
  curl -fsS --max-time 15 "$DECODE_URL/metrics" > "${stem}_decode_${phase}.prom"
}

stage_delta() {
  local stem="$1" output="$2"
  PYTHONPATH=src python -m heteroserve.split_metrics \
    --prefill-before "${stem}_prefill_before.prom" \
    --prefill-after "${stem}_prefill_after.prom" \
    --decode-before "${stem}_decode_before.prom" \
    --decode-after "${stem}_decode_after.prom" > "$output"
}

check_smoke() {
  local split_response="$1" direct_response="$2" stage_file="$3"
  python - "$split_response" "$direct_response" "$stage_file" <<'PY'
import json, sys
split = json.load(open(sys.argv[1]))
direct = json.load(open(sys.argv[2]))
stages = json.load(open(sys.argv[3]))
def completion(data):
    if not data.get("choices") or not isinstance(data["choices"][0].get("text"), str):
        raise SystemExit(f"Missing completion text: {data}")
    if not data["choices"][0]["text"].strip():
        raise SystemExit("Empty completion text")
    return data["choices"][0]["text"]
if completion(split) != completion(direct):
    raise SystemExit("Split and direct-decode smoke outputs differ; inspect model/connector")
bytes_stats = [role.get("kv_transfer_bytes") for role in stages.values()]
if not any(s and s["count"] > 0 and s["sum"] > 0 for s in bytes_stats):
    raise SystemExit("No measured NIXL KV bytes; HTTP 200 alone is not a split success")
for role in stages.values():
    for name in ("vllm:nixl_num_failed_transfers_total",
                 "vllm:nixl_num_failed_notifications_total"):
        if role.get(name, 0):
            raise SystemExit(f"NIXL failure observed: {name}")
print("SPLIT_SMOKE_OK: matching outputs and nonzero NIXL KV transfer")
PY
}

analyze_trials() {
  local run_id="$1" count="$2" a100_id="$3" cap="$4"
  python - "$run_id" "$count" "$a100_id" "$cap" <<'PY'
import hashlib, importlib.metadata, json, os, statistics, sys
from pathlib import Path
run, repeats, a100_id, cap = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
trials = []
for number in range(1, repeats + 1):
    stem = f"results/raw/split_pd/{run}_t{number}"
    manifest = json.loads(Path(stem + ".manifest.json").read_text())
    raw = Path(manifest["raw_results"])
    if (hashlib.sha256(raw.read_bytes()).hexdigest() != manifest["raw_results_sha256"]
            or manifest["evidence_label"] != "real_gpu_final"
            or manifest["environment"].get("slurm_job_id") != os.environ["SLURM_JOB_ID"]):
        raise SystemExit(f"trial {number} provenance or raw hash mismatch")
    stages = json.loads(Path(f"results/processed/{run}_t{number}_stages.json").read_text())
    summary = manifest["summary"]
    if summary["successes"] != summary["requests"]:
        raise SystemExit(f"trial {number} has unsuccessful requests")
    bytes_stats = [role.get("kv_transfer_bytes") for role in stages.values()]
    if not any(s and s["count"] > 0 and s["sum"] > 0 for s in bytes_stats):
        raise SystemExit(f"trial {number} has no measured KV transfer")
    if any(role.get(name, 0) for role in stages.values()
           for name in ("vllm:nixl_num_failed_transfers_total",
                        "vllm:nixl_num_failed_notifications_total")):
        raise SystemExit(f"trial {number} has NIXL failures")
    trials.append({"number": number, "manifest": stem + ".manifest.json",
                   "trace_sha256": manifest["trace_sha256"],
                   "config_sha256": manifest["config_sha256"],
                   "source_sha256": manifest["environment"]["source_sha256"],
                   "stage_metrics": f"results/processed/{run}_t{number}_stages.json",
                   "summary": summary, "stages": stages})
for field in ("trace_sha256", "config_sha256", "source_sha256"):
    if len({trial[field] for trial in trials}) != 1:
        raise SystemExit(f"{field} drift between measured trials")
fields = ("p50_ttft_s", "p95_ttft_s", "p99_ttft_s", "p95_e2e_s",
          "p95_admission_wait_s", "p95_scheduling_delay_s", "success_rate",
          "token_throughput_per_s", "e2e_goodput_tokens_per_s", "output_tokens")
metrics = {}
for field in fields:
    values = [trial["summary"].get(field) for trial in trials]
    metrics[field] = {"trial_values": values,
                      "median": statistics.median(values) if all(v is not None for v in values) else None}
stage_fields = ("prefill_s", "decode_s", "server_ttft_s", "server_inter_token_s",
                "kv_transfer_s", "kv_post_s", "kv_transfer_bytes")
stage_metrics = {}
for role in ("prefill", "decode"):
    stage_metrics[role] = {}
    for field in stage_fields:
        values = [trial["stages"][role].get(field) for trial in trials]
        means = [v["mean"] if v else None for v in values]
        stage_metrics[role][field] = {"trial_means": means,
            "median_mean": statistics.median(means) if all(v is not None for v in means) else None,
            "trial_counts": [v["count"] if v else None for v in values],
            "trial_sums": [v["sum"] if v else None for v in values]}
result = {"schema_version": 1, "evidence_label": "real_gpu_final",
          "deployment": "NixlConnector_A100_prefill_V100_decode_cross_node",
          "run_id": run, "a100_job_id": a100_id,
          "v100_job_id": os.environ["SLURM_JOB_ID"],
          "model_id": os.environ["MODEL_ID"],
          "model_revision": os.environ["MODEL_REVISION"],
          "vllm": "0.11.2", "nixl": importlib.metadata.version("nixl"),
          "repetitions": repeats,
          "prefill_capability": f"results/raw/capabilities/real_gpu_final_{a100_id}_split_a100.json",
          "decode_capability": cap,
          "trace_sha256": trials[0]["trace_sha256"],
          "config_sha256": trials[0]["config_sha256"],
          "source_sha256": trials[0]["source_sha256"],
          "client_metrics": metrics, "server_stage_metrics": stage_metrics,
          "trials": [{k: v for k, v in trial.items() if k not in ("summary", "stages")}
                     for trial in trials],
          "limitations": ["Server stage metrics are aggregate histogram deltas, not per-request timings.",
                          "Client text chunks do not establish exact per-token TPOT.",
                          "No matched colocated control is included in this split-only run."]}
out = Path(f"results/processed/{run}.split_analysis.json")
out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({"analysis": str(out), "run_id": run,
                  "p95_ttft_s": metrics["p95_ttft_s"]["median"],
                  "p95_e2e_s": metrics["p95_e2e_s"]["median"],
                  "throughput_tokens_per_s": metrics["token_throughput_per_s"]["median"],
                  "prefill_mean_s": stage_metrics["prefill"]["prefill_s"]["median_mean"],
                  "decode_mean_s": stage_metrics["decode"]["decode_s"]["median_mean"],
                  "nixl_bytes_mean_prefill": stage_metrics["prefill"]["kv_transfer_bytes"]["median_mean"],
                  "nixl_bytes_mean_decode": stage_metrics["decode"]["kv_transfer_bytes"]["median_mean"]},
                 indent=2, sort_keys=True))
PY
}

decode() {
  local a100_id="${2:-}" ready cap host stem log decode_pid proxy_pid telemetry_pid
  [[ "$a100_id" =~ ^[0-9]+$ ]] || die "Usage: bash scripts/slurm/split_pd.sh decode A100_JOB_ID"
  check_allocation V100
  check_versions
  [[ -f "$PROXY" ]] || die "Run prepare first; missing $PROXY"
  ready="results/raw/split_pd/a100_${a100_id}.ready.json"
  [[ -f "$ready" ]] || die "A100 ready file not found: $ready"
  PREFILL_URL="$(python - "$ready" "$a100_id" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if (d["job_id"] != sys.argv[2] or d["hostname"] == __import__("socket").gethostname() or
        d["model_id"] != "Qwen/Qwen2.5-0.5B-Instruct" or
        d["model_revision"] != "7ae557604adf67be50417f59c2c2f167def9a775" or
        d.get("kv_cache_layout") != "HND" or d.get("block_size") != 16 or
        d["connector"] != "NixlConnector"):
    raise SystemExit("A100 ready file does not match pinned split configuration")
print(d["url"])
PY
)"
  export PREFILL_URL
  host="$(eth0_host)"
  [[ -n "$host" ]] || die "No eth0 IPv4 address"
  DECODE_URL="http://$host:8200"
  export DECODE_URL
  curl -fsS --max-time 10 "$PREFILL_URL/v1/models" > results/raw/split_pd/prefill_models.json
  cap="results/raw/capabilities/real_gpu_final_${SLURM_JOB_ID}_split_v100.json"
  python scripts/probes/collect_capabilities.py --output "$cap" --evidence-label real_gpu_final
  python -m pip freeze > "results/raw/split_pd/v100_${SLURM_JOB_ID}.environment.txt"
  RUN_ID="split_a100_${a100_id}_v100_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
  export RUN_ID
  log="results/raw/split_pd/${RUN_ID}.decode.server.log"
  nvidia-smi --query-gpu=timestamp,index,name,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv --loop-ms=1000 > "results/raw/split_pd/${RUN_ID}.v100.telemetry.csv" 2>&1 &
  telemetry_pid=$!
  DECODE_SERVER_PID=""; DECODE_PROXY_PID=""; DECODE_TEL_PID="$telemetry_pid"
  cleanup_decode() {
    [[ -z "$DECODE_PROXY_PID" ]] || { kill "$DECODE_PROXY_PID" 2>/dev/null || true; wait "$DECODE_PROXY_PID" 2>/dev/null || true; }
    [[ -z "$DECODE_SERVER_PID" ]] || { kill "$DECODE_SERVER_PID" 2>/dev/null || true; wait "$DECODE_SERVER_PID" 2>/dev/null || true; }
    kill "$DECODE_TEL_PID" 2>/dev/null || true
    wait "$DECODE_TEL_PID" 2>/dev/null || true
  }
  trap cleanup_decode EXIT
  trap 'exit 143' TERM INT
  export VLLM_NIXL_SIDE_CHANNEL_HOST="$host" VLLM_NIXL_SIDE_CHANNEL_PORT=5600
  export UCX_NET_DEVICES=all VLLM_KV_CACHE_LAYOUT=HND
  export HOST="$host" PORT=8200 TENSOR_PARALLEL_SIZE=1
  bash scripts/serve_vllm.sh --dtype float16 --max-model-len 2048 --block-size 16 \
    --gpu-memory-utilization 0.5 --no-enable-prefix-caching --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
    > "$log" 2>&1 &
  decode_pid=$!
  DECODE_SERVER_PID="$decode_pid"
  wait_models "$DECODE_URL" "$decode_pid" "$log"
  local prefill_host="${PREFILL_URL#http://}"
  prefill_host="${prefill_host%:8100}"
  python "$PROXY" --host 127.0.0.1 --port 8192 \
    --prefiller-hosts "$prefill_host" --prefiller-ports 8100 \
    --decoder-hosts "$host" --decoder-ports 8200 \
    > "results/raw/split_pd/${RUN_ID}.proxy.log" 2>&1 &
  proxy_pid=$!
  DECODE_PROXY_PID="$proxy_pid"
  local attempt
  for attempt in $(seq 1 60); do
    if curl -fsS --max-time 5 http://127.0.0.1:8192/healthcheck >/dev/null 2>&1; then break; fi
    kill -0 "$proxy_pid" 2>/dev/null || { tail -n 80 "results/raw/split_pd/${RUN_ID}.proxy.log" >&2; die "proxy exited"; }
    sleep 1
  done
  curl -fsS --max-time 5 http://127.0.0.1:8192/healthcheck >/dev/null || die "proxy did not become ready"
  echo "DECODE_READY=$DECODE_URL PROXY=http://127.0.0.1:8192 RUN_ID=$RUN_ID"

  local payload='{"model":"heteroserve-model","prompt":"Explain one benefit of caching.","max_tokens":16,"temperature":0,"stream":false}'
  curl -fsS --max-time 120 "$DECODE_URL/v1/completions" -H 'Content-Type: application/json' \
    -d "$payload" > "results/raw/split_pd/${RUN_ID}.direct_smoke.json"
  stem="results/raw/split_pd/${RUN_ID}.split_smoke"
  snapshot "$stem" before
  curl -fsS --max-time 120 http://127.0.0.1:8192/v1/completions \
    -H 'Content-Type: application/json' -d "$payload" > "results/raw/split_pd/${RUN_ID}.split_smoke.json"
  sleep 12
  snapshot "$stem" after
  stage_delta "$stem" "results/processed/${RUN_ID}.smoke_stages.json"
  check_smoke "results/raw/split_pd/${RUN_ID}.split_smoke.json" \
    "results/raw/split_pd/${RUN_ID}.direct_smoke.json" \
    "results/processed/${RUN_ID}.smoke_stages.json"

  local warmup="${WARMUP_TRACE_PATH:-benchmarks/traces/a40_mixed_warmup.jsonl}"
  local measured="${TRACE_PATH:-benchmarks/traces/a40_mixed_measured.jsonl}"
  [[ -s "$warmup" && -s "$measured" ]] || die "Missing warmup or measured trace"
  local repeats="${REPETITIONS:-3}" concurrency="${CONCURRENCY:-8}"
  [[ "$repeats" =~ ^[0-9]+$ && "$repeats" -ge 3 ]] || die "REPETITIONS must be at least 3"
  PYTHONPATH=src python -m heteroserve.replay --trace "$warmup" \
    --base-url http://127.0.0.1:8192 --model-check-url "$DECODE_URL" \
    --output "results/raw/split_pd/${RUN_ID}_warmup.jsonl" --run-id "${RUN_ID}_warmup" \
    --model-revision "$MODEL_REVISION" --tokenizer-revision "$TOKENIZER_REVISION" \
    --prompt-token-id 23811 --concurrency "$concurrency" --timeout-s 120 \
    --slo-ttft-s 1 --slo-tpot-s 0.1 --slo-e2e-s 5 \
    --evidence-label gpu_dev_regular
  python - "results/raw/split_pd/${RUN_ID}_warmup.manifest.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
if s["successes"] != s["requests"]:
    raise SystemExit("Split warmup failed; measured trials canceled")
PY
  local trial
  for trial in $(seq 1 "$repeats"); do
    stem="results/raw/split_pd/${RUN_ID}_t${trial}"
    snapshot "$stem" before
    PYTHONPATH=src python -m heteroserve.replay --trace "$measured" \
      --base-url http://127.0.0.1:8192 --model-check-url "$DECODE_URL" \
      --output "${stem}.jsonl" --run-id "${RUN_ID}_t${trial}" \
      --model-revision "$MODEL_REVISION" --tokenizer-revision "$TOKENIZER_REVISION" \
      --prompt-token-id 23811 --concurrency "$concurrency" --timeout-s 120 \
      --slo-ttft-s 1 --slo-tpot-s 0.1 --slo-e2e-s 5 \
      --evidence-label real_gpu_final --save-output-text
    sleep 12
    snapshot "$stem" after
    stage_delta "$stem" "results/processed/${RUN_ID}_t${trial}_stages.json"
  done
  analyze_trials "$RUN_ID" "$repeats" "$a100_id" "$cap"
  echo "SPLIT_DONE run_id=$RUN_ID analysis=results/processed/${RUN_ID}.split_analysis.json"
}

case "$MODE" in
  prepare) prepare ;;
  prefill) prefill ;;
  decode) decode "$@" ;;
  *) die "Unknown role: $MODE" ;;
esac
