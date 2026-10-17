#!/usr/bin/env bash
#SBATCH --job-name=heteroserve-final-paired-proxy
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
: "${VENV_PATH:?Set absolute virtual environment path}"
: "${MODEL_ID:?Set model repository}"
: "${MODEL_REVISION:?Set exact model commit}"
: "${TOKENIZER_REVISION:?Set exact tokenizer commit}"
: "${TRACE_PATH:?Set measured trace path}"
: "${WARMUP_TRACE_PATH:?Set warmup trace path}"
: "${PROMPT_TOKEN_ID:?Set ordinary token ID}"
: "${RUN_ID:?Set unique run ID}"
: "${SLO_TTFT_S:?Set predeclared TTFT SLO}"
: "${SLO_TPOT_S:?Set predeclared TPOT SLO}"
: "${SLO_E2E_S:?Set predeclared E2E SLO}"

if [[ "${SLURM_JOB_PARTITION:-}" == debug || "${SLURM_JOB_QOS:-}" == debug ]]; then
  echo "Final evaluation requires a regular GPU allocation" >&2
  exit 2
fi
if [[ "${SLURM_GPUS_ON_NODE:-0}" != 1 ]]; then
  echo "This paired control requires exactly one allocated GPU" >&2
  exit 2
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "${VENV_PATH}/bin/activate"
mkdir -p results/raw/capabilities results/processed
run_dir="results/raw/${RUN_ID}"
capability="results/raw/capabilities/real_gpu_final_${SLURM_JOB_ID}_${RUN_ID}.json"
server_config="results/raw/${RUN_ID}.server_config.json"
router_config="results/raw/${RUN_ID}.router_config.json"
python scripts/probes/collect_capabilities.py --output "${capability}" --evidence-label real_gpu_final
python -m pip freeze > "results/raw/${RUN_ID}.environment.txt"
python - <<'PY' > "${server_config}"
import json, os
print(json.dumps({"model_id": os.environ["MODEL_ID"],
                  "model_revision": os.environ["MODEL_REVISION"],
                  "tokenizer_revision": os.environ["TOKENIZER_REVISION"],
                  "served_model_name": "heteroserve-model", "tensor_parallel_size": 1,
                  "dtype": "float16", "max_model_len": 2048,
                  "gpu_memory_utilization": 0.5, "prefix_caching": False,
                  "generation_config": "vllm", "port": 8000},
                 sort_keys=True, indent=2))
PY
python - <<'PY' > "${router_config}"
import json, os
print(json.dumps({"mode": "static", "model": "heteroserve-model",
                  "static_pool": "a40_colocated",
                  "max_inflight_per_pool": int(os.getenv("CONCURRENCY", "8")),
                  "health_interval_s": 1, "timeout_s": 120,
                  "pools": [{"name": "a40_colocated", "url": "http://127.0.0.1:8000",
                             "route_kind": "colocated", "deployed": True}]},
                 sort_keys=True, indent=2))
PY

export TENSOR_PARALLEL_SIZE=1 PORT=8000
bash scripts/serve_vllm.sh --dtype float16 --max-model-len 2048 \
  --gpu-memory-utilization 0.5 --no-enable-prefix-caching \
  > "results/raw/${RUN_ID}.server.log" 2>&1 &
server_pid=$!
router_pid=""
telemetry_pid=""
cleanup() {
  for pid in "${telemetry_pid}" "${router_pid}" "${server_pid}"; do
    if [[ -n "${pid}" ]]; then kill "${pid}" 2>/dev/null || true; fi
  done
  for pid in "${telemetry_pid}" "${router_pid}" "${server_pid}"; do
    if [[ -n "${pid}" ]]; then wait "${pid}" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
trap 'exit 143' TERM INT

ready=0
for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then break; fi
  sleep 2
done
if [[ "${ready}" != 1 ]]; then
  tail -n 80 "results/raw/${RUN_ID}.server.log" >&2
  exit 1
fi

PYTHONPATH=src python -m heteroserve.router --config "${router_config}" --port 8100 \
  > "results/raw/${RUN_ID}.router.log" 2>&1 &
router_pid=$!
ready=0
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8100/health 2>/dev/null | python -c \
     'import json,sys; sys.exit(not json.load(sys.stdin)["pools"].get("a40_colocated"))' \
     2>/dev/null; then ready=1; break; fi
  if ! kill -0 "${router_pid}" 2>/dev/null; then break; fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then
  tail -n 80 "results/raw/${RUN_ID}.router.log" >&2
  exit 1
fi

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv --loop-ms=1000 > "results/raw/${RUN_ID}.telemetry.csv" 2>&1 &
telemetry_pid=$!

PYTHONPATH=src python -m heteroserve.paired run \
  --trace "${TRACE_PATH}" --warmup-trace "${WARMUP_TRACE_PATH}" \
  --capabilities "${capability}" --server-config "${server_config}" \
  --router-config "${router_config}" --output-dir "${run_dir}" \
  --run-id "${RUN_ID}" --model-revision "${MODEL_REVISION}" \
  --prompt-token-id "${PROMPT_TOKEN_ID}" --concurrency "${CONCURRENCY:-8}" \
  --pairs "${PAIRS:-5}" --slo-ttft-s "${SLO_TTFT_S}" \
  --slo-tpot-s "${SLO_TPOT_S}" --slo-e2e-s "${SLO_E2E_S}"
PYTHONPATH=src python -m heteroserve.paired analyze \
  --index "${run_dir}/paired_index.json" \
  --output "results/processed/${RUN_ID}.paired_analysis.json"
