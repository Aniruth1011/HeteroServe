#!/usr/bin/env bash
#SBATCH --job-name=heteroserve-final-colocated
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script through Slurm}"
: "${VENV_PATH:?Set absolute virtual environment path}"
: "${MODEL_ID:?Set accessible model repository}"
: "${MODEL_REVISION:?Set full model commit}"
: "${TOKENIZER_REVISION:?Set full tokenizer commit}"
: "${TRACE_PATH:?Set measured trace path}"
: "${WARMUP_TRACE_PATH:?Set warmup trace path}"
: "${PROMPT_TOKEN_ID:?Set a validated ordinary token ID}"
: "${RUN_ID:?Set a unique run ID}"
: "${SLO_TTFT_S:?Declare TTFT SLO before execution}"
: "${SLO_TPOT_S:?Declare TPOT SLO before execution}"
: "${SLO_E2E_S:?Declare E2E SLO before execution}"

if [[ "${SLURM_JOB_PARTITION:-}" == debug || "${SLURM_JOB_QOS:-}" == debug ]]; then
  echo "Final evaluation requires a regular GPU allocation" >&2
  exit 2
fi
if [[ "${SLURM_GPUS_ON_NODE:-0}" != 1 ]]; then
  echo "This validated colocated configuration requires exactly one GPU" >&2
  exit 2
fi

cd "${SLURM_SUBMIT_DIR}"
source "${VENV_PATH}/bin/activate"
mkdir -p results/raw/capabilities results/processed
run_dir="results/raw/${RUN_ID}"
capability="results/raw/capabilities/real_gpu_final_${SLURM_JOB_ID}_${RUN_ID}.json"
python scripts/probes/collect_capabilities.py --output "${capability}" --evidence-label real_gpu_final
python -m pip freeze > "results/raw/${RUN_ID}.environment.txt"
python - <<'PY' > "results/raw/${RUN_ID}.server_config.json"
import json, os
print(json.dumps({"model_id": os.environ["MODEL_ID"],
                  "model_revision": os.environ["MODEL_REVISION"],
                  "tokenizer_revision": os.environ["TOKENIZER_REVISION"],
                  "served_model_name": "heteroserve-model", "tensor_parallel_size": 1,
                  "dtype": "float16", "max_model_len": 2048,
                  "gpu_memory_utilization": 0.5, "prefix_caching": False,
                  "generation_config": "vllm", "port": int(os.getenv("PORT", "8000"))},
                 sort_keys=True, indent=2))
PY
export TENSOR_PARALLEL_SIZE=1
export PORT="${PORT:-8000}"
bash scripts/serve_vllm.sh --dtype float16 --max-model-len 2048 \
  --gpu-memory-utilization 0.5 --no-enable-prefix-caching \
  > "results/raw/${RUN_ID}.server.log" 2>&1 &
server_pid=$!
cleanup() { kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT
ready=0
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then break; fi
  sleep 2
done
if [[ "${ready}" != 1 ]]; then
  tail -n 80 "results/raw/${RUN_ID}.server.log" >&2
  exit 1
fi

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv --loop-ms=1000 > "results/raw/${RUN_ID}.telemetry.csv" 2>&1 &
telemetry_pid=$!
cleanup_all() {
  kill "${telemetry_pid}" 2>/dev/null || true
  wait "${telemetry_pid}" 2>/dev/null || true
  cleanup
}
trap cleanup_all EXIT
trap 'cleanup_all; exit 143' TERM INT

PYTHONPATH=src python -m heteroserve.evaluation \
  --trace "${TRACE_PATH}" --warmup-trace "${WARMUP_TRACE_PATH}" \
  --capabilities "${capability}" --server-config "results/raw/${RUN_ID}.server_config.json" \
  --output-dir "${run_dir}" --run-id "${RUN_ID}" \
  --trials "${TRIALS:-5}" --model-revision "${MODEL_REVISION}" \
  --model-id "${MODEL_ID}" \
  --tokenizer-revision "${TOKENIZER_REVISION}" --prompt-token-id "${PROMPT_TOKEN_ID}" \
  --concurrency "${CONCURRENCY:-8}" --slo-ttft-s "${SLO_TTFT_S}" \
  --slo-tpot-s "${SLO_TPOT_S}" --slo-e2e-s "${SLO_E2E_S}" \
  --base-url "http://127.0.0.1:${PORT}"
PYTHONPATH=src python -m heteroserve.analysis \
  --index "${run_dir}/evaluation.json" \
  --output "results/processed/${RUN_ID}.analysis.json"
