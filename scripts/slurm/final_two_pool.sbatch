#!/usr/bin/env bash
#SBATCH --job-name=heteroserve-two-a40
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
  echo "Final evaluation requires a regular GPU allocation" >&2; exit 2
fi
if [[ "${SLURM_GPUS_ON_NODE:-}" != 2 ]]; then
  echo "Exactly two GPUs must be allocated on this node" >&2; exit 2
fi
if [[ "${MODEL_REVISION}" != "${TOKENIZER_REVISION}" ]]; then
  echo "Model and tokenizer revisions must match" >&2; exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != *,* || "${CUDA_VISIBLE_DEVICES}" == *,*,* ]]; then
  echo "Expected two comma-separated CUDA_VISIBLE_DEVICES entries from Slurm" >&2; exit 2
fi
IFS=, read -r gpu0 gpu1 <<< "${CUDA_VISIBLE_DEVICES}"
if [[ -z "${gpu0}" || -z "${gpu1}" || "${gpu0}" == "${gpu1}" ]]; then
  echo "Two distinct allocated GPU identifiers required" >&2; exit 2
fi

cd "${HETEROSERVE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${VENV_PATH}/bin/activate"
mkdir -p results/raw/capabilities results/processed
run_dir="results/raw/${RUN_ID}"
capability="results/raw/capabilities/real_gpu_final_${SLURM_JOB_ID}_${RUN_ID}.json"
server_config="results/raw/${RUN_ID}.server_config.json"
arms_file="results/raw/${RUN_ID}.arms.json"
python scripts/probes/collect_capabilities.py --output "${capability}" --evidence-label real_gpu_final
python -m pip freeze > "results/raw/${RUN_ID}.environment.txt"
python - <<'PY' > "${server_config}"
import json, os
print(json.dumps({"model_id": os.environ["MODEL_ID"], "model_revision": os.environ["MODEL_REVISION"],
                  "tokenizer_revision": os.environ["TOKENIZER_REVISION"], "served_model_name": "heteroserve-model",
                  "tensor_parallel_size_per_server": 1, "dtype": "float16", "max_model_len": 2048,
                  "gpu_memory_utilization": 0.5, "prefix_caching": False,
                  "generation_config": "vllm", "ports": [8000, 8001],
                  "deployment": "two_independent_colocated_pools"}, sort_keys=True, indent=2))
PY

server_pids=()
router_pids=()
telemetry_pid=""
cleanup() {
  for pid in "${router_pids[@]}" "${server_pids[@]}" "${telemetry_pid}"; do
    if [[ -n "${pid}" ]]; then kill "${pid}" 2>/dev/null || true; fi
  done
  for pid in "${router_pids[@]}" "${server_pids[@]}" "${telemetry_pid}"; do
    if [[ -n "${pid}" ]]; then wait "${pid}" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
trap 'exit 143' TERM INT

for entry in "0:${gpu0}:8000" "1:${gpu1}:8001"; do
  IFS=: read -r index device port <<< "${entry}"
  CUDA_VISIBLE_DEVICES="${device}" TENSOR_PARALLEL_SIZE=1 PORT="${port}" \
    bash scripts/serve_vllm.sh --dtype float16 --max-model-len 2048 \
      --gpu-memory-utilization 0.5 --no-enable-prefix-caching \
      > "results/raw/${RUN_ID}.gpu${index}.server.log" 2>&1 &
  server_pids+=("$!")
done
for index in 0 1; do
  port=$((8000 + index))
  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "${server_pids[$index]}" 2>/dev/null; then break; fi
    sleep 2
  done
  if [[ "${ready}" != 1 ]]; then
    tail -n 80 "results/raw/${RUN_ID}.gpu${index}.server.log" >&2; exit 1
  fi
done

export RUN_ID
python - <<'PY'
import json, os
from pathlib import Path
run = os.environ["RUN_ID"]
common = {"model": "heteroserve-model", "max_inflight_per_pool": int(os.getenv("CONCURRENCY", "8")),
          "health_interval_s": 1, "timeout_s": 120,
          "pools": [{"name": f"gpu{i}", "url": f"http://127.0.0.1:{8000+i}",
                     "route_kind": "colocated", "deployed": True} for i in range(2)]}
arms = []
for offset, mode in enumerate(("round_robin", "threshold", "least_active")):
    config = {**common, "mode": mode}
    if mode == "threshold":
        config.update({"short_pool": "gpu0", "long_pool": "gpu1", "threshold": 128})
    path = Path(f"results/raw/{run}.{mode}.router_config.json")
    path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
    arms.append({"name": mode, "url": f"http://127.0.0.1:{8100+offset}",
                 "router_config": str(path)})
Path(f"results/raw/{run}.arms.json").write_text(json.dumps({"arms": arms}, sort_keys=True, indent=2) + "\n")
PY

for offset in 0 1 2; do
  case "${offset}" in 0) mode=round_robin;; 1) mode=threshold;; 2) mode=least_active;; esac
  port=$((8100 + offset))
  PYTHONPATH=src python -m heteroserve.router \
    --config "results/raw/${RUN_ID}.${mode}.router_config.json" --port "${port}" \
    > "results/raw/${RUN_ID}.${mode}.router.log" 2>&1 &
  router_pids+=("$!")
done
for offset in 0 1 2; do
  case "${offset}" in 0) mode=round_robin;; 1) mode=threshold;; 2) mode=least_active;; esac
  port=$((8100 + offset))
  ready=0
  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:${port}/health" 2>/dev/null | python -c \
       'import json,sys; p=json.load(sys.stdin)["pools"]; sys.exit(not (p.get("gpu0") and p.get("gpu1")))' \
       2>/dev/null; then ready=1; break; fi
    if ! kill -0 "${router_pids[$offset]}" 2>/dev/null; then break; fi
    sleep 1
  done
  if [[ "${ready}" != 1 ]]; then
    tail -n 80 "results/raw/${RUN_ID}.${mode}.router.log" >&2; exit 1
  fi
done

nvidia-smi --query-gpu=timestamp,index,name,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv --loop-ms=1000 > "results/raw/${RUN_ID}.telemetry.csv" 2>&1 &
telemetry_pid=$!

PYTHONPATH=src python -m heteroserve.multiarm run \
  --trace "${TRACE_PATH}" --warmup-trace "${WARMUP_TRACE_PATH}" \
  --capabilities "${capability}" --server-config "${server_config}" --arms-file "${arms_file}" \
  --output-dir "${run_dir}" --run-id "${RUN_ID}" --model-id "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" --prompt-token-id "${PROMPT_TOKEN_ID}" \
  --concurrency "${CONCURRENCY:-8}" --repetitions "${REPETITIONS:-3}" \
  --slo-ttft-s "${SLO_TTFT_S}" --slo-tpot-s "${SLO_TPOT_S}" --slo-e2e-s "${SLO_E2E_S}"
PYTHONPATH=src python -m heteroserve.multiarm analyze \
  --index "${run_dir}/multiarm_index.json" \
  --output "results/processed/${RUN_ID}.multiarm_analysis.json"
