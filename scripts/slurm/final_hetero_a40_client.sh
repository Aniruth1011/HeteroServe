#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run in a regular A40 Slurm allocation}"
: "${HETEROSERVE_ROOT:?Set project root}"
: "${VENV_PATH:?Set virtual environment path}"
: "${MODEL_ID:?Set model ID}"
: "${MODEL_REVISION:?Set model commit}"
: "${TOKENIZER_REVISION:?Set tokenizer commit}"
: "${TRACE_PATH:?Set measured trace}"
: "${WARMUP_TRACE_PATH:?Set warmup trace}"
: "${PROMPT_TOKEN_ID:?Set ordinary token ID}"
: "${RUN_ID:?Set unique run ID}"
: "${A100_URL:?Set running A100 endpoint URL}"
: "${A100_CAPABILITY:?Set A100 capability JSON path}"
: "${SLO_TTFT_S:?Set predeclared TTFT SLO}"
: "${SLO_TPOT_S:?Set predeclared TPOT SLO}"
: "${SLO_E2E_S:?Set predeclared E2E SLO}"
: "${CLIENT_GPU_TYPE:=a40}"
if [[ "${CLIENT_GPU_TYPE}" != a40 && "${CLIENT_GPU_TYPE}" != v100 ]]; then
  echo "CLIENT_GPU_TYPE must be a40 or v100" >&2; exit 2
fi
export CLIENT_GPU_TYPE
if [[ "${SLURM_GPUS_ON_NODE:-}" != 1 || "${SLURM_JOB_PARTITION:-}" == debug || "${SLURM_JOB_QOS:-}" == debug ]]; then
  echo "Requires one regular allocated A40" >&2; exit 2
fi
cd "${HETEROSERVE_ROOT}"
source "${VENV_PATH}/bin/activate"
mkdir -p results/raw/capabilities results/processed
cap="results/raw/capabilities/real_gpu_final_${SLURM_JOB_ID}_${RUN_ID}_${CLIENT_GPU_TYPE}.json"
server_config="results/raw/${RUN_ID}.server_config.json"
arms_file="results/raw/${RUN_ID}.arms.json"
run_dir="results/raw/${RUN_ID}"
python scripts/probes/collect_capabilities.py --output "${cap}" --evidence-label real_gpu_final
python - "${cap}" "${CLIENT_GPU_TYPE}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rows = d["probes"]["nvidia_smi_query"]["stdout"].strip().splitlines()
torch = json.loads(d["probes"]["torch_runtime"]["stdout"])
expected = "A40" if sys.argv[2] == "a40" else "V100"
if len(rows) != 1 or expected not in rows[0] or torch["device_count"] != 1 or d["packages"]["vllm"] != "0.11.2":
    raise SystemExit(f"Expected one visible {expected} and vLLM 0.11.2")
PY
python -m pip freeze > "results/raw/${RUN_ID}.${CLIENT_GPU_TYPE}.environment.txt"
if ! curl -fsS "${A100_URL}/v1/models" > "results/raw/${RUN_ID}.a100.models.json"; then
  echo "A40 node cannot reach the A100 vLLM endpoint" >&2; exit 1
fi
python - <<'PY' > "${server_config}"
import json, os
print(json.dumps({"model_id": os.environ["MODEL_ID"],
    "model_revision": os.environ["MODEL_REVISION"],
    "tokenizer_revision": os.environ["TOKENIZER_REVISION"],
    "served_model_name": "heteroserve-model", "dtype": "float16",
    "max_model_len": 2048, "gpu_memory_utilization": 0.5,
    "prefix_caching": False, "generation_config": "vllm",
    "tensor_parallel_size_per_server": 1, "ports": [8000, 8001],
    "deployment": "two_independent_heterogeneous_pools",
    "a100_url": os.environ["A100_URL"],
    "primary_gpu_type": os.environ["CLIENT_GPU_TYPE"]}, sort_keys=True, indent=2))
PY
export RUN_ID A100_URL CLIENT_GPU_TYPE
python - <<'PY'
import json, os
from pathlib import Path
run = os.environ["RUN_ID"]
pools = [{"name":"gpu0", "url":"http://127.0.0.1:8000", "route_kind":"colocated", "deployed":True},
         {"name":"gpu1", "url":os.environ["A100_URL"], "route_kind":"colocated", "deployed":True}]
arms=[]
for offset, mode in enumerate(("round_robin", "threshold", "least_active")):
    config={"mode":mode, "model":"heteroserve-model", "pools":pools,
            "max_inflight_per_pool":int(os.getenv("CONCURRENCY","8")),
            "health_interval_s":1, "timeout_s":120}
    if mode == "threshold":
        config.update({"short_pool":"gpu0", "long_pool":"gpu1", "threshold":128})
    path=Path(f"results/raw/{run}.{mode}.router_config.json")
    path.write_text(json.dumps(config, sort_keys=True, indent=2)+"\n")
    arms.append({"name":mode,"url":f"http://127.0.0.1:{8100+offset}","router_config":str(path)})
Path(f"results/raw/{run}.arms.json").write_text(json.dumps({"arms":arms},sort_keys=True,indent=2)+"\n")
PY
server_pid=""; telemetry_pid=""; router_pids=()
cleanup() {
  for pid in "${router_pids[@]}" "${server_pid}" "${telemetry_pid}"; do
    if [[ -n "${pid}" ]]; then kill "${pid}" 2>/dev/null || true; fi
  done
  for pid in "${router_pids[@]}" "${server_pid}" "${telemetry_pid}"; do
    if [[ -n "${pid}" ]]; then wait "${pid}" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
trap 'exit 143' TERM INT
HOST=127.0.0.1 PORT=8000 TENSOR_PARALLEL_SIZE=1 bash scripts/serve_vllm.sh \
  --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.5 \
  --no-enable-prefix-caching > "results/raw/${RUN_ID}.${CLIENT_GPU_TYPE}.server.log" 2>&1 &
server_pid=$!
ready=0
for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then break; fi
  sleep 2
done
if [[ "${ready}" != 1 ]]; then tail -n 80 "results/raw/${RUN_ID}.${CLIENT_GPU_TYPE}.server.log" >&2; exit 1; fi
for offset in 0 1 2; do
  case "${offset}" in 0) mode=round_robin;; 1) mode=threshold;; 2) mode=least_active;; esac
  port=$((8100+offset))
  PYTHONPATH=src python -m heteroserve.router \
    --config "results/raw/${RUN_ID}.${mode}.router_config.json" --port "${port}" \
    > "results/raw/${RUN_ID}.${mode}.router.log" 2>&1 &
  router_pids+=("$!")
done
for offset in 0 1 2; do
  port=$((8100+offset)); ready=0
  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:${port}/health" 2>/dev/null | python -c \
       'import json,sys; p=json.load(sys.stdin)["pools"]; sys.exit(not (p.get("gpu0") and p.get("gpu1")))' \
       2>/dev/null; then ready=1; break; fi
    if ! kill -0 "${router_pids[$offset]}" 2>/dev/null; then break; fi
    sleep 1
  done
  if [[ "${ready}" != 1 ]]; then tail -n 80 "results/raw/${RUN_ID}.${mode}.router.log" >&2; exit 1; fi
done
nvidia-smi --query-gpu=timestamp,index,name,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv --loop-ms=1000 > "results/raw/${RUN_ID}.${CLIENT_GPU_TYPE}.telemetry.csv" 2>&1 &
telemetry_pid=$!
PYTHONPATH=src python -m heteroserve.multiarm run \
  --trace "${TRACE_PATH}" --warmup-trace "${WARMUP_TRACE_PATH}" \
  --capabilities "${cap}" --secondary-capabilities "${A100_CAPABILITY}" \
  --server-config "${server_config}" --arms-file "${arms_file}" \
  --output-dir "${run_dir}" --run-id "${RUN_ID}" --model-id "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" --prompt-token-id "${PROMPT_TOKEN_ID}" \
  --concurrency "${CONCURRENCY:-8}" --repetitions "${REPETITIONS:-3}" \
  --slo-ttft-s "${SLO_TTFT_S}" --slo-tpot-s "${SLO_TPOT_S}" --slo-e2e-s "${SLO_E2E_S}"
PYTHONPATH=src python -m heteroserve.multiarm analyze \
  --index "${run_dir}/multiarm_index.json" \
  --output "results/processed/${RUN_ID}.multiarm_analysis.json"
