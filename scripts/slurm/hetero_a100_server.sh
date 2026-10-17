#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run in a regular A100 Slurm allocation}"
: "${HETEROSERVE_ROOT:?Set project root}"
: "${VENV_PATH:?Set virtual environment path}"
: "${MODEL_ID:?Set model ID}"
: "${MODEL_REVISION:?Set model commit}"
: "${TOKENIZER_REVISION:?Set tokenizer commit}"
if [[ "${SLURM_GPUS_ON_NODE:-}" != 1 || "${SLURM_JOB_PARTITION:-}" == debug || "${SLURM_JOB_QOS:-}" == debug ]]; then
  echo "Requires one regular allocated A100" >&2; exit 2
fi
cd "${HETEROSERVE_ROOT}"
source "${VENV_PATH}/bin/activate"
mkdir -p results/raw/capabilities
cap="results/raw/capabilities/real_gpu_final_${SLURM_JOB_ID}_a100.json"
python scripts/probes/collect_capabilities.py --output "${cap}" --evidence-label real_gpu_final
python - "${cap}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rows = d["probes"]["nvidia_smi_query"]["stdout"].strip().splitlines()
torch = json.loads(d["probes"]["torch_runtime"]["stdout"])
if len(rows) != 1 or "NVIDIA A100" not in rows[0] or torch["device_count"] != 1 or d["packages"]["vllm"] != "0.11.2":
    raise SystemExit("Expected one visible A100 and vLLM 0.11.2")
PY
python -m pip freeze > "results/raw/a100_${SLURM_JOB_ID}.environment.txt"
HOST="$(ip -4 -o addr show dev eth0 | awk '{split($4,a,"/"); print a[1]; exit}')"
if [[ -z "${HOST}" ]]; then echo "No eth0 IPv4 address" >&2; exit 2; fi
export HOST PORT=8001 TENSOR_PARALLEL_SIZE=1
echo "A100_JOB_ID=${SLURM_JOB_ID}"
echo "A100_URL=http://${HOST}:8001"
echo "A100_CAPABILITY=${cap}"
nvidia-smi --query-gpu=timestamp,index,name,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv --loop-ms=1000 > "results/raw/a100_${SLURM_JOB_ID}.telemetry.csv" 2>&1 &
telemetry_pid=$!
cleanup() { kill "${telemetry_pid}" 2>/dev/null || true; wait "${telemetry_pid}" 2>/dev/null || true; }
trap cleanup EXIT
bash scripts/serve_vllm.sh --dtype float16 --max-model-len 2048 \
  --gpu-memory-utilization 0.5 --no-enable-prefix-caching \
  > "results/raw/a100_${SLURM_JOB_ID}.server.log" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true; cleanup' EXIT
for _ in $(seq 1 180); do
  if curl -fsS "http://${HOST}:8001/v1/models" >/dev/null 2>&1; then
    echo "A100_READY=http://${HOST}:8001"
    wait "${server_pid}"
    exit $?
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then break; fi
  sleep 2
done
tail -n 80 "results/raw/a100_${SLURM_JOB_ID}.server.log" >&2
exit 1
