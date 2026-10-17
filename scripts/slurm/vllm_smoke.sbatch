#!/usr/bin/env bash
#SBATCH --job-name=heteroserve-vllm-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@60
set -euo pipefail

: "${SLURM_JOB_ID:?Run this script inside a Slurm allocation}"
: "${VENV_PATH:?Set VENV_PATH to the installed GPU virtual environment}"
: "${TRACE_PATH:?Set TRACE_PATH to a generated JSONL trace}"
: "${PROMPT_TOKEN_ID:?Set PROMPT_TOKEN_ID to a verified ordinary tokenizer ID}"
: "${MODEL_ID:?Set MODEL_ID}"
: "${MODEL_REVISION:?Set MODEL_REVISION to a full commit hash}"
: "${TOKENIZER_REVISION:?Set TOKENIZER_REVISION to a full commit hash}"
: "${TENSOR_PARALLEL_SIZE:?Set TENSOR_PARALLEL_SIZE to allocated GPUs on this node}"
if [[ "${SLURM_JOB_PARTITION:-}" != *debug* && "${SLURM_JOB_QOS:-}" != *debug* ]]; then
  echo "This diagnostic script requires a verified debug partition or QoS" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
source "${VENV_PATH}/bin/activate"
mkdir -p results/raw
run_id="gpu_debug_vllm_${SLURM_JOB_ID}"
python scripts/probes/collect_capabilities.py \
  --output "results/raw/${run_id}.capabilities.json"
python -m pip freeze > "results/raw/${run_id}.environment.txt"

server_log="results/raw/${run_id}.server.log"
bash scripts/serve_vllm.sh > "${server_log}" 2>&1 &
server_pid=$!
cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

port="${PORT:-8000}"
ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "vLLM exited before readiness; inspect ${server_log}" >&2
    exit 1
  fi
  if curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" != 1 ]]; then
  echo "vLLM did not become ready within six minutes; inspect ${server_log}" >&2
  exit 1
fi

PYTHONPATH=src python -m heteroserve.replay \
  --trace "${TRACE_PATH}" \
  --output "results/raw/${run_id}.jsonl" \
  --run-id "${run_id}" \
  --model-revision "${MODEL_REVISION}" \
  --tokenizer-revision "${TOKENIZER_REVISION}" \
  --prompt-token-id "${PROMPT_TOKEN_ID}" \
  --concurrency "${CONCURRENCY:-2}" \
  --timeout-s "${REQUEST_TIMEOUT_S:-120}" \
  --slo-ttft-s "${SLO_TTFT_S:-2}" \
  --slo-tpot-s "${SLO_TPOT_S:-0.2}" \
  --evidence-label gpu_debug
