#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ID:?Set MODEL_ID to an accessible Hugging Face model}"
: "${MODEL_REVISION:?Set MODEL_REVISION to an exact Hugging Face commit}"
: "${TOKENIZER_REVISION:?Set TOKENIZER_REVISION to an exact Hugging Face commit}"
: "${TENSOR_PARALLEL_SIZE:?Set TENSOR_PARALLEL_SIZE to the GPU count in this allocation}"

if [[ ! "${TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TENSOR_PARALLEL_SIZE must be a positive integer" >&2
  exit 2
fi
python - <<'PY'
import importlib.metadata
version = importlib.metadata.version("vllm")
if version != "0.11.2":
    raise SystemExit(f"Expected vllm==0.11.2, found {version}")
PY

exec vllm serve "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --tokenizer-revision "${TOKENIZER_REVISION}" \
  --served-model-name heteroserve-model \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --generation-config vllm \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  "$@"
