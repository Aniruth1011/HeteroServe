# Real A100 prefill → V100 decode run (vLLM 0.11.2)

## Recommended: packaged runner

From the **Mac** checkout, sync the small runner and its two Python modules:

```bash
cd /Users/aniruthsundararajan/Downloads/HeteroServe
rsync -av --relative scripts/slurm/split_pd.sh \
  src/heteroserve/replay.py src/heteroserve/split_metrics.py \
  aniruths@discovery.usc.edu:/home1/aniruths/Infra/HeteroServe/
```

On Discovery, prepare the shared environment once:

```bash
cd /home1/aniruths/Infra/HeteroServe
bash scripts/slurm/split_pd.sh prepare
```

Inside a **regular A100 allocation**, run:

```bash
cd /home1/aniruths/Infra/HeteroServe
bash scripts/slurm/split_pd.sh prefill
```

It prints `A100_JOB_ID=...` when the server is ready. Leave it running.
Inside a **regular V100 allocation**, run, replacing the number:

```bash
cd /home1/aniruths/Infra/HeteroServe
bash scripts/slurm/split_pd.sh decode A100_JOB_ID
```

The decode command starts the V100 server and pinned official proxy, checks
output parity and actual NIXL bytes, then performs warmup and three measured
trials. On success it prints `SPLIT_DONE` and writes
`results/processed/<RUN_ID>.split_analysis.json`, plus raw requests, manifests,
stage metrics, GPU telemetry and logs. On any failed smoke or transfer check
it stops before reporting final metrics. `REPETITIONS`, `CONCURRENCY`,
`TRACE_PATH` and `WARMUP_TRACE_PATH` can be set in the V100 environment.

The manual commands below show every operation performed by the packaged
runner and can be used for diagnosis.

This uses two **regular GPU compute allocations**, one A100 and one V100,
the pinned vLLM 0.11.2 `NixlConnector`, and vLLM's official v0.11.2 toy
split proxy. It is a new experiment. The prior independent-pool results are
not split results. Run all serving and replay commands on allocated compute
nodes, never on a login node.

The pinned [NIXL usage guide](https://docs.vllm.ai/en/v0.11.2/features/nixl_connector_usage/)
documents producer/consumer roles and the cross-node side-channel host. Its
[connector requirements](https://raw.githubusercontent.com/vllm-project/vllm/v0.11.2/requirements/kv_connectors.txt)
require `nixl>=0.6.0`. The
[official v0.11.2 proxy](https://raw.githubusercontent.com/vllm-project/vllm/v0.11.2/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py)
passes returned KV transfer parameters from prefill to decode.

## 0. Prepare the shared environment

On Discovery, before starting the GPU servers:

```bash
cd /home1/aniruths/Infra/HeteroServe
source .venv/bin/activate
python -m pip install 'nixl>=0.6.0'
python -m pip freeze > results/raw/split_nixl_environment.txt
python -c 'import importlib.metadata as m; print("vllm",m.version("vllm"),"nixl",m.version("nixl"))'
mkdir -p scripts/vendor results/raw results/processed
curl -fsSL 'https://raw.githubusercontent.com/vllm-project/vllm/v0.11.2/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py' -o scripts/vendor/vllm_0_11_2_toy_proxy_server.py
sha256sum scripts/vendor/vllm_0_11_2_toy_proxy_server.py > results/raw/split_proxy_source.sha256
```

The installed NIXL version, proxy hash, and vLLM release must be saved with
the run. If installation or import fails, stop before launching servers. The
proxy is an upstream example, not a production deployment.

From the Mac project checkout, sync the two new Python modules to Discovery:

```bash
cd /Users/aniruthsundararajan/Downloads/HeteroServe
rsync -av --relative src/heteroserve/replay.py src/heteroserve/split_metrics.py \
  aniruths@discovery.usc.edu:/home1/aniruths/Infra/HeteroServe/
```

## 1. A100 terminal: prefill producer

Use a **regular allocated A100 compute-node terminal**. If you need a new
allocation, request it from one Discovery login terminal:

```bash
srun --partition=gpu --account=yzhao010_1531 --nodes=1 --ntasks=1 \
  --gpus-per-task=a100:1 --cpus-per-task=4 --mem=32G \
  --time=00:45:00 --pty bash
```

Do not run the serving block below on the login node. An allocation may queue;
verify `SLURM_JOB_ID`, `hostname` and `nvidia-smi` after it starts.

```bash
cd /home1/aniruths/Infra/HeteroServe
source .venv/bin/activate
test "${SLURM_GPUS_ON_NODE:-}" = 1 && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'A100' || exit 2
export MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
export MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
export TOKENIZER_REVISION="$MODEL_REVISION"
export PREFILL_HOST="$(ip -4 -o addr show dev eth0 | awk '{split($4,a,"/");print a[1];exit}')"
export VLLM_NIXL_SIDE_CHANNEL_HOST="$PREFILL_HOST"
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600 UCX_NET_DEVICES=all
export VLLM_KV_CACHE_LAYOUT=HND
export HOST="$PREFILL_HOST" PORT=8100 TENSOR_PARALLEL_SIZE=1
echo "PREFILL_URL=http://${PREFILL_HOST}:8100 job=${SLURM_JOB_ID}"
set -o pipefail
bash scripts/serve_vllm.sh --dtype float16 --max-model-len 2048 --block-size 16 \
  --gpu-memory-utilization 0.5 --no-enable-prefix-caching --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
  2>&1 | tee "results/raw/split_a100_${SLURM_JOB_ID}.server.log"
```

Copy the printed `PREFILL_URL` into the V100 terminal. Leave the A100 server
running. Do not replace `CUDA_VISIBLE_DEVICES`; Slurm has already set it.

## 2. V100 terminal: decode consumer and official proxy

Use a **regular allocated V100 compute-node terminal**. Set `PREFILL_URL` to
the actual address printed by step 1. Request the second allocation from a
second Discovery login terminal if necessary:

```bash
srun --partition=gpu --account=yzhao010_1531 --nodes=1 --ntasks=1 \
  --gpus-per-task=v100:1 --cpus-per-task=4 --mem=32G \
  --time=00:45:00 --pty bash
```

Inside the allocated V100 shell:

```bash
cd /home1/aniruths/Infra/HeteroServe
source .venv/bin/activate
set -o pipefail
test "${SLURM_GPUS_ON_NODE:-}" = 1 && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'V100' || exit 2
export PREFILL_URL=http://REPLACE_WITH_A100_IP:8100
export MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
export MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
export TOKENIZER_REVISION="$MODEL_REVISION"
export DECODE_HOST="$(ip -4 -o addr show dev eth0 | awk '{split($4,a,"/");print a[1];exit}')"
export VLLM_NIXL_SIDE_CHANNEL_HOST="$DECODE_HOST"
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600 UCX_NET_DEVICES=all
export VLLM_KV_CACHE_LAYOUT=HND
export HOST="$DECODE_HOST" PORT=8200 TENSOR_PARALLEL_SIZE=1
curl -fsS "${PREFILL_URL}/v1/models" > results/raw/split_prefill_models.json
bash scripts/serve_vllm.sh --dtype float16 --max-model-len 2048 --block-size 16 \
  --gpu-memory-utilization 0.5 --no-enable-prefix-caching --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
  > "results/raw/split_v100_${SLURM_JOB_ID}.server.log" 2>&1 &
decode_pid=$!
for i in $(seq 1 180); do
  curl -fsS "http://${DECODE_HOST}:8200/v1/models" >/dev/null 2>&1 && break
  kill -0 "$decode_pid" 2>/dev/null || { tail -n 80 "results/raw/split_v100_${SLURM_JOB_ID}.server.log"; exit 1; }
  sleep 2
done
curl -fsS "http://${DECODE_HOST}:8200/v1/models"
export PREFILL_HOST="${PREFILL_URL#http://}"; export PREFILL_HOST="${PREFILL_HOST%:8100}"
python scripts/vendor/vllm_0_11_2_toy_proxy_server.py \
  --host 127.0.0.1 --port 8192 \
  --prefiller-hosts "$PREFILL_HOST" --prefiller-ports 8100 \
  --decoder-hosts "$DECODE_HOST" --decoder-ports 8200 \
  > "results/raw/split_proxy_${SLURM_JOB_ID}.log" 2>&1 &
proxy_pid=$!
for i in $(seq 1 60); do
  curl -fsS http://127.0.0.1:8192/healthcheck && break
  kill -0 "$proxy_pid" 2>/dev/null || { tail -n 80 "results/raw/split_proxy_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 1
done
```

The official proxy has `/healthcheck` but no `/v1/models`; replay uses the
decode server for model validation through `--model-check-url`.

## 3. Smoke test actual KV transfer

On the V100 compute node, take snapshots around a single proxied request:

```bash
curl -fsS "${PREFILL_URL}/metrics" > results/raw/split_prefill_smoke_before.prom
curl -fsS "http://${DECODE_HOST}:8200/metrics" > results/raw/split_decode_smoke_before.prom
curl -fsS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8192/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"heteroserve-model","prompt":"Explain one benefit of caching.","max_tokens":16,"temperature":0,"stream":false}' \
  | tee results/raw/split_smoke_response.txt
sleep 12
curl -fsS "${PREFILL_URL}/metrics" > results/raw/split_prefill_smoke_after.prom
curl -fsS "http://${DECODE_HOST}:8200/metrics" > results/raw/split_decode_smoke_after.prom
PYTHONPATH=src python -m heteroserve.split_metrics \
  --prefill-before results/raw/split_prefill_smoke_before.prom \
  --prefill-after results/raw/split_prefill_smoke_after.prom \
  --decode-before results/raw/split_decode_smoke_before.prom \
  --decode-after results/raw/split_decode_smoke_after.prom \
  | tee results/processed/split_smoke_metrics.json
```

Require HTTP 200 with generated text, `kv_transfer_bytes.count > 0` and
`kv_transfer_bytes.sum > 0` on at least one server, and zero NIXL failed
transfers. If bytes are absent or zero, inspect both server logs and stop: a
200 response alone can reflect decode-side fallback rather than a working
split. The pinned vLLM release documents the NIXL byte/time/failure metrics
in its [connector source](https://docs.vllm.ai/en/v0.11.2/api/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector/).

## 4. Three measured repetitions

After the smoke gate passes, run the saved mixed trace on the V100 node:

```bash
export RUN_ID=split_a100_v100_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)
PYTHONPATH=src python -m heteroserve.replay \
  --trace benchmarks/traces/a40_mixed_warmup.jsonl \
  --base-url http://127.0.0.1:8192 \
  --model-check-url "http://${DECODE_HOST}:8200" \
  --output "results/raw/${RUN_ID}_warmup.jsonl" --run-id "${RUN_ID}_warmup" \
  --model-revision "$MODEL_REVISION" --tokenizer-revision "$TOKENIZER_REVISION" \
  --prompt-token-id 23811 --concurrency 8 --timeout-s 120 \
  --slo-ttft-s 1 --slo-tpot-s 0.1 --slo-e2e-s 5 \
  --evidence-label gpu_dev_regular
for trial in 1 2 3; do
  curl -fsS "${PREFILL_URL}/metrics" > "results/raw/${RUN_ID}_t${trial}_prefill_before.prom"
  curl -fsS "http://${DECODE_HOST}:8200/metrics" > "results/raw/${RUN_ID}_t${trial}_decode_before.prom"
  PYTHONPATH=src python -m heteroserve.replay \
    --trace benchmarks/traces/a40_mixed_measured.jsonl \
    --base-url http://127.0.0.1:8192 \
    --model-check-url "http://${DECODE_HOST}:8200" \
    --output "results/raw/${RUN_ID}_t${trial}.jsonl" --run-id "${RUN_ID}_t${trial}" \
    --model-revision "$MODEL_REVISION" --tokenizer-revision "$TOKENIZER_REVISION" \
    --prompt-token-id 23811 --concurrency 8 --timeout-s 120 \
    --slo-ttft-s 1 --slo-tpot-s 0.1 --slo-e2e-s 5 \
    --evidence-label real_gpu_final --save-output-text
  sleep 12
  curl -fsS "${PREFILL_URL}/metrics" > "results/raw/${RUN_ID}_t${trial}_prefill_after.prom"
  curl -fsS "http://${DECODE_HOST}:8200/metrics" > "results/raw/${RUN_ID}_t${trial}_decode_after.prom"
  PYTHONPATH=src python -m heteroserve.split_metrics \
    --prefill-before "results/raw/${RUN_ID}_t${trial}_prefill_before.prom" \
    --prefill-after "results/raw/${RUN_ID}_t${trial}_prefill_after.prom" \
    --decode-before "results/raw/${RUN_ID}_t${trial}_decode_before.prom" \
    --decode-after "results/raw/${RUN_ID}_t${trial}_decode_after.prom" \
    > "results/processed/${RUN_ID}_t${trial}_stages.json"
done
echo "RUN_ID=$RUN_ID"
```

Each replay manifest records client TTFT/E2E/output tokens/success. The stage
JSONs contain histogram-count and sum deltas, means, NIXL bytes/time and
failure counters where exposed. They are server-level aggregate stage metrics,
not per-request timings. Client text chunks still do not provide exact TPOT;
server `inter_token_latency_seconds` is an aggregate histogram. Do not infer
per-request KV bytes or exact TPOT from it. Check `tpot_eligible`, all success
counts, and transfer counts before reporting the result. These split numbers
must not be compared directly to the prior independent-pool run because jobs,
server mode and connector configuration differ.
