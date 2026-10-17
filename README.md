# HeteroServe

**Heterogeneous LLM inference experiments with vLLM, PyTorch, CUDA, FastAPI, and Slurm.**

HeteroServe provides a version-pinned vLLM serving setup, an HTTP request router,
a trace replayer, and analysis tools for comparing inference behavior across GPU
pools. Experiments on USC Discovery include an NVIDIA A40 colocated baseline, a
paired proxy-overhead control, and routing across separate V100 and A100 serving
pools. A separate NIXL-based runner implements prefill on one GPU and decode on
another, with an actual KV-transfer check before it records measurements.

## Measured results

All values below come from regular Slurm GPU allocations using vLLM **0.11.2**
and `Qwen/Qwen2.5-0.5B-Instruct` at model/tokenizer revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Times are measured at the
client. The A40 baseline and the mixed-length routing studies use different
traces and loads; their absolute latencies should not be compared as a speedup.

### Single A40 baseline

Five measured trials of 100 requests each on one A40 completed **500/500
requests** successfully. Across trial summaries, median **p95 time to first
token (TTFT) was 9.17 ms** and median **p95 end-to-end latency was 221.21 ms**.
Generated-token throughput was 143.81 tokens/s. This was a short-prompt,
low-arrival-rate screening workload, with 64 generated tokens per request.
The [processed analysis](results/processed/a40_colocated_13051353_20261013_113852.analysis.json)
and [raw run index](results/raw/a40_colocated_13051353_20261013_113852/evaluation.json)
identify run `a40_colocated_13051353_20261013_113852` and its five trial files.

### One A40: direct vLLM versus HTTP proxy

Five alternating pairs replayed the same 100-request mixed-length trace through
direct vLLM and a single-pool proxy: **1,000 measured requests total**. All
requests succeeded. The proxy added a median paired **7.10 ms to p95 TTFT**
(95% run-level bootstrap interval: +1.72 to +8.64 ms). The paired p95
end-to-end difference was +0.40 ms, with an interval spanning zero. This is a
measurement of proxy overhead, not a serving speedup. The
[paired report](docs/analysis/a40_paired_proxy_13051353_20261013_121038.md)
records the configuration, all reported deltas, and correctness observations.
The processed report was supplied from Discovery; its complete raw bundle has
not been independently recomputed in this checkout.

### Cross-node V100 and A100 pool routing

A V100 and an A100 each ran an independent colocated vLLM server. Three policies
were tested for three matched 100-request repetitions each: **900/900 measured
requests succeeded**. Medians across trial summaries are:

| Routing policy | p95 TTFT | p95 end-to-end | Generated-token throughput |
| --- | ---: | ---: | ---: |
| Round robin | 92.04 ms | 550.94 ms | 1,760.17 tokens/s |
| Prompt-length threshold | 83.63 ms | 504.94 ms | 1,799.64 tokens/s |
| Least active | 64.46 ms | 528.61 ms | 1,826.17 tokens/s |

Relative to matched round-robin trials, least-active routing reduced p95 TTFT
by a median paired **26.65 ms** and increased generated-token throughput by
**66.00 tokens/s**. The threshold policy had the lowest p95 end-to-end
latency. These are independent-pool routing results: requests complete on one
GPU, without cross-GPU KV transfer. The
[heterogeneous routing report](docs/analysis/hetero_v100_a100_20261013_132610.md)
contains the run IDs, paired intervals, route counts, and measurement limits.
Its processed JSON was supplied from Discovery; the raw bundle is retained
there and has not been independently recomputed in this checkout.

## System components

- **vLLM serving:** [server launcher](scripts/serve_vllm.sh) pins the vLLM
  release, model/tokenizer commit, and OpenAI-compatible endpoint.
- **Routing:** [router](src/heteroserve/router.py) supports static, round-robin,
  least-active, prompt-threshold, adaptive, and DQA policy modes, with health
  checks and admission limits. The measured cross-node study above evaluated
  round robin, threshold, and least active.
- **Workloads and metrics:** [trace generation](src/heteroserve/text_trace.py),
  [request replay](src/heteroserve/replay.py), [paired evaluation](src/heteroserve/paired.py),
  and [multi-policy analysis](src/heteroserve/multiarm.py) persist raw request
  JSONL, configuration hashes, hardware evidence, trial summaries, and paired
  uncertainty estimates.
- **Prefill/decode deployment:** [split runner](scripts/slurm/split_pd.sh)
  configures vLLM 0.11.2's `NixlConnector` producer and consumer on separate
  allocated GPUs. It checks response parity and positive NIXL transfer bytes
  before warmup and repeated trials, then saves server-stage metric deltas.
  The [runbook](docs/architecture/real_split_runbook.md) gives the Slurm and
  operator commands.

## Reproduce and inspect

On a machine with the project source, run the CPU checks with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/unit -q
```

GPU serving and benchmarks run inside regular Discovery Slurm allocations. The
[cluster capability record](docs/architecture/cluster_capabilities.md) documents
observed devices and environment details. Use the
[A40 evaluation script](scripts/slurm/final_colocated.sbatch),
[paired proxy script](scripts/slurm/final_paired_proxy.sbatch), or
[heterogeneous routing scripts](scripts/slurm/final_hetero_v100_client.sh)
with the pinned environment and trace described in the corresponding report.
The [process log](PROCESS_LOG.md) records design decisions, commands, run IDs,
failures, and provenance.

The current streaming client measures TTFT, end-to-end latency, admission wait,
output tokens, success, throughput, and goodput against the declared end-to-end
SLO. HTTP text chunks do not establish individual token timestamps, so exact
per-token TPOT is excluded from these results. Server prefill/decode histograms
and NIXL transfer counters are handled by the split runner; the routing runs
above do not contain KV-transfer measurements. Some
matched requests produced different text despite equal generated-token counts;
the linked reports disclose those counts.
