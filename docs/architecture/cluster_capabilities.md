# Cluster Capabilities

## Evidence status

### 2026-10-13 12:43 PDT two-A40 runner prepared, allocation pending

The local checkout now contains `scripts/slurm/final_two_pool.sbatch` and
`heteroserve.multiarm`. They are designed for two independent colocated
vLLM 0.11.2 servers, one per allocated A40, with matched repeated routing
trials. Local CPU/syntax checks pass (34 unit tests, two HTTP tests skipped
because local `httpx` is absent). No two-A40 Slurm allocation, dual-server
startup, route attribution, KV connector, or measured two-GPU result has yet
been supplied. The runner checks actual two-A40 visibility before recording
final evidence. The one-A40 paired direct/proxy report is operator-supplied
and awaits independent raw-bundle audit; see the dated analysis document.

The older entries below describe what was known at their stated times and
must not be read as the current project state.

### 2026-10-13 11:55 PDT regular-A40 screening result and paired-control preparation

Regular GPU job `13051353` on one NVIDIA A40 completed five 100-request
colocated vLLM 0.11.2 trials with the pinned Qwen model revision. Its full
artifact bundle is local under `results/raw/a40_colocated_13051353_20261013_113852/`;
the recomputed analysis matches the saved processed JSON, and raw/config/trace
hashes match. All 500 requests succeeded and generated exactly 64 tokens.
Six request IDs generated differing text across trials despite equal token
counts. The trace has only 12–18 prompt tokens, at most three client requests
were in flight, and each response hit its output cap. These are measured
short-prompt screening results, not a tuned throughput or adaptive result.
Client stream chunks do not establish exact TPOT.

A new one-A40 paired direct-versus-one-pool-proxy control is locally coded and
CPU tested. It has not yet run on Discovery; no proxy overhead result exists.
The verified feature matrix still has only one deployed A40 serving pool.
Two-GPU static split, heterogeneous routing, KV connector/transfer metrics,
and adaptive performance remain unverified and must not be inferred from the
one-GPU control.


As of 2026-10-13 11:28 PDT, the complete job `13046849` diagnostic bundle is local and
audited. A regular-A40 repeated-baseline code path now exists, but no new GPU
job or final measurement has run with it. Local CPU tests pass (29 run, two HTTP
tests skipped because `httpx` is absent). The new external router is not
live-GPU validated. The five former CARC-only source files and three tests were
transferred and reconciled with `schema.py`; the fixed schema still needs to be
synced back to CARC and the full remote suite rerun.

### 2026-10-13 A40 allocation update

In a later job `13046849` on `b09-10`, the operator launched the synced repository vLLM wrapper and `/v1/models` returned `heteroserve-model` with maximum length 2048. A three-request replay produced three successful records, each with 32 prompt and eight generated tokens; exact TPOT was unavailable and the diagnostic p95 TTFT was 0.0327769429422915 s. This is evidence of working single-GPU serving and replay, not a reportable performance result. At this point in the chronology, raw files remained on CARC; they were later transferred and audited.

The operator subsequently reported `vllm --version` as 0.11.2 and a replay manifest with Python 3.11.9, httpx 0.28.1, model/tokenizer commit `7ae557604adf67be50417f59c2c2f167def9a775`, and trace SHA-256 `5c4f0733779a598bbe5bc20df1211d98380acb12e3ccb26d03fbb109c8a1bf72`. The manifest's Git SHA is null. A current-job capability JSON was written at `results/raw/capabilities/gpu_dev_13046849.json`; its contents and the full raw results awaited local review at that time, completed later in the chronology below.

The operator then printed the current-job capability summary: one NVIDIA A40, 46,068 MiB, compute capability 8.6, driver 580.159.04, GPU UUID `GPU-98fb7b79-d058-66bc-5856-a7ffe183f64b`. The isolated venv has Torch 2.9.0+cu128 (CUDA build 12.8), reports CUDA available, and sees one A40. Source file SHA-256 hashes were saved at `results/raw/gpu_dev_vllm_13046849.source_sha256.txt`.

The full bundle was later transferred and audited locally. Manifest trace/config hashes matched, all three raw rows matched trace token counts and summary totals, and the server log showed three completion HTTP 200 responses plus shutdown. Twelve common source files matched remote SHA-256; five remote-only files (`client.py`, `provenance.py`, `runner.py`, `statistics.py`, `workload.py`) remain to be retrieved and reviewed. The server log also recorded a 33.3% prefix-cache hit rate on the synthetic repeated-token trace. These observations establish a functional single-GPU vLLM integration smoke, not latency or throughput performance.

At 09:38 PDT the operator reported a successful direct vLLM `/v1/completions` request from this allocation. The server was started on localhost for `Qwen/Qwen2.5-0.5B-Instruct` at model and tokenizer revision `7ae557604adf67be50417f59c2c2f167def9a775`, tensor parallel size 1, FP16, maximum model length 2048, and GPU memory utilization 0.5. The response returned HTTP 200 with 6 prompt tokens, 16 completion tokens, and generated text. This verifies one end-to-end vLLM inference on the A40. The user did not supply `vllm --version`, the environment freeze, or the server log, so the installed package version and full dependency provenance remain unverified. This was a development smoke on a regular allocation, not a benchmark.

The operator supplied the contents of `results/raw/capabilities/a40_13044598.json` from Discovery Slurm job `13044598` on `b09-10`. The file has not yet been copied into this checkout. The regular `gpu` allocation has one GPU, one node, and four allocated CPUs. The device is an NVIDIA A40 with 46,068 MiB reported VRAM, compute capability 8.6, and driver 580.159.04. Python 3.11.9 and Torch 2.7.1+cu118 see one CUDA device; `vllm` is absent. `ib0` is up; topology reports `NODE` to the visible NIC. No CUDA arithmetic result, model execution, profiling, NCCL, or bandwidth measurement has been supplied. The A40 meets the [vLLM 0.11.2 minimum compute capability of 7.0](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/), so an isolated pinned installation and short serving smoke are the next checks.

The P100 section below remains a record of job `13044148`; it does not describe the current allocation.

### 2026-10-13 one-GPU allocation update

The operator supplied the contents of `results/raw/capabilities/gpu_debug_13044148.json` from Slurm job `13044148` on Discovery node `e21-05`. The artifact has not yet been copied into this checkout. This is a regular `gpu` partition allocation (`normal` QoS), one GPU, one node, eight allocated CPUs. The allocated device is a Tesla P100-PCIE-16GB, 16 GiB, compute capability 6.0, driver 580.159.04. Python 3.11.9 and Torch 2.7.1+cu118 report CUDA available with one P100 device; `vllm` is absent. `ib0` is up, but no inter-node bandwidth or NCCL test has been run. This proves GPU visibility, not model execution or performance.

The [vLLM 0.11.2 NVIDIA requirements](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/) specify compute capability 7.0 or higher. Thus this P100 is ineligible for the project's pinned vLLM GPU smoke test. Do not install `vllm==0.11.2` or run the smoke script on this device as evidence of a supported baseline. Request an eligible GPU allocation and probe it first. The current regular allocation must not produce final vLLM numbers on unsupported hardware.

The older context below describes the earlier CPU-only probe and is retained as historical evidence.

On 2026-10-13 this workspace is a local macOS session with Python 3.14.8.
`sinfo`, `srun`, `nvidia-smi`, and an installed `vllm` are absent. The queued
CARC A40 debug job `12995616` has no result artifact in this checkout. No A40
or A100 allocation, vLLM runtime, model, connector, KV transfer, or network
capability was verified in this session. The current DQA controller uses
explicit, uncalibrated inputs and is CPU logic only.

The [vLLM 0.11.2 connector API index](https://docs.vllm.ai/en/v0.11.2/api/vllm/distributed/kv_transfer/kv_connector/v1/)
lists a NIXL connector, and the [versioned deployment guide](https://docs.vllm.ai/en/v0.11.2/serving/expert_parallel_deployment/)
shows `--kv-transfer-config` for split instances. These documents establish a
candidate integration path, not installed API or CARC compatibility. Inspect the
actual installed 0.11.2 package and example proxy in an allocated GPU job before
implementing or claiming a deployed split route.

This document separates observed allocation facts, published CARC capabilities, and unverified project capabilities. Published hardware availability does not prove that a particular account can allocate it or that HeteroServe runs on it.

## Current observed context

Observed on 2026-10-12 in the environment hosting this work session:

- Cluster: Discovery (`SLURM_CLUSTER_NAME=discovery`)
- Compute node: `d05-41` (`SLURM_JOB_NODELIST=d05-41`)
- Existing allocation: job `12993535`, partition `debug`, QoS `normal`, account `yzhao010_1531`
- Allocation size: one node, one task, one CPU; no GPU allocation variable was present
- Host kernel: Linux 4.18.0-553.126.1.el8_10.x86_64
- Python: 3.11.9 from the CARC Spack installation
- Python package metadata in the unisolated user environment: Torch `2.7.1+cu118` (CUDA build `11.8`, Torch git revision `e2d141dbde55c2a4370fac5165b0561b6af4798b`), Transformers `5.8.1`, Tokenizers `0.22.2`, NumPy `2.4.4`; `vllm` is not installed
- Physical CPU node view: 24 Intel Xeon Silver 4116 cores across two sockets and about 188 GiB RAM. The job allocation remains one CPU; physical resources outside the allocation must not be treated as available.
- `nvidia-smi` is not on `PATH` in this CPU-only allocation. This does not establish driver or GPU availability.
- Slurm clients (`sinfo`, `srun`, `sbatch`, `sacctmgr`) are installed. Controller queries from the tool sandbox failed with `Operation not permitted`; an approved outside-sandbox retry failed in the execution wrapper before `sinfo` executed. This is a tooling limitation, not evidence of a CARC outage.

No GPU, CUDA runtime, driver, VRAM, compute capability, topology, NCCL, vLLM execution, model access, or profiler capability has been verified yet.

The prepared one-A40 debug submission could not reach the Slurm controller from the managed execution sandbox. It was subsequently submitted from a normal CARC shell as job `12995616`.

```bash
squeue -j 12995616 -o '%.18i %.12P %.10T %.10M %.10l %R'
```

Live node evidence showed all four `debug` nodes drained: the A40 node `b11-09`, P100 node `e23-02`, and CPU nodes `d05-41`/`d05-42`. Both GPU nodes reported `Kill task failed` drain reasons. Job `12995616` therefore remains pending with no estimated start. This is a cluster availability blocker, not an invalid request. Do not proceed to vLLM dependency selection until `results/raw/capabilities/gpu_debug_12995616.json` exists and is reviewed.

## Current official CARC guidance

The following is documentation evidence, not a completed allocation test:

- [Discovery Resource Overview](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/resource-overview-discovery) lists `gpu` for GPU jobs and `debug` for short debugging. Its published inventory includes L40S, A100, A40, V100, and P100 GPUs; A40 and P100 are listed in `debug`.
- [Running Jobs](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs) says not to run heavy workloads on login nodes and lists a one-hour limit for `debug` and a 48-hour limit for `gpu`. It documents project-account charging and `myaccount` for account discovery.
- [GPU Programming](https://www.carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming.html) documents `--gpus-per-task=<gpu_type>:<number>`, with A40/P100 in `debug` and regular GPU types in `gpu`. It reports at most two GPUs per listed Discovery node types.

The project must still verify live scheduler state, account eligibility, and actual allocated hardware before using these statements as experiment provenance.

## Feature matrix

| Capability | Status | Evidence / next validation |
|---|---|---|
| CPU logic/unit tests | Partially verified | Local unit suite passed 29 tests, with two HTTP tests skipped because `httpx` is absent locally. Remote full rerun pending after schema sync. |
| Single GPU | Verified for vLLM smoke | Job `13046849` directly reports one A40, compute capability 8.6; vLLM 0.11.2 served three successful replay requests. Earlier P100 job `13044148` was ineligible for this vLLM release. |
| Same-node two GPU | Runner prepared; allocation unverified | Official inventory lists up to two A40s per node. Verify Slurm, PyTorch and `nvidia-smi` see the same two A40s before running the prepared workflow. |
| Multi-node GPU | Unverified | Requires explicit allocation and network/NCCL permission testing; do not design around it yet. |
| Homogeneous GPU experiments | Unverified | Likely possible per published inventory, but not allocated or tested. |
| Heterogeneous co-allocation | Unverified | Published inventory alone does not establish mixed-type co-allocation. Treat as unsupported until observed. |
| vLLM import/inference | One A40 replay verified | vLLM 0.11.2, Torch 2.9.0+cu128, Python 3.11.9; one direct completion and three replay completions succeeded. Generated text is not preserved in replay records, so broader content correctness remains open. |
| Profiling | Unverified | Check PyTorch profiler/CUDA events first in a GPU-debug allocation; Nsight only if permitted. |
| Final performance evaluation | One-A40 screen audited; two-A40 evaluation pending | Job `13051353` produced a five-trial one-A40 screen. Its paired direct/proxy follow-up is operator-reported but its raw bundle awaits independent audit. No measured two-GPU metric exists yet. |

## Immediate next evidence

1. Sync the reconciled `schema.py` back to CARC and rerun the full remote unit suite. New manifests include source SHA-256; a Git revision remains preferred.
2. Transfer the new code, run the regular A40 baseline workflow with tokenized prompts, and review raw output hashes, failures, telemetry, and trial variability. The included prompt corpus is synthetic screening material; representative workload selection remains open.
3. Validate router streaming on one A40 as an integration check. A two-pool comparison requires actual co-allocation, measured pool profiles, and proxy overhead characterization.
4. Validate per-token timing before reporting exact TPOT or TPOT-based goodput; test same-node GPU and KV connector feasibility separately before split claims.

## Probe implementation

The following older bootstrap description is retained as historical context. The active probe is now `scripts/probes/collect_capabilities.py`, and the version-checked server launcher is `scripts/serve_vllm.sh`.

The initial bootstrap files were temporarily at repository root because that workspace wrapper blocked directory creation:

- `phase0_collect_capabilities.py`: standard-library JSON collector; optional command failures and timeouts are data rather than fatal errors.
- `phase0_gpu_capability_probe.sbatch`: ten-minute, one-A40 `debug` request that writes `gpu_debug_<job_id>.json` under `results/raw/capabilities/`.

The CPU-context validation reported no allocated GPU, no `nvidia-smi`, a successful Torch runtime import with CUDA unavailable and zero devices, and successful CPU/memory probes. The active probe and Slurm script are now under `scripts/`.

