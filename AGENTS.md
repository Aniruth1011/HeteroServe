AGENTS.md — HeteroServe
Project mission and non-negotiable rules
Build HeteroServe as an independent, reproducible GPU inference systems project with real vLLM integration mandatory. This is a university Slurm environment; never assume root privileges, persistent allocations, multi-node access, or permission to compile CUDA C++ extensions. Pin dependencies and confirm available vLLM APIs before implementation.
Every meaningful task must update PROCESS_LOG.md with actions, alternatives, trade-offs, why an approach was selected, evidence, blockers and next steps. Never claim simulated results as real GPU results.
Target outcomes and performance metrics
- Demonstrate functioning vLLM-based inference with version-pinned, reproducible setup.
- Measure tuned colocated, static disaggregated (when supported), simple heuristic, and adaptive-policy baselines under the same workload, hardware, concurrency, and accuracy constraints.
- Collect per-request TTFT, TPOT, end-to-end latency, output tokens, admission/wait time, and request success; collect aggregate throughput, goodput, KV transfer bytes/time, and GPU utilization/memory.
- Aspirational targets (not results): P95 TTFT 15–25% lower, P95 TPOT 10–20% lower, and SLA-goodput 15–30% higher against a strong eligible baseline. Optimize a declared primary metric; report regressions honestly.
- If heterogeneous GPUs cannot be co-allocated, test actual available topology and label heterogeneity studies simulation-only. If the deployed vLLM release does not expose required connector hooks, implement a separate orchestration layer and document the limitation.
Completion criteria
A project is complete only when: working code; pinned environment; documented hardware; scripts for baseline and optimized runs; raw machine-readable metrics; repeated trials and uncertainty; correctness checks; plots and a written analysis; failure cases; and a final README with honest results. No invented benchmarks, GPU types, or speedups.
2. Roles / agent responsibilities
Agents are logical work roles, not separate autonomous processes. One contributor may take multiple roles.
- Lead/architect: own scope, architectural decision records, milestone acceptance, interfaces, and priority ordering. Prevent feature creep.
- vLLM runtime engineer: own server lifecycle, version compatibility, request scheduling hooks, KV connector integration, and model serving correctness.
- GPU performance engineer: own Nsight/PyTorch profiling where allowed, GPU telemetry, workload characterization, and memory/communication diagnosis.
- Benchmark/reproducibility engineer: own traffic generator, trace metadata, baseline fairness, statistics, and results packaging.
- Reviewer: independently verify claims, configuration drift, fairness, and process-log completeness before merge.
For every substantial task, designate an owner, reviewer, expected artifact, acceptance test, and rollback plan. Never silently bypass a failed test.
3. Required technology stack
Layer	Preferred technology	Notes
Language	Python 3.11+	Pin to vLLM-supported Python and torch versions, not an arbitrary latest version.
Serving	vLLM (mandatory)	Pin exact git commit/release; inspect APIs in installed version.
Models	Hugging Face Transformers/tokenizers	Choose models whose licenses and architecture are supported by chosen vLLM version.
GPU compute	PyTorch + CUDA runtime	CUDA toolkit not assumed; rely on prebuilt compatible packages first.
Communication	vLLM-supported KV connector; NCCL if supported	Do not invent custom GPU networking; check node topology.
MoE	vLLM supported MoE backend / expert parallel when feasible	Avoid replacing optimized kernels with slow Python in production path.
Profiling	torch.profiler, nvidia-smi, optional Nsight Systems/Compute	Check Slurm and administrator permissions.
Load generation	vLLM benchmark tooling, custom asyncio/httpx harness	Validate timestamps, warmups, concurrency, and tokenizer usage.
Data	JSONL/Parquet/CSV; pandas, numpy	Include run_id, git SHA, config hash, hardware and provenance.
Visualization	matplotlib	Reproducible scripts, no hand-edited charts.
QA	pytest, ruff, mypy optional, GitHub Actions CPU tests	GPU tests gated by availability.
Experiment config	YAML + argparse/typer	Config committed; secrets excluded.
Cluster	Slurm sbatch/srun, module/venv/conda	GPU jobs on compute nodes, never assume login-node GPU access.
Packaging	pyproject.toml, lock/constraints, Docker optional	Containers only if cluster allows Apptainer/Singularity.


Before writing integration code, inspect actual vLLM APIs and examples at the pinned revision. vLLM scheduling, disaggregated prefill, KV connectors, MoE backends and expert-parallel APIs can change significantly between releases.
Repository layout
HeteroServe/
  AGENTS.md
  skills.md
  PROCESS_LOG.md
  README.md
  pyproject.toml
  configs/
  scripts/
  src/heteroserve/
  tests/unit/
  tests/integration/
  tests/gpu/
  benchmarks/traces/
  results/raw/
  results/processed/
  results/figures/
  docs/architecture/
  docs/decisions/
  docs/limitations/
5. Phase 0: cluster and environment reconnaissance (mandatory)
1. Identify scheduler/account/partition constraints with sinfo, scontrol, sacctmgr only if authorized, and srun/sbatch for compute-node probing.
2. Record GPU model, count, VRAM, compute capability, driver, CUDA runtime compatibility, interconnect/topology (if visible), network interface, hostnames, available walltime and memory.
3. Test whether multi-GPU same-node and multi-node allocations are possible; check NCCL communication and actual host-to-host bandwidth with permission.
4. Check installed vLLM, PyTorch, tokenizer, and model access. Use a small supported dense model for initial smoke tests.
5. Test profiling permissions and Triton compilation only as optional diagnostics. No root access or driver changes.
6. Record probe outputs and constraints in docs/architecture/cluster_capabilities.md and PROCESS_LOG.md.
7. Choose a supported feature matrix: single GPU; same-node multi-GPU; multi-node; homogeneous/heterogeneous. Mark unsupported cases explicitly.
Gate: no distributed design or performance promise until capability matrix is verified.
6. HeteroServe: detailed implementation plan
6.1 Research hypothesis
Static prefill/decode disaggregation is not universally optimal. The preferred mode depends on prompt length, generation length, GPU compute rates, queueing, transfer bandwidth, and KV-cache size. A conservative adaptive controller can improve tail latency or SLA-goodput on workloads with changing mixes.
6.2 Design boundaries
- Keep vLLM responsible for model execution, attention/KV management, and model-specific correctness.
- Implement an external admission/router/controller first; modify vLLM internals only when justified and version-locked.
- Avoid claiming per-request colocated/disaggregated switching if the selected vLLM topology only supports server-level routing. Clearly distinguish server-pool routing from intra-request dynamic reconfiguration.
- Do not implement cross-GPU KV transfer by copying tensors naively if vLLM connector integration is available.
- Model KV transfer as bytes / measured_effective_bandwidth + fixed_overhead + queueing; fit with observed data and validate held-out traces.
- Include decision hysteresis, cold-start behavior, fallback on unavailable pools, and routing time overhead.
6.3 Deliverables
1. Baseline server: pinned vLLM version, stable model, serving script, correctness and reproducibility smoke tests.
2. Workload harness: synthetic and realistic distributions (short/long input, short/long output, mixed, bursty, Poisson arrival, concurrency sweeps); preserve input token counts and seeds.
3. Profiling data: GPU-specific prefill/decode response surfaces, memory capacity, queueing, transfer latency where supported.
4. Static baselines: tuned colocated; static split if supported; naive threshold router.
5. Adaptive controller: interpretable cost model; request classification using available pre-execution features; periodic measurements; SLA-aware selection.
6. Failure handling: health checks, fallback policy, retries without double counting, queue limits, timeouts, OOM-safe capacity estimates.
7. Evaluation: paired workloads, repeated runs, confidence intervals, tail metrics, sensitivity to transfer bandwidth, routing overhead and GPU mix.
6.4 HeteroServe baselines and ablations
- B0: tuned colocated vLLM.
- B1: static disaggregated vLLM using supported KV connector, only if actually deployed.
- B2: simple prompt-length threshold routing between available pools.
- P: adaptive policy.
- Ablate: no transfer penalty, no queue awareness, no GPU heterogeneity, no hysteresis, perfect-information oracle (simulation-only), controller overhead.
- If B1 unavailable, report P vs B0/B2 only where technically comparable; do not fabricate B1.
6.5 Measurement definitions
- TTFT: time from request submission to first output token, including queueing; specify client vs server timestamp source.
- TPOT: (last_token_time - first_token_time)/(generated_tokens - 1) for outputs with >=2 tokens; exclude shorter outputs from TPOT summary and report count.
- E2E latency: request submission to completion.
- Throughput: completed generated tokens per wall-clock second, with request rate separately.
- SLA-goodput: successfully completed output tokens/sec for requests satisfying predeclared TTFT/TPOT (or E2E) SLOs; also report fraction meeting SLO.
- KV transfer: actual bytes, elapsed transfer time, transfer bandwidth, failures; measure separate from model computation where possible.
- Utilization: GPU SM activity, memory allocation, power when available; avoid interpreting nvidia-smi utilization as actual compute efficiency.
8. Experimental integrity and reproducibility
- Every comparison must pin model revision, tokenizer, dtype, quantization, sampling settings, max tokens, seeds, vLLM commit, GPU allocation, concurrency, cache settings, and traffic trace.
- Record warmup vs measured runs; avoid including compilation/model-load in steady-state metrics unless cold start is the explicit question.
- Run at least 3 repeated trials for screening and preferably 5+ for final claims; report median, variability and confidence intervals where appropriate.
- Alternate or randomize baseline/policy order to reduce drift from shared-cluster interference; log contention and GPU temperature/clocks if accessible.
- Ensure matched generated token counts or normalize appropriately; compare quality/correctness before latency claims.
- Distinguish microbenchmarks from end-to-end vLLM service results; distinguish real hardware, trace replay, and simulation in filenames, figures, and prose.
- Never select only favorable traces or hide regressions. Report workload regimes where optimization loses.
- Persist raw JSONL results and machine-readable manifests before generating summaries.
9. Testing and CI
- Unit: cost model monotonicity and boundary cases; policy fallback; token packing invertibility; empty-expert and top-k parity; metric formulas.
- Integration: start vLLM server; health and generation checks; concurrency/timeouts; model revision and tokenizer consistency; feature flags.
- GPU: event timing sanity; correctness BF16/FP16; CUDA OOM graceful handling; NCCL if allowed; MoE end-to-end parity.
- CI: CPU unit tests and formatting always; GPU tests only under scheduled Slurm jobs. No fake green GPU checks.
- Regression thresholds must be documented with variability, not arbitrarily chosen after seeing results.
10. Slurm safety and operational rules
- Never run sustained inference on login nodes; request compute resources with scheduler.
- Respect partition quotas, walltimes, job arrays, fair-use policies, and network restrictions.
- Bind each job to its allocation; avoid hardcoded CUDA_VISIBLE_DEVICES that conflict with Slurm.
- Store Hugging Face and Triton caches in permitted writable storage; clean up scratch artifacts according to cluster policy.
- Do not download gated weights without authorization; never commit HF tokens, SSH keys, cluster credentials, or personal data.
- Use graceful SIGTERM checkpoint/flush of metrics and logs; keep resumable experiments.
- Avoid assuming multi-node network access, NCCL permissions, GPU peer-to-peer, or Nsight access.
11. Working sequence and task protocol
For every task:
1. Inspect: read AGENTS.md, skills.md, recent PROCESS_LOG.md, current configs and relevant code.
2. Plan: identify hypothesis, affected components, alternatives, trade-offs, expected measurable outcome, dependencies and risk.
3. Log before coding: create PROCESS_LOG.md entry with status planned and task ID.
4. Implement minimally: feature flags, small diff, pinned APIs, no unrequested refactor.
5. Test: correctness first, then microbench, then end-to-end if hardware available.
6. Log evidence: commands, hardware, run IDs, results, errors, decisions and reasons; attach file paths.
7. Review: compare against acceptance criteria, record regressions and open questions.
8. Close or defer: status done, blocked, or deferred, with exact next action.
If hardware or required API is unavailable, document a blocked task and implement a clearly labeled CPU/mock unit test or simulator only for algorithmic validation. Never substitute simulated performance for GPU evidence.
12. Mandatory PROCESS_LOG.md specification
The process log is a living engineering journal, not a retrospective list of completed commands. Append at every meaningful decision, implementation, benchmark, bug, failure, or changed hypothesis. Do not overwrite history; corrections are new entries referencing the original.
Each entry must contain:
## YYYY-MM-DD HH:MM TZ — [PROJECT][TASK-ID] Short action
- Status: planned | in_progress | blocked | done | reverted
- Objective / hypothesis:
- Current context / observed problem:
- Options considered: A / B / C
- Trade-offs: latency, throughput, memory, complexity, portability, maintenance, cluster feasibility
- Decision and WHY this option was selected:
- What is being done now:
- Implementation details / files changed:
- Commands and environment (GPU, Slurm job ID, vLLM SHA, model revision):
- Validation and observed results (links to raw data):
- What worked / what failed / unexpected findings:
- Remaining work and next step:
- Owner / reviewer:
Additionally maintain near the top:
- Project state: current milestone, verified capabilities, last successful baseline, blocking issues.
- Decision index: links to important architecture decisions.
- Experiment index: run ID → config → raw results → analysis.
- Open questions: uncertainties and planned evidence to resolve them.
When recording alternatives, explain what could be done, why it was not chosen, and what evidence would justify revisiting it. Avoid vague entries such as “optimized performance.” Record exact change and metric. Every PR should reference one or more process-log task IDs.
Initial implementation backlog
- H-01: Build matched request trace generator and latency collector.
- H-02: Profile prefill/decode and GPU-specific cost curves.
- H-03: Validate static split/KV connector feasibility and measure transfer cost.
- H-04: Implement colocated/static/threshold/adaptive policy interfaces.
- H-05: Add admission safety, fallback and instrumentation.
- H-06: Run workload sweep, ablations and final report.
14. Stop conditions and escalation
Stop and document before proceeding if: baseline cannot run reliably; model outputs differ unexpectedly; comparisons use different hardware or model configurations; instrumentation materially changes latency; Slurm rules forbid the required operation; or a result is based only on simulation but being described as measured GPU performance. Resolve or narrow scope rather than hiding limitations.
USC CARC operational policy — GPU debug development and regular GPU final evaluation
Correct policy: On USC CARC, development may use real GPUs via the cluster's debug mode (a Slurm debug partition or QoS, depending on the specific CARC system). Do not confuse this with CPU-only application debug flags. Use short GPU debug allocations for model loading, vLLM integration, correctness tests, profiling smoke tests, and fixing runtime failures. CPU tests and mocks are optional complements, not mandatory prerequisites to GPU development.
Final evaluation: Use regular, non-debug GPU allocations for sustained benchmark runs, repeated measurements, baseline comparisons, and reportable latency/throughput statistics. Never run GPU workloads on login nodes. Debug results are diagnostic and must not be presented as final performance numbers.
CARC discovery and commands
CARC configuration varies by cluster. Before issuing a request, check current USC CARC documentation and available partitions/QoS/accounts; do not assume debug is a partition rather than a QoS. The following commands are templates, not guaranteed site-specific syntax.
sinfo -o '%P %a %l %D %G'
squeue -u "$USER"
scontrol show partition
sacctmgr show assoc user="$USER" format=Account,Partition,QOS  # if permitted

# GPU development — choose the ONE form matching the actual CARC configuration:
srun --partition=debug --account=<ACCOUNT> --gres=gpu:1 --time=00:30:00 --pty bash
# OR, when debug is a QoS rather than a partition:
srun --partition=<GPU_PARTITION> --qos=debug --account=<ACCOUNT> --gres=gpu:1 --time=00:30:00 --pty bash

# Inside the allocated GPU compute node:
nvidia-smi
python -m pytest -m gpu -q
# Launch short vLLM correctness/integration smoke tests with project-specific commands.

# Final evaluation — regular GPU allocation (no debug QoS):
sbatch --partition=<GPU_PARTITION> --account=<ACCOUNT> --gres=gpu:1 --time=02:00:00 scripts/slurm/final_eval.sbatch

# Inspect and manage jobs:
squeue -u "$USER"
scontrol show job <JOB_ID>
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS
scancel <JOB_ID>
Replace placeholders and adapt GPU resource flags, account, time limits, and debug selection to the actual CARC system. Confirm debug limits and interactive job permissions before launching. Prefer the smallest feasible GPU allocation for development; release allocations when idle.
Execution stages and evidence
1. DEVELOP / GPU DEBUG: implement and test using short CARC GPU debug jobs as needed. Real vLLM model loading, GPU correctness, and profiling smoke tests are permitted. CPU mocks can accelerate pure logic tests.
2. PREPARE FINAL EVALUATION: pin dependencies/model revision, validate correctness, define workloads and baselines, size the regular allocation, and record benchmark methodology.
3. FINAL / REGULAR GPU: run tuned stock-vLLM baselines and proposed method under comparable hardware/workloads. Collect TTFT/TPOT/throughput or MoE layer/dispatch metrics, repeated trials, and hardware/software provenance.
4. REPORT: label artifacts gpu_debug, cpu_test, simulated, or real_gpu_final; do not treat debug timings or simulations as final GPU results.
Mandatory process log
For each meaningful task, document the goal, alternatives, chosen implementation and WHY, trade-offs, debug vs regular allocation, exact Slurm command and job ID, GPU type/count, vLLM/model versions, tests, failures, results, limitations, and next action. Explicitly document transition from GPU debug validation to regular GPU final evaluation. No GPU jobs were run by this documentation edit.
