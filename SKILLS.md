skills.md — HeteroServe engineering playbooks
Read AGENTS.md first. These are repository-specific execution playbooks. vLLM is mandatory. After each implementation, experiment, or design decision, update PROCESS_LOG.md with trade-offs, reasoning, actual commands and results.
SKILL 01 — Slurm GPU Capability Audit
Trigger: project start, new GPU partition, changed environment, unexpected NCCL/profiler failure.
Inputs: Slurm allocation, partition/account, project directory.
Procedure:
1. Check sinfo and scheduler policies; request interactive compute-node allocation using approved srun/salloc command.
2. Within allocation capture hostname, nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv (omit unsupported fields), nvidia-smi topo -m if available, python -c 'import torch; print(torch.__version__,torch.cuda.is_available(),torch.cuda.device_count())'.
3. Check whether vLLM import and a tiny supported inference job work. Test only permitted multi-GPU/NCCL combinations.
4. Record GPUs, VRAM, driver, scheduler job ID, memory/time limits, topology, tool availability, writable caches.
5. Write docs/architecture/cluster_capabilities.md; log blocked capabilities and alternatives.
Done when: feature matrix for 1 GPU / 2 GPUs / multi-node and profiling permissions is evidence-backed.
SKILL 02 — vLLM Environment and API Compatibility
Trigger: initial setup, vLLM upgrade, new KV connector, MoE backend integration.
Procedure:
1. Pick a vLLM release/commit that supports selected model and needed connector/backend; inspect actual installed APIs, official examples and release notes.
2. Create isolated Python environment with compatible torch/CUDA wheels; pin dependency versions.
3. Start a small vLLM OpenAI-compatible server or offline engine and test deterministic generation.
4. Save environment manifest: Python, torch, CUDA runtime, driver, vLLM SHA, model revision, dtype, quantization, tokenizer.
5. Do not patch vLLM until external adapter approach is ruled out and trade-off logged.
Done when: reproducible smoke test and exact dependency manifest exist.
SKILL 03 — Fair LLM Serving Benchmark
Trigger: any claim about TTFT, TPOT, goodput, latency, throughput or GPU memory.
Procedure:
1. Define workload trace (arrival timestamps, prompt token lengths, output limits, concurrency, seeds), model, sampling parameters and SLOs before measurement.
2. Run warmup, then collect timestamped per-request records including errors/timeouts and actual generated token count.
3. Match baseline/optimized hardware, software, model, sampling, and traffic. Alternate run order.
4. Repeat 3+ times (5+ for final results where feasible). Compute P50/P95/P99, token throughput, success rate, SLA-goodput, variability/CI.
5. Store results/raw/<project>/<run_id>.jsonl, config YAML, manifest JSON, and script-generated figures.
6. Report both wins and regressions, including measurement overhead and cluster contention.
Done when: a third party can reproduce metric calculations from raw traces.
SKILL 04 — HeteroServe Cost Modeling and Controller
Trigger: profiling prefill/decode, choosing routing policy, testing disaggregation.
Procedure:
1. Profile vLLM prefill and decode over representative lengths and concurrency on each actual GPU type.
2. Measure KV bytes and transfer time only through a supported deployed connector; otherwise use explicitly labeled simulation.
3. Fit interpretable cost model: expected queue wait + prefill + transfer + decode + controller overhead. Include capacity constraints and model uncertainty.
4. Compare tuned colocated, static disaggregated (if deployed), simple threshold and adaptive policy.
5. Add hysteresis, fallback, health checking and admission safety; avoid switching active requests if unsupported.
6. Ablate transfer penalty, queue signal and GPU-specific rates; record policy overhead.
Targets (aspirational): P95 TTFT -15–25%, P95 TPOT -10–20%, SLA-goodput +15–30% vs strong eligible baseline. Do not optimize all three at any cost; state primary objective and trade-offs.
Done when: measured improvements or null findings are documented against fair baselines.
SKILL 05 — KV Transfer and GPU Communication Profiling
Trigger: disaggregation experiment, unexpectedly high TTFT, transfer bottleneck.
Procedure:
1. Verify allocation topology and supported KV connector on pinned vLLM release.
2. Instrument KV size, serialization/staging if any, transfer duration, queueing and synchronization.
3. Sweep context lengths, concurrency and GPU pairing; distinguish PCIe/NVLink/network and CPU-staging paths if measurable.
4. Quantify effective bandwidth and fixed overhead; inspect outliers and OOM behavior.
5. Record connector/version limitations and whether metrics are direct measurements or estimates.
Done when: transfer overhead is attributable, reproducible, and represented in the cost model.
SKILL 08 — GPU Profiling and Bottleneck Analysis
Trigger: any optimization is proposed or benchmark regressions appear.
Procedure:
1. Form hypothesis: launch overhead, memory bandwidth, synchronization, compute occupancy, all-to-all, or queueing.
2. Start with PyTorch profiler/CUDA events and nvidia-smi; use Nsight only if allowed.
3. Separate CPU enqueue time, GPU execution time, network transfer and client-observed latency.
4. Profile small representative shapes first; avoid profiling every production run due to overhead.
5. Compare before/after traces and report whether the hypothesized bottleneck actually changed.
Done when: optimization rationale is supported by profile evidence and a recorded trade-off.
SKILL 09 — Experiment Analysis and Statistical Reporting
Trigger: milestone review, paper-style report, resume claim.
Procedure:
1. Verify baseline fairness and manifests.
2. Exclude only predeclared invalid runs, reporting exclusion reasons.
3. Produce distributions and paired comparisons for P95/P99 and goodput; compute confidence intervals using an appropriate resampling unit (usually independent runs, not correlated requests).
4. Break down by workload regime and GPU topology; identify failure cases and regression scenarios.
5. Publish scripts, raw traces where permitted, tables, and a results summary clearly separating real GPU, replay and simulation.
Done when: every headline number is traceable to a run ID and reproducible script.
SKILL 10 — Process Log and Architecture Decision Records
Trigger: every meaningful work session, task, implementation decision, benchmark, bug or scope change.
Procedure:
1. Read last entries and project state before starting.
2. Append task entry with objective, options, trade-offs and rationale before coding.
3. During work, record commands, changed files, failures, alternatives rejected, and why.
4. After work, record tests, run IDs, actual metrics, interpretation, remaining work and next action.
5. Add architecture decision records under docs/decisions/ADR-####-slug.md for significant choices, linking from PROCESS_LOG.md.
6. If plan changes, write new entry with reason; never silently rewrite history.
Required question checklist:
- What are we trying to prove or improve?
- What could be done instead?
- Why choose this approach now?
- What is the latency/throughput/memory/complexity/portability trade-off?
- What evidence would change our decision?
- What exactly was executed and on which hardware?
- What succeeded, failed or remains unverified?
Done when: an unfamiliar engineer can understand the sequence of work and the reasoning without asking the author.
SKILL 11 — Open Source Readiness
Trigger: releasing repository, proposing upstream contribution, final milestone.
Procedure:
1. Write quickstart, prerequisites, tested hardware/software matrix, Slurm examples and limitations.
2. Include reproducible baseline commands, expected outputs, benchmark schema and result regeneration scripts.
3. Add license, contribution guide, CI and clear separation of upstream code vs local patches.
4. Find a small upstream contribution with maintainers' conventions; do not promise acceptance.
5. Audit secrets, weights, private traces and unsupported performance claims.
Done when: repository can be cloned and independently evaluated by someone with equivalent hardware.
Execution priority
1. Cluster audit and pinned vLLM smoke test.
2. Fair optimized vLLM baseline.
3. KV transfer feasibility and cost model.
4. Adaptive controller and ablations.
5. Statistical analysis, documentation and upstream contribution readiness.
All performance numbers in AGENTS.md are targets, not measured outcomes.
SKILL — USC CARC GPU debug development and regular GPU final evaluation
Trigger: any CARC job request, GPU integration work, vLLM smoke test, or benchmark.
1. Inspect CARC sinfo, scontrol show partition, account/QoS rules, and current CARC documentation. Determine whether debug is a partition or QoS; never assume one form universally.
2. For development, request a short GPU debug allocation with srun/salloc using the site's verified flags. Run nvidia-smi, real vLLM loading, correctness tests, and short integration/profiling smoke tests on the allocated compute node. CPU mocks are optional for logic-only tests.
3. Use minimal GPU count and short wall time, respect debug limits, avoid login-node inference, and release idle jobs. Do not run long performance campaigns in debug allocations.
4. Before final evaluation, validate functionality and baseline parity, pin model/runtime versions, define reproducible workload traces and resource needs, and log the decision.
5. Submit regular GPU sbatch jobs for final throughput, TTFT/TPOT, latency, MoE dispatch and layer metrics, and repeated baseline comparisons. Capture GPU model, job ID, vLLM revision, seeds, raw results and sacct provenance.
6. Label outputs gpu_debug, cpu_test, simulated, or real_gpu_final. Never report debug timings as final benchmark evidence.
7. Maintain PROCESS_LOG.md with what was done, alternatives, WHY the approach was chosen, trade-offs, exact Slurm commands, observations, errors, and next steps.
Command examples (verify site-specific syntax): sinfo -o '%P %a %l %D %G'; squeue -u "$USER"; srun --partition=debug --account=<ACCOUNT> --gres=gpu:1 --time=00:30:00 --pty bash or srun --partition=<GPU_PARTITION> --qos=debug --account=<ACCOUNT> --gres=gpu:1 --time=00:30:00 --pty bash; sbatch --partition=<GPU_PARTITION> --account=<ACCOUNT> --gres=gpu:1 scripts/slurm/final_eval.sbatch; sacct -j <JOB_ID>.
Done when: real GPU development is validated in CARC debug mode and final measurements are reproducible on regular GPU allocations.
