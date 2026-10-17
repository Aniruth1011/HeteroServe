# ADR-0002: Two A40s as independent colocated serving pools

Status: provisional until a regular two-A40 allocation completes the live run.

## Context

One A40 has served the pinned Qwen model with vLLM 0.11.2. The operator needs
a measured two-GPU result urgently. No same-node two-A40 allocation or pinned
KV connector deployment has yet been observed. The external HeteroServe router
already forwards streaming Completions to deployed vLLM endpoints.

## Decision

Use one vLLM server per allocated A40 and compare round-robin, prompt-length
threshold and least-active routing on a byte-identical trace. Keep all arms on
the same two servers, rotate trial order, run at least three repetitions, and
record per-request pool attribution and output checksums. Gate final evidence
on actual two-A40 Slurm, `nvidia-smi`, and PyTorch observations.

## Alternatives and trade-offs

- Tensor parallelism would exercise two GPUs in one server, but measures a
  different scaling mechanism and does not compare HeteroServe routing.
- Static prefill/decode disaggregation requires installed vLLM 0.11.2 KV
  connector inspection, matching endpoint setup and measured transfer. A
  documented connector API alone is insufficient for a live claim.
- Independent pools are quickly deployable but use no cross-GPU KV transfer
  and cannot establish a split serving benefit. They may show no policy gain
  for the small pinned model.

## Revisit when

A regular two-GPU run proves the topology and a pinned connector has been
inspected and deployed on those GPUs. Then add a separately labeled static
split baseline and measured transfer metrics. Do not relabel the two-pool run
as disaggregated after the fact.
