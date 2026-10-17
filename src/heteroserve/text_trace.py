"""Build matched token-ID traces from a user-provided UTF-8 prompt corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .trace import TraceConfig, generate, write_jsonl
from .schema import RequestSpec


def build_text_trace(prompts: list[str], tokenizer, config: TraceConfig,
                     max_output_tokens: int, max_model_len: int = 2048,
                     prompt_repeats: tuple[int, ...] = (1,),
                     output_token_limits: tuple[int, ...] | None = None) -> list[RequestSpec]:
    limits = output_token_limits or (max_output_tokens,)
    if not prompts or max_output_tokens < 1 or not prompt_repeats or not limits or any(
        value < 1 for value in (*prompt_repeats, *limits)
    ):
        raise ValueError("nonempty prompts and positive prompt/output lengths required")
    template = generate(config)
    encoded = [tokenizer.encode(prompt if repeat == 1 else (prompt + "\n") * repeat,
                                add_special_tokens=False)
               for prompt in prompts for repeat in prompt_repeats]
    if any(not ids for ids in encoded):
        raise ValueError("each prompt must encode to at least one token")
    if any(len(ids) + max(limits) > max_model_len for ids in encoded):
        raise ValueError("a prompt plus requested output exceeds max_model_len")
    return [RequestSpec(item.request_id, item.arrival_s, len(encoded[i % len(encoded)]),
                        limits[(i // len(prompt_repeats)) % len(limits)],
                        item.seed, encoded[i % len(encoded)])
            for i, item in enumerate(template)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="one prompt per line")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="full immutable tokenizer commit")
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--arrival-rate", type=float, required=True)
    parser.add_argument("--arrival-pattern", choices=["poisson", "bursty"], default="poisson")
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--prompt-repeats", default="1",
                        help="comma-separated prompt repetition factors for a length sweep")
    parser.add_argument("--output-token-limits", default="",
                        help="comma-separated output caps; default uses --max-output-tokens")
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                               trust_remote_code=False)
    prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    config = TraceConfig(requests=args.requests, seed=args.seed,
                         arrival_rate=args.arrival_rate, arrival_pattern=args.arrival_pattern)
    try:
        repeats = tuple(int(value) for value in args.prompt_repeats.split(","))
        limits = (tuple(int(value) for value in args.output_token_limits.split(","))
                  if args.output_token_limits else None)
    except ValueError as exc:
        parser.error(f"invalid length sweep: {exc}")
    trace = build_text_trace(prompts, tokenizer, config, args.max_output_tokens,
                             args.max_model_len, repeats, limits)
    write_jsonl(trace, args.output)
    metadata = {"model": args.model, "tokenizer_revision": args.revision,
                "source_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
                "trace_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "requests": len(trace), "max_model_len": args.max_model_len,
                "prompt_repeats": repeats,
                "output_token_limits": limits or (args.max_output_tokens,),
                "prompt_tokens_min": min(r.prompt_tokens for r in trace),
                "prompt_tokens_max": max(r.prompt_tokens for r in trace),
                "output_tokens_requested_min": min(r.max_output_tokens for r in trace),
                "output_tokens_requested_max": max(r.max_output_tokens for r in trace)}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
