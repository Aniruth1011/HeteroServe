from __future__ import annotations

import argparse
from pathlib import Path

from .trace import TraceConfig, generate, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(prog="heteroserve")
    subparsers = parser.add_subparsers(dest="command", required=True)
    trace = subparsers.add_parser("generate-trace")
    trace.add_argument("--output", type=Path, required=True)
    trace.add_argument("--requests", type=int, default=100)
    trace.add_argument("--seed", type=int, default=17)
    trace.add_argument("--arrival-rate", type=float, default=2.0)
    trace.add_argument("--arrival-pattern", choices=["poisson", "bursty"], default="poisson")
    trace.add_argument("--burst-factor", type=float, default=4.0)
    trace.add_argument("--burst-period-s", type=float, default=10.0)
    trace.add_argument("--shape", action="append", default=[], metavar="INPUT:OUTPUT[:WEIGHT]",
                       help="fixed token counts; repeat for a weighted mix")
    args = parser.parse_args()
    if args.command == "generate-trace":
        try:
            shapes = []
            for value in args.shape:
                parts = value.split(":")
                if len(parts) not in {2, 3}:
                    raise ValueError("shape must be INPUT:OUTPUT[:WEIGHT]")
                shapes.append((int(parts[0]), int(parts[1]), float(parts[2]) if len(parts) == 3 else 1.0))
            config = TraceConfig(args.requests, args.seed, args.arrival_rate, shapes=tuple(shapes),
                                 arrival_pattern=args.arrival_pattern, burst_factor=args.burst_factor,
                                 burst_period_s=args.burst_period_s)
        except ValueError as exc:
            parser.error(str(exc))
        write_jsonl(generate(config), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
