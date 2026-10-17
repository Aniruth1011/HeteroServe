"""Plot per-trial measurements and bootstrap median intervals from checked analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(report_path: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(report_path.read_text())
    if report["evidence_label"] != "real_gpu_final":
        raise ValueError("plot requires a real_gpu_final analysis")
    names = ("p95_ttft_s", "p95_e2e_s", "token_throughput_per_s",
             "e2e_goodput_tokens_per_s")
    labels = ("P95 TTFT (s)", "P95 E2E (s)", "Output tokens/s", "E2E goodput tokens/s")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    paired = report.get("comparison") == "direct_vllm_vs_single_pool_proxy_overhead"
    multiarm = report.get("comparison") in {
        "two_gpu_colocated_pool_routing", "heterogeneous_cross_node_pool_routing"}
    for axis, name, label in zip(axes.flat, names, labels):
        item = None if multiarm else report["metrics"].get(name)
        if item is None:
            if multiarm:
                for arm, metrics in report["metrics"].items():
                    values = metrics[name]["trial_values"]
                    axis.plot(range(1, len(values) + 1), values, "o-", label=arm)
                axis.set_xlim(.5, report["repetitions"] + .5)
                axis.set_xticks(range(1, report["repetitions"] + 1))
            else:
                axis.text(.5, .5, "Unavailable", ha="center", va="center", transform=axis.transAxes)
        elif paired:
            direct = item["direct_trial_values"]
            proxy = item["proxy_trial_values"]
            axis.plot(range(1, len(direct) + 1), direct, "o-", label="Direct vLLM")
            axis.plot(range(1, len(proxy) + 1), proxy, "s-", label="One-pool proxy")
            axis.set_title(f"paired Δ {item['median_delta']:.4g}; 95% CI "
                           f"[{item['ci95_low']:.4g}, {item['ci95_high']:.4g}]")
            axis.set_xlim(.5, len(direct) + .5)
            axis.set_xticks(range(1, len(direct) + 1))
        else:
            values = item["trial_values"]
            axis.plot(range(1, len(values) + 1), values, "o", label="Trial")
            axis.axhline(item["median"], color="tab:orange", label="Median")
            low, high = item["bootstrap_median_ci95"]
            axis.axhspan(low, high, color="tab:orange", alpha=.2, label="95% CI")
            axis.set_xlim(.5, len(values) + .5)
            axis.set_xticks(range(1, len(values) + 1))
        axis.set_xlabel("Measured repetition" if multiarm else
                        "Measured pair" if paired else "Measured trial")
        axis.set_ylabel(label)
    title = ("two-GPU colocated pool policies" if multiarm else
             "direct versus one-pool proxy" if paired else "observed colocated baseline")
    fig.suptitle(f"{report['run_id']} — {title}: {report['gpu']}")
    axes.flat[0].legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
