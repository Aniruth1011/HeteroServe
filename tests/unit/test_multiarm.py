import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from heteroserve.multiarm import analyze


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MultiarmAnalysisTests(unittest.TestCase):
    def test_matched_two_gpu_analysis_and_token_parity_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cap = root / "cap.json"
            cap.write_text(json.dumps({"evidence_label": "real_gpu_final",
                "slurm": {"SLURM_JOB_ID": "123", "SLURM_GPUS_ON_NODE": "2"},
                "probes": {"nvidia_smi_query": {"stdout":
                    "0, NVIDIA A40\n1, NVIDIA A40\n"}}}))
            server = root / "server.json"
            server.write_text("{}")
            trace = root / "trace.jsonl"
            trace.write_text("{}\n")
            arms = []
            names = ("round_robin", "threshold")
            for name in names:
                config = root / f"{name}.json"
                config.write_text(json.dumps({"mode": name}))
                arms.append({"name": name, "url": f"http://{name}",
                             "router_config": str(config),
                             "router_config_file_sha256": sha(config),
                             "router_config_sha256": hashlib.sha256(
                                 json.dumps(json.loads(config.read_text()),
                                            sort_keys=True).encode()).hexdigest()})
            repetitions = []
            for number in range(1, 4):
                rep = {"number": number, "order": list(names), "manifests": {}}
                for arm in arms:
                    name = arm["name"]
                    raw = root / f"{number}_{name}.jsonl"
                    output = "hello"
                    raw.write_text(json.dumps({"request_id": "req", "outcome": "success",
                        "output_tokens": 8, "output_text": output, "output_chars": len(output),
                        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                        "router_config_sha256": arm["router_config_sha256"],
                        "pool": "gpu0" if name == "threshold" else "gpu1"}) + "\n")
                    manifest = root / f"{number}_{name}.manifest.json"
                    manifest.write_text(json.dumps({"evidence_label": "real_gpu_final",
                        "trace": str(trace), "trace_sha256": sha(trace),
                        "raw_results": str(raw), "raw_results_sha256": sha(raw),
                        "config": {"base_url": arm["url"], "concurrency": 8},
                        "summary": {"requests": 1, "successes": 1,
                            "success_rate": 1.0,
                            "p95_ttft_s": .1 if name == "round_robin" else .2,
                            "p95_e2e_s": .3 if name == "round_robin" else .4,
                            "token_throughput_per_s": 10.0,
                            "e2e_goodput_tokens_per_s": 10.0},
                        "environment": {"slurm_job_id": "123", "vllm": "0.11.2",
                                        "source_sha256": "source"}}))
                    rep["manifests"][name] = str(manifest)
                repetitions.append(rep)
            index = root / "index.json"
            index.write_text(json.dumps({"evidence_label": "real_gpu_final",
                "run_id": "two_gpu", "comparison": "two_gpu_colocated_pool_routing",
                "capabilities": str(cap), "capabilities_sha256": sha(cap),
                "server_config": str(server), "server_config_sha256": sha(server),
                "trace_sha256": sha(trace), "arms": arms,
                "repetitions": repetitions}))
            report = analyze(index)
            self.assertTrue(report["latency_comparison_eligible"])
            self.assertAlmostEqual(report["metrics"]["threshold"]["p95_ttft_s"]
                                   ["paired_delta_vs_round_robin"]["median_delta"], .1)
            raw = root / "3_threshold.jsonl"
            row = json.loads(raw.read_text())
            row["output_tokens"] = 7
            raw.write_text(json.dumps(row) + "\n")
            manifest = Path(repetitions[2]["manifests"]["threshold"])
            payload = json.loads(manifest.read_text())
            payload["raw_results_sha256"] = sha(raw)
            manifest.write_text(json.dumps(payload))
            self.assertFalse(analyze(index)["latency_comparison_eligible"])
            raw.write_text(raw.read_text() + "corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                analyze(index)


if __name__ == "__main__":
    unittest.main()
