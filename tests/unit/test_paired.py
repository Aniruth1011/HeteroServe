import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from heteroserve.paired import analyze
from heteroserve.text_trace import build_text_trace
from heteroserve.trace import TraceConfig


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PairedTests(unittest.TestCase):
    def test_varied_trace_preserves_lengths_and_limits(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text.encode())

        trace = build_text_trace(["hello"], Tokenizer(), TraceConfig(requests=6), 8,
                                 prompt_repeats=(1, 2), output_token_limits=(8, 16, 32))
        self.assertEqual([r.max_output_tokens for r in trace], [8, 8, 16, 16, 32, 32])
        self.assertEqual([r.prompt_tokens for r in trace], [5, 12, 5, 12, 5, 12])
        with self.assertRaises(ValueError):
            build_text_trace(["hello"], Tokenizer(), TraceConfig(requests=1), 8,
                             max_model_len=20, prompt_repeats=(8,))

    def test_paired_analysis_checks_token_parity_and_raw_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cap = root / "cap.json"
            cap.write_text(json.dumps({"slurm": {"SLURM_JOB_ID": "123"},
                                       "probes": {"nvidia_smi_query": {"stdout": "A40"}}}))
            server = root / "server.json"
            server.write_text("{}")
            router = root / "router.json"
            router.write_text("{}")
            trace = root / "trace.jsonl"
            trace.write_text("{}\n")
            pairs = []
            for number in range(1, 4):
                pair = {"number": number, "order": ["direct", "proxy"]}
                for side in ("direct", "proxy"):
                    run_id = f"pair{number}_{side}"
                    raw = root / f"{run_id}.jsonl"
                    output = "hello"
                    row = {"request_id": "req", "run_id": run_id, "outcome": "success",
                           "output_tokens": 8, "output_text": output,
                           "output_chars": len(output),
                           "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                           "router_config_sha256": hashlib.sha256(b"{}").hexdigest()
                           if side == "proxy" else None}
                    raw.write_text(json.dumps(row) + "\n")
                    manifest = root / f"{run_id}.manifest.json"
                    summary = {"requests": 1, "successes": 1, "success_rate": 1.0,
                               "p95_ttft_s": .1 if side == "direct" else .2,
                               "p95_e2e_s": .3 if side == "direct" else .4,
                               "token_throughput_per_s": 10.0,
                               "e2e_goodput_tokens_per_s": 10.0}
                    manifest.write_text(json.dumps({"evidence_label": "real_gpu_final",
                        "trace": str(trace), "trace_sha256": sha(trace),
                        "raw_results": str(raw), "raw_results_sha256": sha(raw),
                        "config": {"base_url": f"http://{side}", "concurrency": 8},
                        "summary": summary,
                        "environment": {"slurm_job_id": "123", "vllm": "0.11.2",
                                        "source_sha256": "source"}}))
                    pair[side] = str(manifest)
                pairs.append(pair)
            index = root / "index.json"
            index.write_text(json.dumps({"evidence_label": "real_gpu_final",
                "run_id": "paired", "comparison": "proxy_overhead",
                "capabilities": str(cap), "capabilities_sha256": sha(cap),
                "server_config": str(server), "server_config_sha256": sha(server),
                "router_config": str(router), "router_config_file_sha256": sha(router),
                "router_config_sha256": hashlib.sha256(b"{}").hexdigest(),
                "trace_sha256": sha(trace),
                "endpoints": {"direct": "http://direct", "proxy": "http://proxy"},
                "pairs": pairs}))
            report = analyze(index)
            self.assertTrue(report["latency_comparison_eligible"])
            self.assertEqual(report["metrics"]["p95_ttft_s"]["median_delta"], .1)
            raw = root / "pair3_proxy.jsonl"
            row = json.loads(raw.read_text())
            row["output_tokens"] = 7
            raw.write_text(json.dumps(row) + "\n")
            manifest = Path(pairs[2]["proxy"])
            data = json.loads(manifest.read_text())
            data["raw_results_sha256"] = sha(raw)
            manifest.write_text(json.dumps(data))
            self.assertFalse(analyze(index)["latency_comparison_eligible"])
            raw.write_text(raw.read_text() + "corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                analyze(index)


if __name__ == "__main__":
    unittest.main()
