import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from heteroserve.analysis import analyze, bootstrap_ci
from heteroserve.evaluation import validate_final_environment
from heteroserve.metrics import SLO, summarize
from heteroserve.schema import RequestResult, RequestSpec
from heteroserve.text_trace import build_text_trace
from heteroserve.trace import TraceConfig


class FinalWorkflowTests(unittest.TestCase):
    def test_e2e_goodput_works_without_exact_tpot(self):
        results = [RequestResult("ok", 0, .1, 1, 10, True, 0, "colocated"),
                   RequestResult("slow", .1, .2, 3, 10, True, 0, "colocated")]
        summary = summarize(results, SLO(1, 1, 1.5))
        self.assertEqual(summary["e2e_slo_requests"], 1)
        self.assertAlmostEqual(summary["e2e_goodput_tokens_per_s"], 10 / 3)
        self.assertIsNone(summary["sla_goodput_tokens_per_s"])

    def test_text_trace_uses_real_token_sequences(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(character) for character in text]
        trace = build_text_trace(["abc", "xy"], Tokenizer(),
                                 TraceConfig(requests=3, seed=17), 8)
        self.assertEqual(trace[0].prompt_token_ids, [97, 98, 99])
        self.assertEqual(trace[1].prompt_tokens, 2)
        with self.assertRaises(ValueError):
            RequestSpec("bad", 0, 2, 8, 17, [1])

    def test_final_environment_rejects_debug(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_JOB_PARTITION": "gpu",
                                     "SLURM_JOB_QOS": "debug"}):
            with self.assertRaises(RuntimeError):
                validate_final_environment()

    def test_analysis_rejects_trace_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cap = root / "cap.json"
            cap.write_text(json.dumps({"slurm": {"SLURM_JOB_ID": "123"},
                                       "probes": {"nvidia_smi_query": {"stdout": "A40"}}}))
            server_config = root / "server.json"
            server_config.write_text("{}")
            trace = root / "trace.jsonl"
            trace.write_text("{}\n")
            trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
            config = {"model": "heteroserve-model"}
            config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            manifests = []
            for i in range(3):
                raw = root / f"{i}.jsonl"
                raw.write_text(json.dumps({"run_id": f"run{i}", "request_id": "r",
                                           "outcome": "success", "output_tokens": 8,
                                           "output_sha256": "same", "pool": "colocated"}) + "\n")
                manifest = root / f"{i}.manifest.json"
                manifest.write_text(json.dumps({
                    "evidence_label": "real_gpu_final", "run_id": f"run{i}",
                    "trace": str(trace),
                    "trace_sha256": trace_hash,
                    "config": config, "config_sha256": config_hash,
                    "raw_results": str(raw),
                    "raw_results_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                    "summary": {"requests": 1, "successes": 1, "success_rate": 1.0,
                                "p95_ttft_s": .1, "p95_e2e_s": .2,
                                "token_throughput_per_s": 40.0,
                                "e2e_goodput_tokens_per_s": 40.0},
                    "environment": {"slurm_job_id": "123", "vllm": "0.11.2",
                                    "source_sha256": "source"}}))
                manifests.append(str(manifest))
            index = root / "evaluation.json"
            index.write_text(json.dumps({"evidence_label": "real_gpu_final", "run_id": "r",
                                         "trials": manifests, "capabilities": str(cap),
                                         "capabilities_sha256": hashlib.sha256(cap.read_bytes()).hexdigest(),
                                         "server_config": str(server_config),
                                         "server_config_sha256": hashlib.sha256(server_config.read_bytes()).hexdigest()}))
            report = analyze(index)
            self.assertTrue(report["output_consistent_across_trials"])
            self.assertEqual(report["metrics"]["p95_ttft_s"]["median"], .1)
            changed = json.loads(Path(manifests[2]).read_text())
            changed["trace_sha256"] = "different"
            Path(manifests[2]).write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "drift"):
                analyze(index)

    def test_bootstrap_requires_repetitions(self):
        with self.assertRaises(ValueError):
            bootstrap_ci([1, 2])


if __name__ == "__main__":
    unittest.main()
