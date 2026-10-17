import json
import unittest
from pathlib import Path
from unittest import mock

from heteroserve.metrics import SLO, summarize
from heteroserve.policy import AdaptivePolicy, CostModel, DQAPolicy, GPUProfile, KVReservations, PoolState, ThresholdPolicy
from heteroserve.schema import RequestResult, RequestSpec
from heteroserve.trace import TraceConfig, generate, write_jsonl


class CoreTests(unittest.TestCase):
    def test_trace_is_deterministic_and_round_trippable(self) -> None:
        config = TraceConfig(requests=5, seed=7)
        self.assertEqual(generate(config), generate(config))
        path = Path("benchmarks/traces/unit.jsonl")
        with mock.patch.object(Path, "mkdir"), mock.patch.object(Path, "write_text") as write:
            write_jsonl(generate(config), path)
        payload = write.call_args.args[0]
        rows = [json.loads(line) for line in payload.splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["request_id"], "req-000000")

    def test_metric_formulas_and_tpot_exclusion(self) -> None:
        results = [
            RequestResult("a", 0, 1, 3, 3, True, 0.2, "p", last_token_s=3, exact_token_timing=True),
            RequestResult("b", 0, 2, 4, 1, True, 0.1, "p"),
            RequestResult("c", 1, None, 5, 0, False, 0.0, "p", "timeout"),
        ]
        summary = summarize(results, SLO(1.5, 1.1))
        self.assertEqual(results[0].tpot_s, 1.0)
        self.assertIsNone(results[1].tpot_s)
        self.assertEqual(summary["tpot_eligible"], 1)
        self.assertEqual(summary["tpot_excluded"], 1)
        self.assertEqual(summary["slo_requests"], 1)

    def test_cost_monotonicity_and_transfer_penalty(self) -> None:
        model = CostModel()
        pool = PoolState("p", True, 0, 100, 50)
        short = RequestSpec("s", 0, 10, 10, 1)
        long = RequestSpec("l", 0, 100, 10, 1)
        self.assertGreater(model.predict(long, pool), model.predict(short, pool))
        transfer = PoolState("x", True, 0, 100, 50, transfer_bandwidth_bytes_s=1000, bytes_per_prompt_token=10)
        self.assertGreater(model.predict(short, transfer), model.predict(short, pool))

    def test_threshold_fallback_and_adaptive_hysteresis(self) -> None:
        request = RequestSpec("r", 0, 1000, 10, 1)
        pools = [PoolState("short", True, 0, 100, 100), PoolState("long", False, 0, 1000, 100)]
        self.assertEqual(ThresholdPolicy(500, "short", "long").choose(request, pools), "short")
        adaptive = AdaptivePolicy(CostModel(), hysteresis_fraction=0.1)
        adaptive.previous = "short"
        close = [PoolState("short", True, 0, 1000, 100), PoolState("long", True, 0, 1050, 100)]
        self.assertEqual(adaptive.choose(request, close), "short")

    def test_fixed_and_bursty_trace(self) -> None:
        config = TraceConfig(requests=100, seed=11, arrival_rate=3,
                             shapes=((1024, 128, 1), (4096, 512, 1)),
                             arrival_pattern="bursty", burst_factor=5, burst_period_s=2)
        trace = generate(config)
        self.assertEqual(trace, generate(config))
        self.assertEqual({(r.prompt_tokens, r.max_output_tokens) for r in trace},
                         {(1024, 128), (4096, 512)})
        self.assertTrue(all(a.arrival_s <= b.arrival_s for a, b in zip(trace, trace[1:])))

    def test_dqa_congestion_transfer_and_reservation_lifecycle(self) -> None:
        request = RequestSpec("r", 0, 1000, 100, 1)
        base = PoolState("base", True, 0, 100, 100, route_kind="colocated")
        split = PoolState("split", True, 0, 1000, 100, route_kind="disaggregated",
                          transfer_bandwidth_bytes_s=1_000_000,
                          bytes_per_prompt_token=100, kv_bytes_per_token=10,
                          kv_capacity_bytes=20_000)
        model = CostModel()
        reservations = KVReservations()
        policy = DQAPolicy(model, reservations)
        self.assertEqual(policy.admit(request, [base, split]), "split")
        self.assertEqual(reservations.reserved_bytes("split"), 11_000)
        self.assertEqual(policy.finish("r"), 11_000)
        self.assertEqual(policy.finish("r"), 0)
        congested = PoolState("split", True, 0, 1000, 100, route_kind="disaggregated",
                              transfer_bandwidth_bytes_s=1_000_000,
                              bytes_per_prompt_token=100, kv_bytes_per_token=10,
                              kv_capacity_bytes=20_000, active_decode_requests=8,
                              remaining_decode_tokens=800, recent_tpot_s=0.1)
        self.assertGreater(model.predict(request, congested), model.predict(request, split))
        self.assertEqual(policy.choose(request, [base, congested]), "base")
        full = PoolState("split", True, 0, 1000, 100, route_kind="disaggregated",
                         kv_bytes_per_token=10, kv_capacity_bytes=10_000)
        self.assertEqual(policy.choose(request, [base, full]), "base")
        self.assertEqual(policy.choose(request, [base, PoolState("split", True, 0, 1000, 100,
                         route_kind="disaggregated", deployed=False)]), "base")

    def test_gpu_profile_overrides_pool_rates(self) -> None:
        request = RequestSpec("r", 0, 1000, 100, 1)
        pool = PoolState("p", True, 0, 100, 100, gpu_type="measured_gpu",
                         active_decode_requests=2)
        measured = CostModel({"measured_gpu": GPUProfile(1000, 50, 0.1, 0.2)})
        self.assertGreater(measured.predict(request, pool), 1.2)
        self.assertNotEqual(measured.predict(request, pool), CostModel().predict(request, pool))


if __name__ == "__main__":
    unittest.main()
