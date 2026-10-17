import json
import tempfile
import unittest
from pathlib import Path

from heteroserve.router import load_config, make_policy, make_states
from heteroserve.schema import RequestSpec


class RouterConfigurationTests(unittest.TestCase):
    def config(self, mode="threshold"):
        return {"mode": mode, "model": "heteroserve-model", "threshold": 100,
                "short_pool": "a", "long_pool": "b", "max_inflight_per_pool": 2,
                "pools": [{"name": "a", "url": "http://127.0.0.1:8001"},
                          {"name": "b", "url": "http://127.0.0.1:8002"}]}

    def test_threshold_uses_only_deployed_available_pool(self):
        config = self.config()
        states = make_states(config, {"a": True, "b": True}, {"a": 0, "b": 2})
        request = RequestSpec("r", 0, 200, 8, 17)
        self.assertEqual(make_policy(config).choose(request, states), "a")

    def test_disaggregated_requires_connector_evidence(self):
        config = self.config()
        config["pools"][1]["route_kind"] = "disaggregated"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "connector evidence"):
                load_config(path)

    def test_adaptive_requires_rates(self):
        config = self.config("adaptive")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "measured positive"):
                load_config(path)

    def test_round_robin_uses_both_eligible_pools(self):
        config = self.config("round_robin")
        policy = make_policy(config)
        request = RequestSpec("r", 0, 20, 8, 17)
        states = make_states(config, {"a": True, "b": True}, {"a": 0, "b": 0})
        self.assertEqual([policy.choose(request, states) for _ in range(4)],
                         ["a", "b", "a", "b"])
        states = make_states(config, {"a": False, "b": True}, {"a": 0, "b": 0})
        self.assertEqual(policy.choose(request, states), "b")

    def test_least_active_prefers_less_loaded_pool(self):
        config = self.config("least_active")
        policy = make_policy(config)
        request = RequestSpec("r", 0, 20, 8, 17)
        states = make_states(config, {"a": True, "b": True}, {"a": 2, "b": 0})
        self.assertEqual(policy.choose(request, states), "b")
        states = make_states(config, {"a": True, "b": False}, {"a": 0, "b": 0})
        self.assertEqual(policy.choose(request, states), "a")


if __name__ == "__main__":
    unittest.main()
