import unittest

from heteroserve.statistics import paired_bootstrap_delta


class StatisticsTests(unittest.TestCase):
    def test_paired_delta_uses_run_pairs(self) -> None:
        result = paired_bootstrap_delta([1, 2, 3], [2, 3, 4], samples=1000)
        self.assertEqual(result["paired_runs"], 3)
        self.assertEqual(result["median_delta"], 1)
        self.assertEqual(result["ci95_low"], 1)
        self.assertEqual(result["ci95_high"], 1)

    def test_rejects_unpaired_vectors(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_delta([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
