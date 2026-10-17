import tempfile
import unittest
from pathlib import Path

from heteroserve.split_metrics import summarize


class SplitMetricsTests(unittest.TestCase):
    def test_nixl_and_stage_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            before = Path(directory) / "before.prom"
            after = Path(directory) / "after.prom"
            before.write_text(
                'vllm:request_decode_time_seconds_sum{engine="0"} 2\n'
                'vllm:request_decode_time_seconds_count{engine="0"} 4\n'
                'vllm:nixl_bytes_transferred_sum{engine="0"} 1024\n'
                'vllm:nixl_bytes_transferred_count{engine="0"} 1\n'
            )
            after.write_text(
                'vllm:request_decode_time_seconds_sum{engine="0"} 5\n'
                'vllm:request_decode_time_seconds_count{engine="0"} 6\n'
                'vllm:nixl_bytes_transferred_sum{engine="0"} 5120\n'
                'vllm:nixl_bytes_transferred_count{engine="0"} 3\n'
                'vllm:nixl_num_failed_transfers_total{engine="0"} 0\n'
            )
            result = summarize(before, after, "decode")
            self.assertEqual(result["decode_s"], {"count": 2, "sum": 3.0, "mean": 1.5})
            self.assertEqual(result["kv_transfer_bytes"],
                             {"count": 2, "sum": 4096.0, "mean": 2048.0})
            self.assertEqual(result["vllm:nixl_num_failed_transfers_total"], 0)


if __name__ == "__main__":
    unittest.main()
