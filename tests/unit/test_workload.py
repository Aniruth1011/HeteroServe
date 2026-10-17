import io
import json
import unittest
from pathlib import Path
from unittest import mock

from heteroserve.workload import build_hybrid_workload, load_burst_schedule, load_oasst_pairs


class WorkloadTests(unittest.TestCase):
    def test_oasst_uses_best_ranked_direct_answer(self) -> None:
        rows = [
            {"message_id": "p", "parent_id": None, "role": "prompter", "lang": "en", "text": "question"},
            {"message_id": "a2", "parent_id": "p", "role": "assistant", "lang": "en", "text": "second", "rank": 1},
            {"message_id": "a1", "parent_id": "p", "role": "assistant", "lang": "en", "text": "first", "rank": 0},
        ]
        stream = io.StringIO("".join(json.dumps(row) + "\n" for row in rows))
        with mock.patch("heteroserve.workload._open_text", return_value=stream):
            pairs = load_oasst_pairs(Path("oasst.jsonl"))
        self.assertEqual(pairs, [("p", "question", "first")])

    def test_burst_schedule_filters_failures_and_relativizes_time(self) -> None:
        content = "Timestamp,Request tokens,Response tokens\n10,20,0\n12,30,4\n15,40,8\n"
        with mock.patch.object(Path, "open", return_value=io.StringIO(content)):
            rows = load_burst_schedule(Path("burst.csv"))
        self.assertEqual(rows, [(0.0, 30, 4, "1"), (3.0, 40, 8, "2")])

    def test_hybrid_label_and_determinism(self) -> None:
        pairs = [("p1", "hello world", "answer text")]
        schedule = [(0.0, 100, 4, "9")]
        first = build_hybrid_workload(pairs, schedule, seed=17)
        second = build_hybrid_workload(pairs, schedule, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first[0].source, "hybrid_oasst1_burstgpt")
        self.assertEqual(first[0].prompt_tokens, 2)


if __name__ == "__main__":
    unittest.main()
