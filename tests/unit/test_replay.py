import asyncio
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None

if httpx is not None:
    from heteroserve.replay import replay
from heteroserve.schema import RequestResult


@unittest.skipIf(httpx is None, "httpx==0.28.1 is not installed")
class ReplayTests(unittest.TestCase):
    def test_streaming_replay_writes_raw_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            trace.write_text(json.dumps({
                "request_id": "r1", "arrival_s": 0, "prompt_tokens": 3,
                "max_output_tokens": 2, "seed": 17,
            }) + "\n")
            output = root / "raw.jsonl"
            args = Namespace(
                trace=trace, output=output, run_id="mock-1", base_url="http://test",
                model="heteroserve-model", model_revision="commit", tokenizer_revision="commit",
                prompt_token_id=42, concurrency=1, timeout_s=2, slo_ttft_s=10,
                slo_tpot_s=10, evidence_label="gpu_debug",
            )

            def handle(request: httpx.Request) -> httpx.Response:
                if request.url.path == "/v1/models":
                    return httpx.Response(200, json={"data": [{"id": "heteroserve-model"}]})
                payload = json.loads(request.content)
                self.assertEqual(payload["prompt"], [42, 42, 42])
                self.assertEqual(payload["stream_options"], {"include_usage": True})
                events = [
                    {"choices": [{"text": "a"}]},
                    {"choices": [{"text": "b"}]},
                    {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
                ]
                body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
                return httpx.Response(200, text=body + "data: [DONE]\n\n")

            async def run() -> dict:
                async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                    return await replay(args, client)

            manifest = asyncio.run(run())
            record = json.loads(output.read_text())
            self.assertTrue(record["success"])
            self.assertEqual(record["output_tokens"], 2)
            self.assertEqual(record["stream_text_chunks"], 2)
            self.assertIsNone(record["tpot_s"])
            self.assertIsNotNone(record["approximate_tpot_s"])
            self.assertEqual(record["outcome"], "success")
            self.assertLessEqual(record["scheduled_s"], record["actual_arrival_s"])
            self.assertLessEqual(record["actual_arrival_s"], record["server_submission_s"])
            self.assertEqual(manifest["summary"]["successes"], 1)
            self.assertTrue(output.with_suffix(".manifest.json").exists())

    def test_batched_text_chunks_exclude_tpot(self) -> None:
        result = RequestResult("r", 0, 1, 3, 3, True, 0, "p", None, 2, False)
        self.assertIsNone(result.tpot_s)

    def test_proxy_can_validate_model_at_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            trace.write_text(json.dumps({
                "request_id": "r1", "arrival_s": 0, "prompt_tokens": 3,
                "max_output_tokens": 2, "seed": 17,
            }) + "\n")
            args = Namespace(
                trace=trace, output=root / "raw.jsonl", run_id="split-mock",
                base_url="http://proxy", model_check_url="http://decode",
                model="heteroserve-model", model_revision="commit",
                tokenizer_revision="commit", prompt_token_id=42,
                concurrency=1, timeout_s=2, slo_ttft_s=10, slo_tpot_s=10,
                evidence_label="gpu_debug",
            )

            def handle(request: httpx.Request) -> httpx.Response:
                if request.url.host == "decode" and request.url.path == "/v1/models":
                    return httpx.Response(200, json={"data": [{"id": "heteroserve-model"}]})
                self.assertEqual(request.url.host, "proxy")
                self.assertEqual(request.url.path, "/v1/completions")
                events = [
                    {"choices": [{"text": "a"}]},
                    {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
                ]
                body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
                return httpx.Response(200, text=body + "data: [DONE]\n\n")

            async def run() -> dict:
                async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                    return await replay(args, client)

            manifest = asyncio.run(run())
            self.assertEqual(manifest["summary"]["successes"], 1)
            self.assertEqual(manifest["config"]["model_check_url"], "http://decode")


if __name__ == "__main__":
    unittest.main()
