import io
import unittest
from unittest import mock

from heteroserve.client import Endpoint, _completion_path, stream_completion
from heteroserve.schema import WorkloadRequest


class ClientTests(unittest.TestCase):
    def test_completion_path_preserves_prefix(self) -> None:
        self.assertEqual(
            _completion_path("http://127.0.0.1:8000/api"),
            ("http", "127.0.0.1", 8000, "/api/v1/completions"),
        )

    def test_rejects_non_http_url(self) -> None:
        with self.assertRaises(ValueError):
            _completion_path("file:///tmp/socket")

    def test_stream_records_usage_and_first_content(self) -> None:
        events = (
            'data: {"choices":[{"text":"hello","finish_reason":null}]}\n'
            'data: {"choices":[],"usage":{"completion_tokens":3}}\n'
            'data: [DONE]\n'
        ).encode()

        class Response:
            status = 200

            def __init__(self) -> None:
                self.stream = io.BytesIO(events)

            def readline(self):
                return self.stream.readline()

        connection = mock.Mock()
        connection.getresponse.return_value = Response()
        request = WorkloadRequest("r", 0, 1, 3, 17, "prompt", "test", "1")
        with mock.patch("heteroserve.client.http.client.HTTPConnection", return_value=connection):
            result = stream_completion(Endpoint("pool", "http://localhost:8000", "model"), request, 0, 0, 0)
        self.assertTrue(result.success)
        self.assertEqual(result.output_tokens, 3)
        self.assertIsNotNone(result.response_text_sha256)
        connection.request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
