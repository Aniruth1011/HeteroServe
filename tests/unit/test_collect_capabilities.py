import contextlib
import io
import json
import os
import unittest
from unittest import mock

from scripts.probes import collect_capabilities as capabilities


class CapabilityProbeTests(unittest.TestCase):
    def test_gpu_label_requires_nonempty_allocation_value(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(capabilities.has_gpu_allocation())
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            self.assertTrue(capabilities.has_gpu_allocation())

    def test_missing_command_is_structured(self) -> None:
        result = capabilities.run(["heteroserve-command-that-does-not-exist"])
        self.assertEqual(result["status"], "not_found")
        self.assertIn("error", result)

    def test_missing_package_is_null(self) -> None:
        with mock.patch.object(
            capabilities.importlib.metadata,
            "version",
            side_effect=capabilities.importlib.metadata.PackageNotFoundError,
        ):
            versions = capabilities.package_versions()
        self.assertTrue(all(value is None for value in versions.values()))

    def test_stdout_json_is_round_trippable(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            capabilities.write_json({"schema_version": 1}, "-")
        self.assertEqual(json.loads(stream.getvalue()), {"schema_version": 1})


if __name__ == "__main__":
    unittest.main()
