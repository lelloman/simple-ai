import importlib.util
import json
import os
import tempfile
from unittest.mock import patch
from pathlib import Path
import time
import unittest

spec = importlib.util.spec_from_file_location(
    "qualification", Path(__file__).resolve().parents[1] / "scripts/check-small-embedders-live.py")
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


def response(count=32, dimension=1024):
    return {"data": [{"index": i, "embedding": [0.5] * dimension} for i in range(count)]}


class FakeClient:
    def request(self, endpoint, payload):
        time.sleep(.001)
        return response(len(payload["input"]))

    def probe(self, probes):
        time.sleep(.001)
        return {endpoint: .001 for endpoint in probes}


class QualificationTests(unittest.TestCase):
    def test_input_size_and_variation(self):
        self.assertEqual(len(qualification.inputs(0)), 32)
        self.assertTrue(all(len(text) == 700 for text in qualification.inputs(0)))
        self.assertEqual(len(set(qualification.inputs(0) + qualification.inputs(1))), 64)

    def test_rejects_bad_count_order_dimensions_and_values(self):
        bad = [response(31), response(dimension=768)]
        ordered = response()
        ordered["data"][0]["index"] = 1
        bad.append(ordered)
        for value in (float("nan"), float("inf"), "bad", 0):
            invalid = response()
            invalid["data"][0]["embedding"] = [value] * 1024
            bad.append(invalid)
        for payload in bad:
            with self.subTest(payload=str(payload)[:50]), self.assertRaises(ValueError):
                qualification.validate_embeddings(payload, 32, 1024)

    def test_throughput_and_contention(self):
        for concurrency in (1, 4, 8):
            result = qualification.benchmark(FakeClient(), qualification.MODELS[0], concurrency)
            self.assertEqual(result["texts"], 512)
            self.assertEqual(result["dimension"], 1024)
            self.assertEqual(result["request_latency"]["count"], 16)
            self.assertGreater(result["texts_per_s"], 0)
            self.assertIsNone(result["during_probes"])
        result = qualification.benchmark(FakeClient(), qualification.MODELS[1], 4, {"probe": {}})
        self.assertGreater(result["during_probes"]["probe"]["count"], 0)

    def test_missing_models_persists_incomplete_report(self):
        probes = {"/v1/embeddings": {"model": "class:embed_large", "input": "text"},
                  "/v1/classifications": {"model": "existing", "input": "text"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe_file = root / "probes.json"
            baseline_file = root / "before.json"
            output = root / "loaded.json"
            probe_file.write_text(json.dumps(probes))
            digest = qualification.hashlib.sha256(json.dumps(probes, sort_keys=True).encode()).hexdigest()
            baseline_file.write_text(json.dumps({"complete": True, "probe_sha256": digest,
                                                "url": "https://ai.lelloman.com", "probes": {}}))
            args = ["check", "--probes", str(probe_file), "--baseline", str(baseline_file),
                    "--output", str(output)]
            with patch("sys.argv", args), patch.dict(os.environ, {"SIMPLE_AI_API_KEY": "test"}), \
                    patch.object(qualification.Client, "request", return_value={"data": []}):
                with self.assertRaisesRegex(ValueError, "missing"):
                    qualification.main()
            report = json.loads(output.read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["error"], "ValueError")
            self.assertEqual(report["matrix"], [])

    def test_failed_batch_or_probe_fails_measurement(self):
        class FailedBatch(FakeClient):
            calls = 0

            def request(self, endpoint, payload):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("batch failed")
                return super().request(endpoint, payload)

        class FailedProbe(FakeClient):
            def probe(self, probes):
                raise RuntimeError("probe failed")

        with self.assertRaisesRegex(RuntimeError, "batch failed"):
            qualification.benchmark(FailedBatch(), qualification.MODELS[0], 4)
        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            qualification.benchmark(FailedProbe(), qualification.MODELS[0], 4, {"probe": {}})


if __name__ == "__main__":
    unittest.main()
