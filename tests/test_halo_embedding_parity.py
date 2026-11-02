import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import io
import json

spec = importlib.util.spec_from_file_location("parity", Path(__file__).resolve().parents[1] / "scripts/check-halo-embedding-parity.py")
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


class Parity(unittest.TestCase):
    def test_threshold_and_complete_pair_report(self):
        a = [1.] + [0.] * 1023
        b = [0., 1.] + [0.] * 1022
        good = parity.compare([a] * 200, [a] * 200)
        self.assertTrue(good["accepted"])
        self.assertEqual(good["min_cosine"], 1)
        bad = parity.compare([a] * 200, [a] * 199 + [b])
        self.assertFalse(bad["accepted"])
        self.assertEqual(len(bad["pair_cosines"]), 200)
        self.assertEqual(bad["min_cosine"], 0)
        self.assertEqual(bad["mean_cosine"], .995)

    def test_invalid_output_cannot_pass(self):
        a = [1.] + [0.] * 1023
        for rows in [[a] * 199, [[0.] * 1024] * 200,
                     [[float("nan")] * 1024] * 200, [[1.]] * 200]:
            with self.assertRaises(ValueError):
                parity.compare([a] * 200, rows)


class DirectRunnerRequests(unittest.TestCase):
    def test_qualification_explicitly_permits_loading_on_both_runners(self):
        requests = []
        def respond(request, timeout):
            payload = json.loads(request.data)
            self.assertIs(payload["allow_engine_swap"], True)
            self.assertEqual(payload["model"], parity.MODEL)
            self.assertLessEqual(len(payload["input"]), 8)
            requests.extend(payload["input"])
            return io.BytesIO(json.dumps({"data": [
                {"index": i, "embedding": [float(text)]}
                for i, text in enumerate(payload["input"])
            ]}).encode())
        with patch.object(parity.urllib.request, "urlopen", side_effect=respond):
            texts = [str(i) for i in range(200)]
            rows = parity.embed("http://direct-runner", texts)
        self.assertEqual(requests, texts)
        self.assertEqual(rows, [[float(i)] for i in range(200)])
