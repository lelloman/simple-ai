import importlib.util
from pathlib import Path
import unittest

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
