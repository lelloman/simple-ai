"""Provider contract tests; tensor tests use CPU torch, never download weights."""
import importlib.util
import json
from http.server import ThreadingHTTPServer
import urllib.request
import urllib.error
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("embeddings", Path(__file__).resolve().parents[1] / "scripts/simple_ai_embedding_provider.py")
provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provider)


class Validation(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(provider.validate({"model": provider.MODEL_ID, "input": "hello"}), ["hello"])
        for value in [[], [" "], ["a"] * 257, ["a" * 32769], ["a" * 32768] * 33, [5], None]:
            with self.subTest(value=str(value)[:30]), self.assertRaises(ValueError):
                provider.validate({"model": provider.MODEL_ID, "input": value})
        for payload in [{"model": "other", "input": "hello"}, {"model": provider.MODEL_ID, "input": "hello", "dimensions": 512}]:
            with self.assertRaises(ValueError):
                provider.validate(payload)

    def test_http_rejects_oversize_with_400(self):
        class FakeProvider:
            def embed(self, payload):
                provider.validate(payload)
                return {"data": []}

        handler = type("TestHandler", (provider.Handler,), {"provider": FakeProvider()})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            payload = json.dumps({"model": provider.MODEL_ID, "input": ["text"] * 257}).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/embeddings", data=payload,
                headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


class TensorContract(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def test_pool_normalise_dimension(self):
        torch = self.torch
        hidden = torch.zeros((2, 3, 1024), dtype=torch.float16)
        hidden[:, 0, 20] = 50  # Padding/earlier tokens must not enter pooling.
        hidden[0, -1, :2] = torch.tensor([3., 4.])
        hidden[1, -1, 2] = 2.
        mask = torch.tensor([[0, 0, 1], [1, 1, 1]])
        result = provider.vectors(torch, hidden, mask)
        self.assertEqual(len(result[0]), 1024)
        self.assertAlmostEqual(result[0][0], .6)
        self.assertAlmostEqual(result[0][1], .8)
        self.assertEqual(result[1][2], 1.)
        for bad in [torch.zeros((2, 3, 1024)), torch.ones((2, 3, 512)), hidden * float("nan")]:
            with self.assertRaises(RuntimeError):
                provider.vectors(torch, bad, mask)
        with self.assertRaises(RuntimeError):
            provider.vectors(torch, hidden, torch.tensor([[1, 0, 0], [1, 1, 1]]))

    def test_microbatch_order_raw_text_truncation_and_failure_recovery(self):
        torch = self.torch
        seen = []

        class Encoding(dict):
            def to(self, device):
                self.device = device
                return self

        def tokenizer(texts, **kwargs):
            self.assertEqual(kwargs, dict(padding=True, truncation=True, max_length=512, return_tensors="pt"))
            seen.extend(texts)
            return Encoding(attention_mask=torch.ones((len(texts), 2), dtype=torch.int),
                            ids=torch.tensor([int(x) for x in texts]))

        def model(**kwargs):
            hidden = torch.zeros((len(kwargs["ids"]), 2, 1024), dtype=torch.float16)
            for row, value in enumerate(kwargs["ids"]):
                hidden[row, -1, int(value)] = 2
            return types.SimpleNamespace(last_hidden_state=hidden)

        p = provider.Provider.__new__(provider.Provider)
        p.torch, p.batch_size, p.tokenizer, p.model = torch, 2, tokenizer, model
        p.lock, p.slots = threading.Lock(), threading.BoundedSemaphore(8)
        request = {"model": provider.MODEL_ID, "input": ["4", "1", "7", "3", "8"]}
        result = p.embed(request)
        self.assertEqual(seen, request["input"])
        self.assertEqual([x["index"] for x in result["data"]], list(range(5)))
        self.assertEqual([x["embedding"].index(1.) for x in result["data"]], [4, 1, 7, 3, 8])
        with patch.object(p, "model", side_effect=RuntimeError("GPU error")):
            with self.assertRaises(RuntimeError):
                p.embed(request)
        self.assertEqual(len(p.embed(request)["data"]), 5)


if __name__ == "__main__":
    unittest.main()
