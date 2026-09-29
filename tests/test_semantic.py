"""JEV protocol/decision tests; no model download or GPU required."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import simple_ai_semantic as semantic

spec = importlib.util.spec_from_file_location("semantic_evaluation", SCRIPTS / "evaluate-semantic.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
spec = importlib.util.spec_from_file_location("semantic_benchmark", SCRIPTS / "benchmark-semantic.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
FIXTURES = Path(__file__).parent / "fixtures/jev"
HEAD = json.loads((FIXTURES / "decision_head.json").read_text())
TEMPERATURES = json.loads((FIXTURES / "calibration.json").read_text())["per_kind"]


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        words = ["false", "true"] + [str(i) for i in range(6)] + list("ABCDEFGHIJKLMNOP")
        if text in words:
            return [HEAD["verbalizer_ids"][words.index(text)]]
        return [ord(c)+100000 for c in text]

    def decode(self, ids):
        return "".join(chr(i-100000) for i in ids)


class Backend:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def generate(self, ids, labels):
        with self.lock:
            self.calls.append((ids, labels))
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(.005)
        with self.lock:
            self.active -= 1
        return {"logprobs": [math.log(.2)] + [math.log(.8/(len(labels)-1))]*(len(labels)-1),
                "prompt_tokens": len(ids), "completion_tokens": 1, "cached_tokens": None}


def scorer(backend=None, **kwargs):
    return semantic.SemanticScorer(Tokenizer(), backend or Backend(), HEAD, TEMPERATURES, **kwargs)


def wire_response(labels):
    return {"choices": [{"text": "DO NOT PARSE", "finish_reason": "length", "logprobs": {
        "top_logprobs": [{f"token_id:{token}": math.log(1/len(labels)) for token in labels}]}}],
        "usage": {"prompt_tokens": 25, "completion_tokens": 1}}


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.scorer = scorer(self.backend, concurrency=2)

    def test_noul_trained_order_bias_and_temperature(self):
        result = self.scorer.semantic_score("state", "predicate")
        self.assertEqual(self.backend.calls[0][1], HEAD["verbalizer_ids"][:2])
        expected = semantic.distribution([(math.log(p)+b)/TEMPERATURES["noul"]
                                         for p,b in zip([.2,.8], HEAD["bias"][:2])])[1]
        self.assertAlmostEqual(result["score"], expected)
        self.assertEqual(result["selected"], "True")
        self.assertFalse(result["calibrated"])
        self.assertTrue(result["upstream_temperature_applied"])
        self.assertNotIn("label_probability_mass", result)

    def test_exact_template_preserves_structured_state(self):
        prepared = self.scorer.prepare({"text": "😀", "metadata": [1]}, ["p"], ["False", "True"])
        self.assertEqual(self.scorer.tokenizer.decode(prepared.branches[0]),
                         '[kind] noul\n[state] {"text": "😀", "metadata": [1]}\n[question] p\n[options]\nfalse\ntrue\n[decision]:')

    def test_batch_has_no_extra_warmup_and_preserves_order(self):
        result = self.scorer.semantic_score_many("state", ["p1", "p2", "p3", "p4"])
        self.assertEqual(len(self.backend.calls), 4)
        self.assertEqual(self.backend.peak, 2)
        self.assertEqual([r["predicate"] for r in result["results"]], ["p1", "p2", "p3", "p4"])
        self.assertEqual(result["usage"]["warmup_calls"], 0)
        self.assertIsNone(result["usage"]["cached_tokens"])

    def test_choice_slots_and_limit(self):
        result = self.scorer.semantic_classify("refund", ["billing", "returns", "security"])
        self.assertEqual(self.backend.calls[0][1], HEAD["verbalizer_ids"][8:11])
        self.assertAlmostEqual(sum(result["scores"].values()), 1)
        self.assertIn("A) billing", self.scorer.tokenizer.decode(self.backend.calls[0][0]))
        self.scorer.semantic_classify("s", [str(i) for i in range(16)])
        for options in [["x"], ["x", "x"], [str(i) for i in range(17)]]:
            with self.assertRaises(ValueError):
                self.scorer.semantic_classify("s", options)

    def test_validate_all_branches_before_inference(self):
        for predicates in [[], [""], [1], ["p"]*129, ["x"*8193], ["ok", "x"*8000]]:
            self.scorer.context = 600
            with self.assertRaises(ValueError):
                self.scorer.semantic_score_many("state", predicates)
            self.assertEqual(self.backend.calls, [])

    def test_busy_and_release_after_failure(self):
        self.scorer.admission.acquire()
        with self.assertRaises(semantic.BusyError):
            self.scorer.semantic_score("state", "p")
        self.scorer.admission.release()
        with self.assertRaises(ValueError):
            self.scorer.semantic_score(None, "p")
        self.assertGreater(self.scorer.semantic_score("state", "p")["score"], .7)

    def test_parse_failures_and_unknown_cache(self):
        valid = wire_response([1,2])
        self.assertIsNone(semantic.parse_readout(valid, [1,2])["cached_tokens"])
        for values in [{"token_id:1": None}, {"token_id:1": -1},
                       {"token_id:1": math.nan, "token_id:2": -1},
                       {"token_id:1": -math.inf, "token_id:2": -math.inf}]:
            valid["choices"][0]["logprobs"]["top_logprobs"] = [values]
            with self.assertRaises(semantic.BackendError):
                semantic.parse_readout(valid, [1,2])
        self.assertEqual(semantic.distribution([-10000,-10000]), [.5,.5])
        self.assertEqual(semantic.json_safe({"raw": -math.inf}), {"raw": "-Infinity"})

    def test_failed_cache_reset_is_not_treated_as_cold(self):
        backend = semantic.VLLM()
        with patch.object(backend, "request", return_value={"success": False}):
            with self.assertRaises(semantic.BackendError):
                backend.flush()

    def test_model_verification_requires_adapter_and_revision(self):
        backend = semantic.VLLM()
        root = "/models/" + semantic.REVISION
        with patch.object(backend, "request", return_value={"data": [
            {"id": semantic.MODEL, "root": root},
            {"id": "jev-decision", "root": root+"/adapter_vllm"}]}):
            backend.verify_model(semantic.MODEL, semantic.REVISION)
            with self.assertRaises(semantic.BackendError):
                backend.verify_model(semantic.MODEL, "wrong-revision")

    def test_calibration_known_answers_and_ambiguous_exclusion(self):
        rows = [{"kind": "binary", "expected": "True", "selected": "True", "scores": {"True": .8, "False": .2}},
                {"kind": "binary", "expected": "False", "selected": "True", "scores": {"True": .8, "False": .2}},
                {"kind": "binary", "expected": None, "selected": "True", "scores": {"True": 1, "False": 0}}]
        result = evaluation.metrics(rows)
        self.assertEqual(result["labelled_count"], 2)
        self.assertEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["binary_brier"], .34)
        self.assertAlmostEqual(result["ece_10_equal_width_bins"], .3)

    def test_fixture_integrity(self):
        cases = json.loads((SCRIPTS.parent / "tests/fixtures/semantic-quality.json").read_text())
        self.assertEqual(len({c["id"] for c in cases}), len(cases))
        for case in cases:
            if case["expected"] is not None:
                self.assertIn(case["expected"], ["True", "False"] if case["kind"] == "binary" else case["options"])


class WireTests(unittest.TestCase):
    def test_vllm_and_http_service_end_to_end(self):
        calls = []
        class FakeVLLM(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(body)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(wire_response(body["allowed_token_ids"])).encode())
            def log_message(self, *args):
                pass
        backend = ThreadingHTTPServer(("127.0.0.1",0), FakeVLLM)
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        service = scorer(semantic.VLLM(f"http://127.0.0.1:{backend.server_port}"))
        handler = type("Handler", (semantic.Handler,), {"scorer": service})
        server = ThreadingHTTPServer(("127.0.0.1",0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for route, body, field in [
                ("score", {"state":"log", "predicate":"passed"}, "score"),
                ("score-many", {"state":"log", "predicates":["p1","p2"]}, "results"),
                ("classify", {"state":"log", "options":["success","failure"]}, "scores")]:
                with urlopen(Request(f"http://127.0.0.1:{server.server_port}/v1/semantic/{route}",
                                     data=json.dumps(body).encode())) as response:
                    self.assertIn(field, json.load(response))
            for body in calls:
                self.assertEqual(body["model"], "jev-decision")
                self.assertEqual(body["max_tokens"], 1)
                self.assertTrue(body["return_tokens_as_token_ids"])
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(f"http://127.0.0.1:{server.server_port}/v1/semantic/score-many",
                                data=b'{"state":"logs","predicates":[]}'))
            self.assertEqual(caught.exception.code, 400)
        finally:
            server.shutdown(); backend.shutdown()
            server.server_close(); backend.server_close()


class BenchmarkTests(unittest.TestCase):
    def test_baseline_clears_every_decision_shared_warms_only_once(self):
        class Recorder:
            def __init__(self):
                self.events = []
                self.backend = self

            def flush(self):
                self.events.append("flush")

            def run(self, state, predicates, keys, warm):
                self.events.append((len(predicates), warm))
                return {"usage": {"branch_cached_tokens": [0]}}

            def semantic_score_many(self, state, predicates):
                self.events.append((len(predicates), True))
                return {"usage": {"branch_cached_tokens": [10] * len(predicates)}}

        with patch.object(benchmark.MemorySampler, "sample", lambda self: None):
            scorer = Recorder()
            result = benchmark.trial(scorer, "state", ["p1", "p2", "p3"], "independent", 0)
            self.assertEqual(scorer.events, ["flush", (1, False), "flush", (1, False), "flush", (1, False)])
            self.assertFalse(result["cache_reuse_observed"])
            self.assertIsNone(result["peak_device_memory_mib"])
            scorer.events.clear()
            result = benchmark.trial(scorer, "state", ["p1", "p2", "p3"], "shared", 0)
            self.assertEqual(scorer.events, ["flush", (3, True)])
            self.assertTrue(result["cache_reuse_observed"])

    def test_metadata_does_not_record_credentials(self):
        self.assertEqual(benchmark.backend_metadata({"api_key": "secret", "version": "test",
                                                    "serve": {"revision": "abc", "admin_api_key": "secret"}}),
                         {"version": "test", "serve": {"revision": "abc"}})



class DecisionTests(unittest.TestCase):
    def payload(self):
        return {"model":semantic.MODEL,"state":{"message":"Please refund the duplicate charge."},"questions":[
            {"id":"refund","type":"boolean","instruction":"Does the customer request a refund?"},
            {"id":"category","type":"choice","instruction":"Choose the topic.","options":[{"id":"billing","description":"Billing"},{"id":"technical","description":"Technical support"}]},
            {"id":"urgency","type":"rating","instruction":"Rate urgency.","levels":["none","very low","low","medium","high","critical"]}]}
    def test_mixed_answers_are_ordered_normalized_and_typed(self):
        result=scorer().decisions(self.payload())
        self.assertEqual([a['id'] for a in result['answers']],['refund','category','urgency'])
        self.assertEqual(result['usage']['question_count'],3)
        self.assertFalse(result['calibrated'])
        for answer in result['answers']:
            self.assertAlmostEqual(sum(answer['probabilities'].values()),1)
        rating=result['answers'][2]
        self.assertAlmostEqual(rating['value'],sum(int(k)*v for k,v in rating['probabilities'].items()))
        self.assertEqual(set(result['answers'][1]['probabilities']),{'billing','technical'})
    def test_validation_happens_before_any_inference(self):
        backend=Backend()
        for mutate in [lambda p:p['questions'].append(p['questions'][0]),lambda p:p['questions'][2]['levels'].pop(),lambda p:p.update(unexpected=True),lambda p:p['questions'][1]['options'].append(p['questions'][1]['options'][0])]:
            payload=self.payload();mutate(payload)
            with self.assertRaises(ValueError):scorer(backend).decisions(payload)
        with self.assertRaises(ValueError):scorer(backend,context=10).decisions(self.payload())
        self.assertEqual(backend.calls,[])
    def test_busy_is_rejected(self):
        instance=scorer();instance.admission.acquire()
        try:
            with self.assertRaises(semantic.BusyError):instance.decisions(self.payload())
        finally:instance.admission.release()


if __name__ == "__main__":
    unittest.main()
