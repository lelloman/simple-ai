import importlib.util
from pathlib import Path
import sys
import tempfile
import json
import copy
import threading
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('summary_provider', Path(__file__).parents[1] / 'scripts/simple_ai_summary_provider.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SummaryTests(unittest.TestCase):
    def test_limits(self):
        self.assertEqual(m.validate({'model': m.MODEL, 'input': ['hello']})[1], 60)
        for changes in [{'input': []}, {'input': [' ']}, {'input': ['x'] * 65},
                        {'input': ['x' * 32769]}, {'max_tokens': True}, {'max_tokens': 0},
                        {'max_tokens': 61}, {'model': 'class:big'}, {'target': 'en'}]:
            with self.assertRaises(ValueError):
                m.validate({'model': m.MODEL, 'input': ['hello'], **changes})

    def test_prompt_and_no_truncation(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                return [1] * len(messages[1]['content'])
        tokenizer = Tokenizer()
        self.assertEqual(len(m.prompts(tokenizer, ['x' * 964], 60)[0]['prompt_token_ids']), 964)
        self.assertIn('facts only', tokenizer.messages[0]['content'])
        self.assertIn('If the card gives no place, omit the location', tokenizer.messages[0]['content'])
        self.assertIn('Never infer a location from the domain, TLD, language or names', tokenizer.messages[0]['content'])
        self.assertEqual(tokenizer.messages[1]['content'], 'x' * 964)
        self.assertTrue(tokenizer.kwargs['add_generation_prompt'])
        self.assertIs(tokenizer.kwargs['return_dict'], False)
        with self.assertRaises(ValueError):
            m.prompts(tokenizer, ['x' * 965], 60)

    def test_installed_transformers_returns_integer_token_ids(self):
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from transformers import PreTrainedTokenizerFast
        except ImportError:
            self.skipTest('install transformers==5.15.0 for tokenizer integration check')
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0, 'hello': 1}, unk_token='[UNK]')),
            unk_token='[UNK]', chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}")
        ids = m.prompts(tokenizer, ['hello'], 60)[0]['prompt_token_ids']
        self.assertIsInstance(ids, list)
        self.assertTrue(ids)
        self.assertTrue(all(type(token) is int for token in ids))

    def test_main_disables_flashinfer_before_provider_load(self):
        import os
        def provider(path):
            self.assertEqual(os.environ['VLLM_USE_FLASHINFER_SAMPLER'], '0')
            return object()
        with patch.object(sys, 'argv', ['provider', '--model', m.MODEL,
                                       '--model-path', '/local/model', '--port', '8000']), \
             patch.dict(os.environ, {'VLLM_USE_FLASHINFER_SAMPLER': '1'}), \
             patch.object(m, 'check_checkpoint'), patch.object(m, 'Provider', side_effect=provider), \
             patch.object(m, 'serve'):
            m.main()

    def test_order_usage_busy_and_error_recovery(self):
        provider = m.Provider.__new__(m.Provider)
        provider.lock = threading.Lock()
        provider.tokenizer = types.SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2])
        class LLM:
            fail = True
            def generate(self, inputs, params, **kwargs):
                if self.fail:
                    self.fail = False
                    raise RuntimeError('GPU error')
                return [types.SimpleNamespace(prompt_token_ids=[1, 2], outputs=[
                    types.SimpleNamespace(text=str(i), token_ids=[3])]) for i in range(len(inputs))]
        provider.llm = LLM()
        payload = {'model': m.MODEL, 'input': ['one', 'two']}
        with patch.dict(sys.modules, {'vllm': types.SimpleNamespace(SamplingParams=lambda **kw: kw)}):
            with self.assertRaises(RuntimeError):
                provider.summarize(payload)
            result = provider.summarize(payload)
            self.assertEqual(result['data'], [{'index': 0, 'summary': '0'}, {'index': 1, 'summary': '1'}])
            self.assertEqual(result['usage'], {'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6})
            provider.lock.acquire()
            try:
                with self.assertRaises(BlockingIOError):
                    provider.summarize(payload)
            finally:
                provider.lock.release()

    def test_selected_vllm_settings(self):
        captured = {}
        def llm(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(get_tokenizer=lambda: object())
        with patch.dict(sys.modules, {'vllm': types.SimpleNamespace(LLM=llm)}):
            m.Provider('/local/Qwen3-4B-W8A8')
        self.assertEqual(captured, {
            'model': '/local/Qwen3-4B-W8A8', 'dtype': 'bfloat16', 'trust_remote_code': False,
            'enable_prefix_caching': True, 'max_model_len': 1024,
            'max_num_batched_tokens': 16384, 'max_num_seqs': 512,
            'speculative_config': None, 'distributed_executor_backend': 'uni',
            'gpu_memory_utilization': 0.92})

    def test_literal_place_metric(self):
        self.assertEqual(m.place_leaks('Name: Arona', 'Based in Arona, Italy.'), ['Italy'])
        self.assertEqual(m.place_leaks('Place: ITALY', 'An Italian shop in Italy.'), ['Italian'])
        self.assertEqual(m.place_leaks('Name: Devia.it', 'An Italian service.'), ['Italian'])
        self.assertEqual(m.place_leaks('Milano', 'A shop in Milan.'), ['Milan'])
        self.assertEqual(m.place_leaks('Roma', 'Roman history and Italianate art.'), [])
        self.assertEqual(m.place_leaks('Place: Roma', 'A shop in Roma.'), [])

    def test_nationality_filter_preserves_supported_facts(self):
        cases = [
            ('Devia', 'Devia is an Italian organization.', 'Devia is an organization.'),
            ('Devia', 'Devia is an italian company.', 'Devia is a company.'),
            ('Devia', 'AN ITALIAN ORGANIZATION.', 'An ORGANIZATION.'),
            ('Devia', 'A Italian academy.', 'An academy.'),
            ('Devia', 'Devia offers support in Italian and English.',
             'Devia offers support in Italian and English.'),
            ('Devia', 'Devia offers Italian lessons.', 'Devia offers Italian lessons.'),
            ('Devia', 'An Italian language course.', 'An Italian language course.'),
            ('Arona', 'A shop in Arona, Italy.', 'A shop in Arona.'),
            ('Devia', 'Based in Italy, Devia blocks calls.', 'Devia blocks calls.'),
            ('Devia', 'Devia blocks calls, based in Italy.', 'Devia blocks calls.'),
            ('Italian Italy', 'An Italian shop in Italy.', 'An Italian shop in Italy.'),
            # Unrecognized clauses remain visible to the independent metric.
            ('Devia', 'Devia is based in Italy.', 'Devia is based in Italy.'),
            ('Devia', 'Devia has offices in Rome.', 'Devia has offices in Rome.'),
        ]
        for text, summary, expected in cases:
            with self.subTest(summary=summary):
                self.assertEqual(m.filter_nationality(text, summary), expected)

    def test_provider_applies_filter_and_keeps_raw_token_usage(self):
        provider = m.Provider.__new__(m.Provider)
        provider.lock = threading.Lock()
        provider.tokenizer = types.SimpleNamespace(apply_chat_template=lambda *a, **kw: [1])
        provider.llm = types.SimpleNamespace(generate=lambda *a, **kw: [
            types.SimpleNamespace(prompt_token_ids=[1], outputs=[
                types.SimpleNamespace(text='Devia is an Italian organization.', token_ids=[2, 3, 4])])])
        with patch.dict(sys.modules, {'vllm': types.SimpleNamespace(SamplingParams=lambda **kw: kw)}):
            result = provider.summarize({'model': m.MODEL, 'input': ['Name: Devia']})
        self.assertEqual(result['data'][0]['summary'], 'Devia is an organization.')
        self.assertEqual(result['usage']['completion_tokens'], 3)

    def test_saved_output_scoring(self):
        sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
        try:
            score_spec = importlib.util.spec_from_file_location(
                'score_summaries', Path(__file__).parents[1] / 'scripts/score-summaries.py')
            scorer = importlib.util.module_from_spec(score_spec)
            score_spec.loader.exec_module(scorer)
        finally:
            sys.path.pop(0)
        report = scorer.score([
            {'input': 'Devia', 'summary': 'Italian Devia in Italy.'},
            {'id': 'ok', 'text': 'Arona', 'summary': 'Shop in Arona.'}])
        self.assertEqual(report['place_leak_count'], 1)
        self.assertEqual(report['place_leak_rate'], .5)
        self.assertEqual(report['place_leaks_by_id'][0]['tokens'], ['Italy', 'Italian'])
        for rows in [[], [{'input': 'x', 'summary': ''}], [{'summary': 'Italy'}]]:
            with self.assertRaises(ValueError):
                scorer.score(rows)

    def test_matrix_cells_with_fake_gpu(self):
        bench_spec = importlib.util.spec_from_file_location(
            'summary_bench', Path(__file__).parents[1] / 'scripts/benchmark-summaries.py')
        bench = importlib.util.module_from_spec(bench_spec)
        bench_spec.loader.exec_module(bench)
        ready = threading.Event()
        seen_prompts = []
        class LLM:
            def __init__(self, **kwargs):
                pass
            def get_tokenizer(self):
                def template(messages, **kwargs):
                    seen_prompts.append(messages[0]['content'])
                    return [1]
                return types.SimpleNamespace(apply_chat_template=template)
            def generate(self, inputs, *args, **kwargs):
                if not ready.wait(5):
                    raise RuntimeError('sampler did not start')
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(
                    text='Devia is an Italian organization.')]) for _ in inputs]
        def nvidia(*args, **kwargs):
            ready.set()
            return types.SimpleNamespace(stdout='1000')
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            source = root / 'cards.jsonl'
            source.write_text(''.join(json.dumps({'id': str(i), 'text': 'Name: Devia'}) + '\n'
                                      for i in range(1000)))
            for variant, expected_prompt in [('baseline', m.PROMPT_V1), ('negative', m.PROMPT),
                                             ('place-field', m.PROMPT_PLACE_FIELD), ('postfilter', m.PROMPT)]:
                target = root / (variant + '.json')
                argv = ['benchmark', '--input', str(source), '--model-path', path,
                        '--max-num-seqs', '512', '--prompt-variant', variant, '--output', str(target)]
                with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {
                    'simple_ai_summary_provider': m,
                    'vllm': types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw, __version__='fake')
                }), patch.object(bench.subprocess, 'run', side_effect=nvidia), patch.object(m, 'check_checkpoint'):
                    bench.main()
                report = json.loads(target.read_text())
                self.assertEqual(report['sites'], 1000)
                self.assertEqual(report['prompt'], expected_prompt)
                self.assertEqual(seen_prompts[-1], expected_prompt)
                self.assertEqual(report['place_leak_rate'], 0 if variant == 'postfilter' else 1)
                self.assertIsNone(report['grounding_score'])
                self.assertEqual(len(Path(str(target) + '.outputs.jsonl').read_text().splitlines()), 1000)

    def test_only_int8_checkpoint_is_accepted(self):
        group = {'targets': ['Linear'], 'weights': {'type': 'int', 'num_bits': 8},
                 'input_activations': {'type': 'int', 'num_bits': 8}}
        valid = {'quantization_config': {'quant_method': 'compressed-tensors',
                                        'config_groups': {'group_0': group}}}
        with tempfile.TemporaryDirectory() as path:
            config_path = Path(path) / 'config.json'
            config_path.write_text(json.dumps(valid))
            m.check_checkpoint(path)
            invalid = [{}, {'quantization_config': {'quant_method': 'awq'}}]
            for key, value in [('type', 'float'), ('num_bits', 4)]:
                data = copy.deepcopy(valid)
                data['quantization_config']['config_groups']['group_0']['weights'][key] = value
                invalid.append(data)
            data = copy.deepcopy(valid)
            del data['quantization_config']['config_groups']['group_0']['input_activations']
            invalid.append(data)
            for data in invalid:
                config_path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    m.check_checkpoint(path)


class CalibrationTests(unittest.TestCase):
    def test_separate_calibration_and_no_overwrite_inputs(self):
        quant_spec = importlib.util.spec_from_file_location(
            'summary_quant', Path(__file__).parents[1] / 'scripts/summary-quantization/quant.py')
        quant = importlib.util.module_from_spec(quant_spec)
        quant_spec.loader.exec_module(quant)
        with tempfile.TemporaryDirectory() as path:
            calibration, heldout = Path(path) / 'cal.jsonl', Path(path) / 'eval.jsonl'
            rows = [{'id': str(i), 'text': f'Card {i}'} for i in range(512)]
            def write(target, records):
                target.write_text(''.join(json.dumps(r) + '\n' for r in records))
            write(calibration, rows)
            write(heldout, [{'id': 'eval', 'text': 'Evaluation site'}])
            self.assertEqual(quant.calibration_rows(calibration, heldout), rows)
            for records in [[rows[0]], [{'id': 'other-id', 'text': rows[0]['text']}], []]:
                write(heldout, records)
                with self.assertRaises(ValueError):
                    quant.calibration_rows(calibration, heldout)
            write(heldout, [{'id': 'eval', 'text': 'Evaluation site'}])
            write(calibration, rows[:-1])
            with self.assertRaises(ValueError):
                quant.calibration_rows(calibration, heldout)
