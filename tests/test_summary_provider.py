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
        with self.assertRaises(ValueError):
            m.prompts(tokenizer, ['x' * 965], 60)

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
