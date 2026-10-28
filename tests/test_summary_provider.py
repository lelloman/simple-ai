import importlib.util
from pathlib import Path
import sys
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
