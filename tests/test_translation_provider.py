import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('translation', Path(__file__).parents[1] / 'scripts/simple_ai_translation_provider.py')
provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provider)

class TranslationTests(unittest.TestCase):
    def test_validation(self):
        valid = {'input': ['Ciao'], 'target': 'zh-Hans', 'source': 'ita'}
        self.assertEqual(provider.validate(valid), (['Ciao'], 'zh', 256))
        for patch in [{'input': []}, {'input': ['a'] * 257}, {'input': ['a' * 4001]}, {'target': 'bogus'}, {'max_tokens': 0}, {'input': [' ']}]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                provider.validate(valid | patch)

    def test_batch_order_and_prefix(self):
        calls = []
        def generate(texts, limit):
            calls.append((texts, limit))
            return [t.removeprefix('<2de> ') + '-translated' for t in texts]
        result = provider.translate_batches(['uno', 'due', 'tre', 'quattro', 'cinque'], 'de', 128, 2, generate)
        self.assertEqual([r['index'] for r in result], list(range(5)))
        self.assertEqual([r['translation'] for r in result], [t + '-translated' for t in ['uno', 'due', 'tre', 'quattro', 'cinque']])
        self.assertEqual([len(c[0]) for c in calls], [2, 2, 1])
        self.assertEqual(calls[0], (['<2de> uno', '<2de> due'], 128))

if __name__ == '__main__':
    unittest.main()
