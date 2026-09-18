"""Contract checks without model downloads: python3 -m unittest discover -s tests -p test_extraction_provider.py."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('extraction_provider', Path(__file__).resolve().parents[1] / 'scripts/simple_ai_extraction_provider.py')
provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provider)

class ExtractionValidation(unittest.TestCase):
    def setUp(self):
        self.payload = {'model': provider.MODEL_ID, 'input': 'Zoë lavora a Zürich', 'schema': {'entities': ['person', 'location']}}

    def test_single_batch_and_unicode_limits(self):
        self.assertEqual(provider.validate(self.payload), ['Zoë lavora a Zürich'])
        self.payload['input'] = ['😀' * 1500, 'Italiano']
        self.assertEqual(len(provider.validate(self.payload)), 2)
        self.payload['input'][0] += '😀'
        with self.assertRaises(ValueError): provider.validate(self.payload)
        self.payload['long_text'] = True
        self.assertEqual(len(provider.validate(self.payload)), 2)

    def test_rejects_invalid_or_unbounded_work(self):
        for key, value in [('input', []), ('input', ['ok', ' ']), ('input', ['ok'] * 17), ('threshold', float('nan')), ('threshold', 1.1), ('threshold', True), ('chunk_size', 0), ('chunk_overlap', 256), ('splitter', 'guess'), ('schema', {}), ('surprise', 1)]:
            with self.subTest(key=key, value=value):
                p = copy.deepcopy(self.payload);p[key] = value
                with self.assertRaises(ValueError): provider.validate(p)
        p = copy.deepcopy(self.payload);p.update(long_text=True, chunk_overlap=255)
        with self.assertRaises(ValueError): provider.validate(p)

    def test_rejects_duplicate_reserved_and_unknown_schema_fields(self):
        for schema in [{'entities': ['person', 'person']}, {'unknown': ['person']}, {'classifications': [{'task': 'entities', 'labels': ['yes', 'no']}]}, {'structures': {'purchase': {'anchor': 'missing', 'fields': [{'name': 'buyer'}]}}}]:
            with self.subTest(schema=schema):
                self.payload['schema'] = schema
                with self.assertRaises(ValueError): provider.validate(self.payload)

    def test_accepts_combined_tasks_and_records(self):
        self.payload['schema'].update(classifications=[{'task': 'sentiment', 'labels': ['positive', 'negative']}], relations={'works_for': 'Employment relationship'}, structures={'purchase': {'anchor': 'buyer', 'fields': [{'name': 'buyer', 'cardinality': 'required_one'}, {'name': 'item', 'dtype': 'list'}]}})
        self.assertEqual(len(provider.validate(self.payload)), 1)

if __name__ == '__main__': unittest.main()
