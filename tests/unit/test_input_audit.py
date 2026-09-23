import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mmdl.evaluation.input_audit import _prepare_sample, audit_inputs


class InputAuditTests(unittest.TestCase):
    def test_audit_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with self.assertRaises(FileExistsError):
                audit_inputs('unused', output, output)

    def test_prepare_sample_records_stable_id(self):
        class FakeImage:
            size = (2, 3)
            def convert(self, _mode): return self
            def tobytes(self): return b'pixels'

        def fake_prepare(*_args):
            return ({'input_tokens': 7, 'prompt_token_ids': [1, 2],
                     'input_tensors': {'pixel_values': {'sha256': 'p'}},
                     'image_grid_thw': [[1, 1, 1]], 'input_sha256': 'i'}, [FakeImage()])

        sample = {'id': 'x', 'images': ['images/x.png'], 'image_references': [],
                  'question': 'q', 'options': ['a'], 'question_type': 'multiple-choice'}
        processor = SimpleNamespace(image_processor=SimpleNamespace(size={}))
        with patch('mmdl.evaluation.input_audit.prepare_protocol_inputs', fake_prepare), \
                patch('mmdl.evaluation.input_audit.resolve_image', return_value=Path('x.png')), \
                patch('mmdl.evaluation.input_audit.Image.open', return_value=FakeImage()):
            result = _prepare_sample(sample, Path('/tmp'), processor, object(),
                                     {"min_pixels": 256, "max_pixels": 1024})
        self.assertEqual(result['id'], 'x')
        self.assertEqual(result['input_tokens'], 7)
        self.assertEqual(processor.image_processor.size, {"shortest_edge": 256, "longest_edge": 1024})


if __name__ == '__main__':
    unittest.main()
