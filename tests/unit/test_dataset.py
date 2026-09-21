import io
import unittest

from PIL import Image

from mmdl.evaluation.datasets.mmmu import separate_sample


class DatasetTests(unittest.TestCase):
    def test_real_image_bytes_gold_separation_and_broken_reference(self):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 5), "red").save(buffer, format="PNG")
        row = dict(id="validation_Accounting_1", question="Synthetic <image 1>",
                   question_type="multiple-choice", options="['one', 'two']",
                   answer="B", explanation="PRIVATE GOLD", image_1={"bytes": buffer.getvalue()})
        sample, images, gold = separate_sample(row, "Accounting")
        self.assertNotIn("answer", sample)
        self.assertNotIn("explanation", sample)
        self.assertEqual(gold, {"answer": "B", "gold_explanation": "PRIVATE GOLD"})
        self.assertEqual(images[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(sample["image_references"][0]["content_index"], 0)
        with self.assertRaises(ValueError):
            separate_sample(row | {"question": "Missing <image 2>"}, "Accounting")
        with self.assertRaises(ValueError):
            separate_sample(row | {"question_type": "unknown"}, "Accounting")


if __name__ == "__main__":
    unittest.main()
