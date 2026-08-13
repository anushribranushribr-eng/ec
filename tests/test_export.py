"""
Unit Tests for Core ML Export & Validation.
"""

import os
import unittest
import torch
import shutil
from deepinfant.models.cnn_lstm import DeepInfantCNNLSTM
from deepinfant.export.coreml import export_pytorch_to_coreml, validate_coreml_against_pytorch


class TestExport(unittest.TestCase):

    def setUp(self):
        self.model = DeepInfantCNNLSTM(num_classes=9)
        self.export_path = "tests/test_DeepInfant_V2.mlpackage"

    def tearDown(self):
        if os.path.exists(self.export_path):
            if os.path.isdir(self.export_path):
                shutil.rmtree(self.export_path)
            else:
                os.remove(self.export_path)

    def test_coreml_export_and_validation(self):
        out_path = export_pytorch_to_coreml(self.model, output_path=self.export_path)
        self.assertTrue(os.path.exists(out_path))
        
        report = validate_coreml_against_pytorch(self.model, out_path, test_samples=5)
        self.assertIn("class_agreement_percentage", report)
        self.assertGreaterEqual(report["class_agreement_percentage"], 90.0)


if __name__ == "__main__":
    unittest.main()
