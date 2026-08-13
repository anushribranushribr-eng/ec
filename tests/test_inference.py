"""
Unit Tests for Predictor, Rejection Thresholding, and Temporal Smoothing.
"""

import unittest
import torch
import numpy as np
from deepinfant.models.cnn_lstm import DeepInfantCNNLSTM
from deepinfant.inference.predictor import CryPredictor
from deepinfant.inference.aggregation import TemporalPredictionSmoother


class TestInference(unittest.TestCase):

    def setUp(self):
        self.config = {
            "audio": {"sample_rate": 16000, "n_mels": 128, "target_duration_sec": 4.0},
            "model": {"num_classes": 9},
            "classes": ["belly_pain", "burping", "cold_hot", "discomfort", "hungry", "lonely", "scared", "tired", "unknown"]
        }
        self.model = DeepInfantCNNLSTM(num_classes=9)
        self.predictor = CryPredictor(self.model, self.config, confidence_threshold=0.50)

    def test_predict_returns_valid_structure(self):
        audio = np.random.randn(16000 * 4).astype(np.float32)
        res = self.predictor.predict_audio_array(audio)
        self.assertIn("prediction", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)
        self.assertEqual(len(res["probabilities"]), 9)
        self.assertIn("safety_disclaimer", res)

    def test_temporal_smoother(self):
        smoother = TemporalPredictionSmoother(alpha=0.5, num_classes=9)
        vector1 = np.ones(9) / 9.0
        smoothed1 = smoother.update(vector1)
        self.assertAlmostEqual(np.sum(smoothed1), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
