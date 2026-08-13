"""
Unit Tests for DeepInfant CNN-LSTM Model Architecture.
"""

import unittest
import torch
from deepinfant.models.cnn_lstm import DeepInfantCNNLSTM


class TestModelArchitecture(unittest.TestCase):

    def test_cnn_lstm_forward_shape(self):
        batch_size = 4
        n_mels = 128
        time_steps = 125 # ~4 seconds
        dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)
        
        model = DeepInfantCNNLSTM(num_classes=9)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
            
        self.assertEqual(output.shape, (batch_size, 9))

    def test_softmax_probabilities_sum(self):
        dummy_input = torch.randn(2, 1, 128, 125)
        model = DeepInfantCNNLSTM(num_classes=9)
        probs = model.get_probabilities(dummy_input)
        sums = torch.sum(probs, dim=-1)
        
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
