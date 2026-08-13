"""
Unit Tests for STFT and 128-bin Mel-Spectrogram Extraction.
"""

import unittest
import numpy as np
from deepinfant.audio.features import compute_stft, compute_mel_spectrogram


class TestFeatures(unittest.TestCase):

    def test_stft_dimensions(self):
        sr = 16000
        audio = np.random.randn(sr * 4).astype(np.float32) # 4 seconds
        stft_mag = compute_stft(audio, n_fft=2048, hop_length=512)
        # Expected frequency bins: 2048 // 2 + 1 = 1025
        self.assertEqual(stft_mag.shape[0], 1025)
        self.assertTrue(stft_mag.shape[1] > 100)

    def test_mel_spectrogram_dimensions(self):
        sr = 16000
        audio = np.random.randn(sr * 4).astype(np.float32)
        log_mel = compute_mel_spectrogram(
            audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128, log_scale=True
        )
        self.assertEqual(log_mel.shape[0], 128)
        self.assertEqual(log_mel.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
