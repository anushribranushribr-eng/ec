"""
Unit Tests for Audio Preprocessing & Processing Modules.
"""

import unittest
import numpy as np
from deepinfant.audio.preprocessing import pad_or_crop, extract_sliding_windows
from deepinfant.audio.augmentation import pitch_shift, add_background_noise


class TestAudioModules(unittest.TestCase):

    def test_pad_or_crop_padding(self):
        audio = np.ones(8000, dtype=np.float32)
        padded = pad_or_crop(audio, target_length=16000, mode="center")
        self.assertEqual(len(padded), 16000)
        self.assertEqual(np.sum(padded[:4000]), 0.0)
        self.assertEqual(np.sum(padded[4000:12000]), 8000.0)

    def test_pad_or_crop_cropping(self):
        audio = np.ones(32000, dtype=np.float32)
        cropped = pad_or_crop(audio, target_length=16000, mode="center")
        self.assertEqual(len(cropped), 16000)

    def test_extract_sliding_windows(self):
        audio = np.random.randn(32000).astype(np.float32) # 2 seconds at 16kHz
        windows, timestamps = extract_sliding_windows(audio, sr=16000, window_sec=0.975, stride_sec=0.25)
        self.assertTrue(len(windows) > 1)
        self.assertEqual(len(windows[0]), int(16000 * 0.975))

    def test_add_background_noise(self):
        audio = np.zeros(1000, dtype=np.float32)
        noisy = add_background_noise(audio, noise_factor=0.01)
        self.assertEqual(len(noisy), 1000)
        self.assertGreater(np.max(np.abs(noisy)), 0.0)


if __name__ == "__main__":
    unittest.main()
