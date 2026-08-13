"""
Audio Data Augmentation Module.
Training-only audio augmentations: pitch shifting, time stretching, background noise, specaugment.
"""

import numpy as np
import librosa


def pitch_shift(audio: np.ndarray, sr: int = 16000, n_steps: float = 0.0) -> np.ndarray:
    """Shifts pitch by n_steps semitones (-2 to +2)."""
    if abs(n_steps) < 1e-3:
        return audio
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def time_stretch(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Stretches or compresses audio duration (0.9x to 1.1x)."""
    if abs(rate - 1.0) < 1e-3:
        return audio
    return librosa.effects.time_stretch(audio, rate=rate)


def add_background_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Adds white noise / room ambience to waveform."""
    if noise_factor <= 0:
        return audio
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    # Maintain [-1.0, 1.0] bound
    max_val = np.max(np.abs(augmented))
    if max_val > 1.0:
        augmented = augmented / max_val
    return augmented.astype(np.float32)


def apply_spec_augment(
    mel_spectrogram: np.ndarray,
    freq_mask_param: int = 15,
    time_mask_param: int = 35
) -> np.ndarray:
    """
    Applies SpecAugment (frequency masking and time masking) on 2D Mel-Spectrogram [n_mels, time].
    """
    augmented = mel_spectrogram.copy()
    num_mels, num_time = augmented.shape
    
    # Frequency mask
    if freq_mask_param > 0 and num_mels > freq_mask_param:
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, num_mels - f)
        augmented[f0 : f0 + f, :] = 0.0
        
    # Time mask
    if time_mask_param > 0 and num_time > time_mask_param:
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, num_time - t)
        augmented[:, t0 : t0 + t] = 0.0
        
    return augmented
