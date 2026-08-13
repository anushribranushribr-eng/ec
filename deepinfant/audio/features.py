"""
Acoustic Feature Extraction Module.
Computes STFT and 128-bin Mel-Spectrogram with logarithmic normalization.
"""

import numpy as np
import librosa
from typing import Tuple


def compute_stft(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann"
) -> np.ndarray:
    """
    Computes Short-Time Fourier Transform (STFT) magnitude representation.
    
    Formula: X(f,t) = \sum x(n) w(n-t) exp(-j 2\pi f n / N)
    
    Returns:
        stft_mag: Magnitude spectrum of shape [(n_fft // 2) + 1, time_steps]
    """
    stft_complex = librosa.stft(
        audio, n_fft=n_fft, hop_length=hop_length, window=window
    )
    stft_mag = np.abs(stft_complex)
    return stft_mag


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 50.0,
    fmax: float = 8000.0,
    log_scale: bool = True,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Converts time-domain audio waveform into a 128-bin Mel-spectrogram with Log normalization.
    
    Args:
        audio: 1D float32 waveform (16,000 Hz)
        sr: Sample rate (16000)
        n_fft: FFT window size (2048)
        hop_length: Hop length (512)
        n_mels: Number of Mel filterbank bins (128)
        fmin: Minimum frequency (50 Hz)
        fmax: Maximum frequency (8000 Hz)
        log_scale: Apply log(mel + eps) transformation
        
    Returns:
        log_mel: 2D array of shape [n_mels, time_steps]
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )
    
    if log_scale:
        log_mel = np.log(mel_spec + epsilon)
        # Normalize to mean 0, std 1
        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-8
        log_mel = (log_mel - mean) / std
        return log_mel.astype(np.float32)
        
    return mel_spec.astype(np.float32)
